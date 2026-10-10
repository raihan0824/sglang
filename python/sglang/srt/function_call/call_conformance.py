"""Warn about parsed tool calls that would fail OpenRouter's tool-call checks.

OpenRouter scores every call: the name must be a declared tool (Unknown Name),
the arguments must be strict JSON (Invalid JSON) and must validate against the
tool's parameters schema (Schema Mismatch). A failing call logs one WARNING that
says where it breaks -- tool name, mode, JSON path, violated keyword and its
schema value, the JSON type found -- and never an argument value.
"""

import json
import logging
from functools import lru_cache
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool

logger = logging.getLogger(__name__)

# Arbitrary; tool sets repeat per client, so a small cache keeps validation cheap.
_VALIDATOR_CACHE_SIZE = 256
_MAX_ERRORS = 5


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


@lru_cache(maxsize=_VALIDATOR_CACHE_SIZE)
def _validator(schema_json: str):
    from jsonschema import Draft202012Validator

    return Draft202012Validator(json.loads(schema_json))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def describe_schema_error(error) -> str:
    path = "$" + "".join(
        f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path
    )
    # Key names only: the messages of these validators carry no argument values.
    if error.validator in ("required", "additionalProperties"):
        return f"{path} {error.message[:200]}"
    expected = json.dumps(error.validator_value, ensure_ascii=False, default=str)
    return f"{path} {error.validator}={expected[:120]} got={_json_type(error.instance)}"


def log_call_conformance(
    prefix: str,
    func_name: Optional[str],
    arguments: str,
    tools: Optional[List[Tool]],
    mode: str,
) -> None:
    """Log one WARNING if the call would fail OpenRouter's checks; never raises."""
    try:
        _log_call_conformance(prefix, func_name, arguments, tools or [], mode)
    except Exception as e:
        logger.debug("%s conformance check skipped: %r", prefix, e)


def _log_call_conformance(
    prefix: str, func_name: Optional[str], arguments: str, tools: List[Tool], mode: str
) -> None:
    params = {t.function.name: t.function.parameters for t in tools}
    if func_name not in params:
        logger.warning(
            "%s name not in tools: name=%r mode=%s tools=%d",
            prefix,
            (func_name or "")[:64],
            mode,
            len(params),
        )
        return
    try:
        args = json.loads(arguments, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        logger.warning(
            "%s arguments are not valid JSON: tool=%s mode=%s chars=%d",
            prefix,
            func_name,
            mode,
            len(arguments),
        )
        return
    schema = params[func_name]
    if not isinstance(schema, dict):
        return
    if not isinstance(args, dict):
        logger.warning(
            "%s fails its schema: tool=%s mode=%s errors=$ got=%s",
            prefix,
            func_name,
            mode,
            _json_type(args),
        )
        return
    validator = _validator(json.dumps(schema, sort_keys=True, ensure_ascii=False))
    errors = []
    for error in validator.iter_errors(args):
        errors.append(describe_schema_error(error))
        if len(errors) >= _MAX_ERRORS:
            break
    if errors:
        logger.warning(
            "%s fails its schema: tool=%s mode=%s errors=%s",
            prefix,
            func_name,
            mode,
            "; ".join(errors),
        )
