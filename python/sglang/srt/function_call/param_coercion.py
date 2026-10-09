"""Schema-aware coercion of raw-text tool-call parameter values.

XML-style tool formats (``<parameter=name>value</parameter>``) carry every
value as text; the JSON type comes from the tool's ``parameters`` schema.
Reading the schema follows vLLM's ``rust/src/parser/src/schema.rs``: local
``$ref`` and single-schema ``allOf`` are followed, ``anyOf``/``oneOf`` and type
arrays give one option per alternative in schema order, ``const``/``enum``
values are literal options. Unlike vLLM, OpenAPI's ``"nullable": true`` adds no
null option: JSON Schema validators reject the null, so an optional parameter
spelled null is dropped instead.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.function_call.utils import (
    _normalize_single_type,
    get_schema_properties,
    safe_literal_eval,
)

# Bound on $ref / combinator nesting; deeper schemas accept any value.
_MAX_SCHEMA_DEPTH = 32

_TYPE_NAMES = ("string", "integer", "number", "boolean", "null", "object", "array")
_INT_RE = re.compile(r"[+-]?\d+")

# Returned when a value spells null for a parameter whose schema has no null
# option; the caller drops the key when the parameter is optional.
OMIT = object()

# An option is (kind, payload): kind is a JSON type name with payload None, or
# "literal" with the non-string const/enum value as payload.
Option = Tuple[str, Any]


def resolve_schema(root: Any, schema: Any) -> Any:
    """Follow ``$ref`` and single-schema ``allOf`` until a concrete schema."""
    for _ in range(_MAX_SCHEMA_DEPTH):
        if not isinstance(schema, dict):
            return schema
        ref = schema.get("$ref")
        if isinstance(ref, str):
            schema = _resolve_ref(root, ref)
            if schema is None:
                return {}
            continue
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1:
            schema = all_of[0]
            continue
        return schema
    return {}


def _resolve_ref(root: Any, ref: str) -> Any:
    if not ref.startswith("#"):
        return None
    node = root
    for part in ref[1:].split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def schema_options(root: Any, schema: Any) -> Optional[List[Option]]:
    """Options a value under ``schema`` may take, or None when unconstrained."""
    options = _options_at(root, schema, 0, set())
    if options is None:
        return None
    deduped: List[Option] = []
    for option in options:
        if option not in deduped:
            deduped.append(option)
    # Text spelling a non-string literal decodes to it before any type does.
    deduped.sort(key=lambda option: option[0] != "literal")
    return deduped or None


def _options_at(
    root: Any, schema: Any, depth: int, expanded: set
) -> Optional[List[Option]]:
    if depth > _MAX_SCHEMA_DEPTH or not isinstance(schema, dict):
        return None
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_ref(root, ref)
        if target is None or id(target) in expanded:
            return None
        return _options_at(root, target, depth + 1, expanded | {id(target)})
    if "const" in schema:
        return [_literal_option(schema["const"])]
    if isinstance(schema.get("enum"), list):
        return [_literal_option(value) for value in schema["enum"]]
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            options: List[Option] = []
            for branch in branches:
                branch_options = _options_at(root, branch, depth + 1, expanded)
                if branch_options is None:
                    return None
                options.extend(branch_options)
            return options
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        # Several schemas: the first one that constrains the type stands for all.
        for branch in all_of:
            branch_options = _options_at(root, branch, depth + 1, expanded)
            if branch_options is not None:
                return branch_options
        return None
    declared = schema.get("type")
    if isinstance(declared, str):
        declared = [declared]
    if isinstance(declared, list):
        kinds = [_normalize_single_type(t) for t in declared if isinstance(t, str)]
        kinds = [k for k in kinds if k in _TYPE_NAMES]
        return [(k, None) for k in kinds] or None
    if "properties" in schema or "additionalProperties" in schema:
        return [("object", None)]
    if "items" in schema:
        return [("array", None)]
    return None


def _literal_option(value: Any) -> Option:
    if isinstance(value, str):
        return ("string", None)
    return ("literal", value)


def is_null_spelling(value: str) -> bool:
    return value.strip().lower() in ("null", "none")


def coerce_value(raw: str, options: Optional[List[Option]]) -> Any:
    """Convert raw parameter text to the first option it fits.

    Returns ``OMIT`` for a null spelling the options do not allow, and the raw
    text when nothing fits (or the schema does not constrain the value).
    """
    if options is None:
        return None if is_null_spelling(raw) else raw
    kinds = [kind for kind, _ in options]
    if is_null_spelling(raw) and kinds != ["string"]:
        if "null" in kinds:
            return None
        if "string" not in kinds:
            return OMIT
    for kind, payload in options:
        ok, value = _convert(raw, kind, payload)
        if ok:
            return value
    return raw


def _convert(raw: str, kind: str, payload: Any) -> Tuple[bool, Any]:
    text = raw.strip()
    if kind == "string":
        return True, raw
    if kind == "null":
        return is_null_spelling(text), None
    if kind == "integer":
        return _to_integer(text)
    if kind == "number":
        return _to_number(text)
    if kind == "boolean":
        lowered = _unquote(text).lower()
        if lowered in ("true", "1"):
            return True, True
        if lowered in ("false", "0"):
            return True, False
        return False, None
    if kind == "object":
        if not text:
            return True, {}
        return _decode_container(text, dict)
    if kind == "array":
        if not text:
            return True, []
        return _decode_container(text, (list, tuple))
    if kind == "literal":
        ok, value = _json_loads(text)
        if ok and type(value) is type(payload) and value == payload:
            return True, value
        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            ok, value = _to_number(text)
            if ok and value == payload:
                return True, payload
        return False, None
    return False, None


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def _without_digit_separators(text: str) -> str:
    if "_" in text and re.fullmatch(r"[+-]?\d+(_\d+)*(\.\d+)?", text):
        return text.replace("_", "")
    return text


def _to_integer(text: str) -> Tuple[bool, Any]:
    text = _without_digit_separators(_unquote(text))
    if _INT_RE.fullmatch(text):
        return True, int(text)
    ok, number = _to_number(text)
    if ok and float(number).is_integer():
        return True, int(number)
    return False, None


def _to_number(text: str) -> Tuple[bool, Any]:
    text = _without_digit_separators(_unquote(text))
    if _INT_RE.fullmatch(text):
        return True, int(text)
    try:
        number = float(text)
    except ValueError:
        return False, None
    if not math.isfinite(number):
        return False, None
    return True, number


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _json_loads(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        return False, None


def _decode_container(text: str, types) -> Tuple[bool, Any]:
    ok, value = _json_loads(text)
    if ok and isinstance(value, types):
        return True, list(value) if isinstance(value, tuple) else value
    try:
        value = safe_literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None
    if not isinstance(value, types):
        return False, None
    if isinstance(value, tuple):
        value = list(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False, None
    return True, value


class ToolParamSchema:
    """One tool's parameter schemas, resolved once per call."""

    def __init__(self, parameters: Any):
        self.root = parameters if isinstance(parameters, dict) else {}
        resolved = resolve_schema(self.root, self.root)
        resolved = resolved if isinstance(resolved, dict) else {}
        properties = resolved.get("properties")
        if not isinstance(properties, dict):
            properties = get_schema_properties(resolved)
        self.properties: Dict[str, Any] = properties
        required = resolved.get("required")
        self.required = set(required) if isinstance(required, list) else set()

    def convert(self, name: str, raw: str) -> Any:
        """Coerced value of parameter ``name``, or ``OMIT`` to drop the key."""
        if name not in self.properties:
            return None if is_null_spelling(raw) else raw
        value = coerce_value(raw, schema_options(self.root, self.properties[name]))
        if value is OMIT and name in self.required:
            return None
        return value
