"""Any-order typed structural tag for Qwen3-Coder XML tool calls.

xgrammar's builtin qwen_3_coder tag either leaves parameters unconstrained
(non-strict tools) or forces them into the schema's property order (strict),
which Qwen3.8 does not follow: it then drops parameters or repeats the call
until max_tokens. This tag keeps the model's own order and constrains what
OpenRouter checks: the tool name is a declared tool, parameter names are
declared, and every value has its declared JSON type.

Call shape (the chat template's example):
    <tool_call>\\n<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>
"""

import itertools
from typing import Any, Dict, List, Optional

from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OrFormat,
    SequenceFormat,
    StarFormat,
    StructuralTag,
    TagFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.param_coercion import resolve_schema, schema_options
from sglang.srt.function_call.utils import get_schema_properties

TOOL_CALL_TRIGGER = "<tool_call>\n<function="
THINK_EXCLUDES = ["<think>", "</think>"]
# Required parameters are enforced in any order by one alternative per
# permutation; arbitrary cap to keep the grammar small (3! = 6 alternatives).
MAX_ENFORCED_REQUIRED = 3


# Arbitrary bound on nesting when loosening a value schema.
_MAX_LOOSE_DEPTH = 8


def _loose_schema(root: Dict[str, Any], schema: Any, depth: int = 0) -> Dict[str, Any]:
    """The value's types, enums and array item types; objects become any-order objects.

    xgrammar's JSON schema rule writes object properties in schema order, and
    Qwen3.8 does not follow it inside values either.
    """
    resolved = resolve_schema(root, schema)
    if depth > _MAX_LOOSE_DEPTH or not isinstance(resolved, dict):
        return {}
    out: Dict[str, Any] = {}
    for keyword in ("anyOf", "oneOf"):
        branches = resolved.get(keyword)
        if isinstance(branches, list):
            out[keyword] = [_loose_schema(root, b, depth + 1) for b in branches]
    # Types and enums only: a digit-by-digit range or an item-count limit cuts a
    # value short (7200 above maximum 3600 decoded as 720), changing what the
    # model meant; out-of-range values stay visible as schema errors instead.
    for keyword in ("type", "enum", "const"):
        if keyword in resolved:
            out[keyword] = resolved[keyword]
    if "type" not in out and (
        "properties" in resolved or "additionalProperties" in resolved
    ):
        out["type"] = "object"
    items = resolved.get("items")
    if isinstance(items, dict):
        out["items"] = _loose_schema(root, items, depth + 1)
    if "type" not in out and isinstance(items, dict):
        out["type"] = "array"
    return out


def _value_format(root: Dict[str, Any], schema: Any):
    """Format of one parameter value, as the qwen3_coder parser reads it back."""
    resolved = resolve_schema(root, schema)
    if isinstance(resolved, dict):
        enum = resolved.get("enum")
        if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
            return OrFormat(elements=[ConstStringFormat(value=v) for v in enum])
    options = schema_options(root, schema)
    if options is None or any(kind == "string" for kind, _ in options):
        # Strings are raw text; unconstrained schemas accept anything.
        return AnyTextFormat()
    kinds = {kind for kind, _ in options}
    if kinds <= {"boolean", "null"}:
        # The parser also reads Python spellings, which Qwen3.8 writes.
        spellings = ["true", "false", "True", "False"] if "boolean" in kinds else []
        if "null" in kinds:
            spellings += ["null", "None"]
        return OrFormat(elements=[ConstStringFormat(value=v) for v in spellings])
    return JSONSchemaFormat(json_schema=_loose_schema(root, schema), style="json")


def _param_tag(root: Dict[str, Any], name: str, schema: Any) -> TagFormat:
    return TagFormat(
        begin=f"<parameter={name}>\n",
        content=_value_format(root, schema),
        end="\n</parameter>\n",
    )


def _params_format(parameters: Any):
    root = parameters if isinstance(parameters, dict) else {}
    resolved = resolve_schema(root, root)
    resolved = resolved if isinstance(resolved, dict) else {}
    properties = resolved.get("properties")
    if not isinstance(properties, dict):
        properties = get_schema_properties(resolved)
    if not properties:
        return AnyTextFormat()
    tags = {name: _param_tag(root, name, schema) for name, schema in properties.items()}
    any_param = StarFormat(content=OrFormat(elements=list(tags.values())))
    required = [name for name in resolved.get("required") or [] if name in tags]
    if not required or len(required) > MAX_ENFORCED_REQUIRED:
        return any_param
    alternatives = []
    for order in itertools.permutations(required):
        elements = [any_param]
        for name in order:
            elements += [tags[name], any_param]
        alternatives.append(SequenceFormat(elements=elements))
    return (
        alternatives[0] if len(alternatives) == 1 else OrFormat(elements=alternatives)
    )


def _tool_tag(tool: Tool) -> TagFormat:
    return TagFormat(
        begin=f"{TOOL_CALL_TRIGGER}{tool.function.name}>\n",
        content=_params_format(tool.function.parameters),
        end="</function>\n</tool_call>",
    )


def build_auto_tool_call_tag(
    tools: Optional[List[Tool]],
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> Optional[StructuralTag]:
    """Free text, plus any-order typed tool calls once the model opens one."""
    tools = [t for t in tools or [] if t.function.name]
    if not tools:
        return None
    suffix = TriggeredTagsFormat(
        triggers=[TOOL_CALL_TRIGGER],
        tags=[_tool_tag(t) for t in tools],
        excludes=THINK_EXCLUDES,
        stop_after_first=not parallel_tool_calls,
    )
    if not thinking_mode:
        return StructuralTag(format=suffix)
    prefix = SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
            ConstStringFormat(value="\n\n"),
        ]
    )
    return StructuralTag(format=SequenceFormat(elements=[prefix, suffix]))
