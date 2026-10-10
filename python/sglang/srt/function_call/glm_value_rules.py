"""Typed EBNF for GLM tool-call argument values.

GLM writes every value as text inside <arg_value>...</arg_value>: strings raw,
everything else as JSON. The rules here type a value by its parameter schema
(local $ref and single allOf followed, unions as alternatives, enums exact,
integers as integers, array items typed). Nested objects stay any-order JSON
objects: a grammar that writes properties in schema order makes the model drop
or repeat arguments. Numeric ranges and item counts are not enforced: enforced
digit by digit they cut the model's value short (7200 above maximum 3600
becomes 720).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

TEXT_RULE = "text_without_special_tokens"
# Arbitrary bounds; deeper schemas fall back to any JSON value.
_MAX_DEPTH = 8
_TYPE_RULES = {
    "string": "basic_string",
    "integer": "basic_integer",
    "number": "basic_number",
    "boolean": "basic_boolean",
    "null": "basic_null",
    "object": "basic_object",
}
_STANDARD_TYPES = set(_TYPE_RULES) | {"array"}


def _ebnf_literal(text: str) -> str:
    return '"' + json.dumps(text, ensure_ascii=False)[1:-1] + '"'


def resolve(root: Any, schema: Any) -> Any:
    """Follow local $ref and single-schema allOf."""
    for _ in range(_MAX_DEPTH):
        if not isinstance(schema, dict):
            return schema
        ref = schema.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#"):
                return {}
            node = root
            for part in ref[1:].split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return {}
            schema = node
            continue
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1:
            schema = all_of[0]
            continue
        return schema
    return {}


def _types(root: Any, schema: Any, depth: int = 0) -> Optional[set]:
    """JSON types a value may take, or None when the schema does not constrain it."""
    schema = resolve(root, schema)
    if depth > _MAX_DEPTH or not isinstance(schema, dict):
        return None
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            kinds = set()
            for branch in branches:
                branch_kinds = _types(root, branch, depth + 1)
                if branch_kinds is None:
                    return None
                kinds |= branch_kinds
            return kinds
    values = [schema["const"]] if "const" in schema else schema.get("enum")
    if isinstance(values, list) and values:
        return {"string" if isinstance(v, str) else "json" for v in values}
    declared = schema.get("type")
    if isinstance(declared, str):
        declared = [declared]
    if isinstance(declared, list):
        kinds = {t for t in declared if t in _STANDARD_TYPES}
        return kinds or None
    if "properties" in schema or "additionalProperties" in schema:
        return {"object"}
    if "items" in schema:
        return {"array"}
    return None


class GlmValueRules:
    """Collects the EBNF rules typing one tool's argument values."""

    def __init__(self, root: Any, prefix: str):
        self.root = root if isinstance(root, dict) else {}
        self.prefix = prefix
        self.rules: List[str] = []
        self._count = 0

    def _rule(self, body: str) -> str:
        name = f"{self.prefix}_{self._count}"
        self._count += 1
        self.rules.append(f"{name} ::= {body}")
        return name

    def top(self, schema: Any) -> str:
        """Rule for the raw text of one <arg_value>."""
        resolved = resolve(self.root, schema)
        if isinstance(resolved, dict):
            values = (
                [resolved["const"]] if "const" in resolved else resolved.get("enum")
            )
            if isinstance(values, list) and values:
                return (
                    "("
                    + " | ".join(
                        _ebnf_literal(v if isinstance(v, str) else json.dumps(v))
                        for v in values
                    )
                    + ")"
                )
        kinds = _types(self.root, schema)
        if kinds is None or "string" in kinds:
            return TEXT_RULE
        return self.json(schema, 0)

    def json(self, schema: Any, depth: int) -> str:
        """Rule for a JSON value under ``schema``."""
        schema = resolve(self.root, schema)
        if depth > _MAX_DEPTH or not isinstance(schema, dict):
            return "basic_any"
        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if isinstance(branches, list) and branches:
                alts = [self.json(b, depth + 1) for b in branches]
                return "(" + " | ".join(dict.fromkeys(alts)) + ")"
        values = [schema["const"]] if "const" in schema else schema.get("enum")
        if isinstance(values, list) and values:
            return (
                "("
                + " | ".join(
                    _ebnf_literal(json.dumps(v, ensure_ascii=False)) for v in values
                )
                + ")"
            )
        declared = schema.get("type")
        if isinstance(declared, str):
            declared = [declared]
        if not isinstance(declared, list):
            if "properties" in schema or "additionalProperties" in schema:
                declared = ["object"]
            elif "items" in schema:
                declared = ["array"]
            else:
                return "basic_any"
        alts = []
        for kind in declared:
            if kind == "array":
                alts.append(self._array(schema, depth))
            elif kind in _TYPE_RULES:
                alts.append(_TYPE_RULES[kind])
        if not alts:
            return "basic_any"
        return (
            alts[0] if len(alts) == 1 else "(" + " | ".join(dict.fromkeys(alts)) + ")"
        )

    def _array(self, schema: Dict[str, Any], depth: int) -> str:
        items = schema.get("items")
        if not isinstance(items, dict):
            return "basic_array"
        item = self.json(items, depth + 1)
        if item == "basic_any":
            return "basic_array"
        return self._rule(f'"[" ws ( {item} ( ws "," ws {item} )* )? ws "]"')


def tool_properties(params: Any) -> Tuple[Dict[str, Any], List[str], bool]:
    """(property schemas, top-level required names, extra keys allowed) of a tool.

    A root $ref is followed and top-level anyOf/oneOf/allOf branches are merged;
    a property declared differently by two branches takes either schema. Returns
    no properties when the schema cannot list them (patternProperties, if/then,
    dynamic refs), so the caller keeps keys free.
    """
    root = params if isinstance(params, dict) else {}
    schema = resolve(root, root)
    if not isinstance(schema, dict):
        return {}, [], True
    collected: Dict[str, List[Any]] = {}
    opaque = False
    extra_allowed = False

    def collect(node: Any, depth: int) -> None:
        nonlocal opaque, extra_allowed
        node = resolve(root, node)
        if depth > _MAX_DEPTH or not isinstance(node, dict):
            opaque = True
            return
        if any(
            k in node
            for k in ("$dynamicRef", "patternProperties", "dependentSchemas", "if")
        ):
            opaque = True
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, prop in properties.items():
                collected.setdefault(name, [])
                if prop not in collected[name]:
                    collected[name].append(prop)
        branches = [
            branch
            for keyword in ("allOf", "anyOf", "oneOf")
            for branch in node.get(keyword) or []
        ]
        if any(
            node.get(k) not in (None, False)
            for k in ("additionalProperties", "unevaluatedProperties")
        ):
            extra_allowed = True
        elif (
            not branches
            and not properties
            and node.get("additionalProperties") is not False
        ):
            # A branch that lists no keys accepts any key.
            extra_allowed = True
        for branch in branches:
            collect(branch, depth + 1)

    collect(schema, 0)
    if opaque or not collected:
        return {}, [], True
    properties = {
        name: schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
        for name, schemas in collected.items()
    }
    required = [r for r in schema.get("required") or [] if r in properties]
    return properties, required, extra_allowed
