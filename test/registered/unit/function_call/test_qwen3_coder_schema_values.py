import json
import unittest

import jsonschema

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.param_coercion import (
    OMIT,
    ToolParamSchema,
    coerce_value,
    schema_options,
)
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EDIT_DEFS = {
    "Edit": {
        "type": "object",
        "properties": {
            "old": {"type": "string"},
            "new": {"type": "string"},
            "all": {"type": "boolean"},
        },
        "required": ["old", "new"],
    }
}


def _tool(properties, required=(), **extra):
    params = {"type": "object", "properties": properties, "required": list(required)}
    params.update(extra)
    return Tool(type="function", function=Function(name="f", parameters=params))


def _call(**params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function=f>\n{body}</function>\n</tool_call>"


def _stream(tools, text, step):
    detector = Qwen3CoderDetector()
    args = ""
    for i in range(0, len(text), step):
        for call in detector.parse_streaming_increment(text[i : i + step], tools).calls:
            args += call.parameters or ""
    for call in detector.finish(tools).calls:
        args += call.parameters or ""
    return json.loads(args)


class TestQwen3CoderSchemaValues(unittest.TestCase):
    """Arguments must validate against the declared schema, streamed or not."""

    def assert_args(self, tool, text, expected):
        non_stream = Qwen3CoderDetector().detect_and_parse(text, [tool])
        self.assertEqual(len(non_stream.calls), 1)
        args = json.loads(non_stream.calls[0].parameters)
        self.assertEqual(args, expected)
        for step in (1, 7):
            self.assertEqual(_stream([tool], text, step), expected)
        jsonschema.validate(args, tool.function.parameters)

    def test_ref_object_is_decoded(self):
        tool = _tool({"e": {"$ref": "#/$defs/Edit"}}, ["e"], **{"$defs": EDIT_DEFS})
        text = _call(e='{"old": "a", "new": "b", "all": true}')
        self.assert_args(tool, text, {"e": {"old": "a", "new": "b", "all": True}})

    def test_nullable_ref_object_is_decoded(self):
        schema = {"anyOf": [{"$ref": "#/$defs/Edit"}, {"type": "null"}]}
        tool = _tool({"e": schema}, **{"$defs": EDIT_DEFS})
        text = _call(e='{"old": "a", "new": "b", "all": true}')
        self.assert_args(tool, text, {"e": {"old": "a", "new": "b", "all": True}})

    def test_single_allof_ref_and_root_ref(self):
        tool = _tool(
            {"e": {"allOf": [{"$ref": "#/$defs/Edit"}]}}, **{"$defs": EDIT_DEFS}
        )
        self.assert_args(
            tool, _call(e='{"old": "a", "new": "b"}'), {"e": {"old": "a", "new": "b"}}
        )
        root_ref = Tool(
            type="function",
            function=Function(
                name="f",
                parameters={
                    "$ref": "#/$defs/A",
                    "$defs": {
                        "A": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                        }
                    },
                },
            ),
        )
        self.assert_args(root_ref, _call(n="3"), {"n": 3})

    def test_null_spelling_follows_schema(self):
        tool = _tool(
            {
                "s": {"type": "string"},
                "req_s": {"type": "string"},
                "n": {"type": "integer"},
                "nn": {"type": ["integer", "null"]},
                "oa": {"type": "integer", "nullable": True},
            },
            ["req_s"],
        )
        text = _call(s="null", req_s="null", n="None", nn="null", oa="null")
        self.assert_args(tool, text, {"s": "null", "req_s": "null", "nn": None})

    def test_numbers(self):
        tool = _tool(
            {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
                "c": {"type": "integer"},
                "d": {"type": "number"},
                "e": {"type": "number"},
                "g": {"type": "integer"},
            }
        )
        text = _call(a="5.0", b='"7"', c="1e3", d="2.50", e="3", g="1_000")
        self.assert_args(
            tool, text, {"a": 5, "b": 7, "c": 1000, "d": 2.5, "e": 3, "g": 1000}
        )

    def test_non_finite_number_never_emits_nan(self):
        tool = _tool({"x": {"type": "number"}})
        result = Qwen3CoderDetector().detect_and_parse(_call(x="NaN"), [tool])
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": "NaN"})
        self.assertNotIn("NaN,", result.calls[0].parameters.replace('"NaN"', ""))

    def test_unions_try_each_type_in_order(self):
        tool = _tool(
            {
                "ia": {"anyOf": [{"type": "integer"}, {"type": "array"}]},
                "nb": {"anyOf": [{"type": "number"}, {"type": "boolean"}]},
                "as": {"type": ["array", "string"]},
                "en": {"enum": [1, 2, 3]},
            }
        )
        text = _call(ia="[1, 2]", nb="true", **{"as": "123"}, en="2")
        self.assert_args(tool, text, {"ia": [1, 2], "nb": True, "as": "123", "en": 2})

    def test_empty_container_values(self):
        tool = _tool({"o": {"type": "object"}, "a": {"type": "array"}})
        self.assert_args(tool, _call(o="", a=""), {"o": {}, "a": []})

    def test_python_literal_containers(self):
        tool = _tool({"a": {"type": "array"}, "o": {"type": "object"}})
        text = _call(a="[{'k': True}]", o="{'a': None}")
        self.assert_args(tool, text, {"a": [{"k": True}], "o": {"a": None}})

    def test_string_values_stay_verbatim(self):
        tool = _tool(
            {"s": {"type": "string"}, "t": {"type": "string", "enum": ["1", "2"]}}
        )
        text = _call(s='{\n  "a": 1\n}', t="1")
        self.assert_args(tool, text, {"s": '{\n  "a": 1\n}', "t": "1"})

    def test_unconstrained_schema_keeps_raw_text(self):
        self.assertIsNone(schema_options({}, {"description": "anything"}))
        self.assertEqual(coerce_value("42", None), "42")
        self.assertIsNone(coerce_value("null", None))

    def test_ref_cycle_terminates(self):
        root = {"$defs": {"A": {"$ref": "#/$defs/A"}}}
        self.assertIsNone(schema_options(root, {"$ref": "#/$defs/A"}))
        schema = ToolParamSchema(
            {"type": "object", "properties": {"x": {"$ref": "#/$defs/A"}}, **root}
        )
        self.assertEqual(schema.convert("x", "1"), "1")

    def test_optional_null_is_omitted_required_null_kept(self):
        schema = ToolParamSchema(
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["b"],
            }
        )
        self.assertIs(schema.convert("a", "null"), OMIT)
        self.assertIsNone(schema.convert("b", "null"))


if __name__ == "__main__":
    unittest.main()
