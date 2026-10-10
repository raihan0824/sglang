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


class TestQwen3CoderStreamingFraming(unittest.TestCase):
    """Streamed calls must be complete JSON and match the non-streamed parse."""

    tool = _tool({"n": {"type": "integer"}, "s": {"type": "string"}}, ["n"])

    def _stream_calls(self, text, step):
        detector = Qwen3CoderDetector()
        calls, normal = {}, ""
        for i in range(0, len(text), step):
            result = detector.parse_streaming_increment(text[i : i + step], [self.tool])
            normal += result.normal_text
            for call in result.calls:
                entry = calls.setdefault(call.tool_index, [None, ""])
                entry[0] = call.name or entry[0]
                entry[1] += call.parameters or ""
        result = detector.finish([self.tool])
        normal += result.normal_text
        for call in result.calls:
            calls.setdefault(call.tool_index, [None, ""])[1] += call.parameters
        return calls, normal

    def test_call_without_function_end_is_closed(self):
        for text in (
            "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n</tool_call>",
            "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n",
            "<tool_call>\n<function=f>\n<parameter=n>\n1\n",
        ):
            for step in (1, 5):
                calls, _ = self._stream_calls(text, step)
                self.assertEqual(calls, {0: ["f", '{"n": 1}']}, (text, step))

    def test_stray_function_end_is_not_a_call(self):
        """Claude-style <invoke> XML closed with </function> once streamed a nameless {} call."""
        text = (
            '<function_calls>\n<invoke name="f">\n<parameter name="n">1</parameter>\n'
            "</invoke>\n</function>\n</tool_call>"
        )
        for step in (1, 4):
            calls, normal = self._stream_calls(text, step)
            self.assertEqual(calls, {})
            self.assertIn('<invoke name="f">', normal)
        self.assertEqual(
            Qwen3CoderDetector().detect_and_parse(text, [self.tool]).calls, []
        )

    def test_next_function_closes_previous(self):
        text = (
            "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n"
            "<function=f>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>"
        )
        calls, _ = self._stream_calls(text, 3)
        self.assertEqual(calls, {0: ["f", '{"n": 1}'], 1: ["f", '{"n": 2}']})

    def test_names_are_stripped(self):
        text = "<tool_call>\n<function= f\n>\n<parameter= n >\n1\n</parameter>\n</function>\n</tool_call>"
        calls, _ = self._stream_calls(text, 2)
        self.assertEqual(calls, {0: ["f", '{"n": 1}']})
        parsed = Qwen3CoderDetector().detect_and_parse(text, [self.tool]).calls
        self.assertEqual((parsed[0].name, parsed[0].parameters), ("f", '{"n": 1}'))


class TestQwen3CoderForcedCallGrammar(unittest.TestCase):
    """Forced (required/named) calls must be able to spell every declared parameter."""

    def test_required_grammar_accepts_declared_dashed_name(self):
        try:
            import xgrammar as xgr
            from xgrammar.testing import _is_grammar_accept_string
        except ImportError:
            self.skipTest("xgrammar not installed")
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        grep = Tool(
            type="function",
            function=Function(
                name="Grep",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "-i": {"type": "boolean"},
                    },
                    "required": ["pattern"],
                },
            ),
        )
        parser = FunctionCallParser([grep], "qwen3_coder")
        for choice in ("required", {"type": "function", "function": {"name": "Grep"}}):
            if isinstance(choice, dict):
                from sglang.srt.entrypoints.openai.protocol import ToolChoice

                choice = ToolChoice(**choice)
            kind, tag = parser.get_structure_constraint(choice)
            self.assertEqual(kind, "structural_tag")
            grammar = xgr.Grammar.from_structural_tag(tag)
            call = "<tool_call>\n<function=Grep>\n<parameter=pattern>\nTODO\n</parameter>\n<parameter=-i>\ntrue\n</parameter>\n</function>\n</tool_call>"
            self.assertTrue(_is_grammar_accept_string(grammar, call))
            self.assertFalse(
                _is_grammar_accept_string(grammar, call.replace("=-i>", "=i>"))
            )


class TestQwen3CoderConformanceLog(unittest.TestCase):
    """Calls that would fail OpenRouter's checks log where they break, never values."""

    LOGGER = "sglang.srt.function_call.call_conformance"
    tool = _tool(
        {
            "command": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "integer", "maximum": 3600},
        },
        ["command"],
        additionalProperties=False,
    )

    def _stream(self, text, step=3):
        detector = Qwen3CoderDetector()
        for i in range(0, len(text), step):
            detector.parse_streaming_increment(text[i : i + step], [self.tool])
        detector.finish([self.tool])

    def test_schema_mismatch_logged_without_values(self):
        text = _call(command="cat /secret/path.txt", timeout="7200", extra="v")
        with self.assertLogs(self.LOGGER, "WARNING") as logs:
            Qwen3CoderDetector().detect_and_parse(text, [self.tool])
            self._stream(text)
        joined = "\n".join(logs.output)
        self.assertEqual(
            joined.count("qwen3_coder tool call fails its schema: tool=f"), 2
        )
        self.assertIn("mode=nonstream", joined)
        self.assertIn("mode=stream ", joined)
        self.assertIn('$.command type="array" got=string', joined)
        self.assertIn("$.timeout maximum=3600 got=integer", joined)
        self.assertIn("'extra' was unexpected", joined)
        self.assertNotIn("secret", joined)
        self.assertNotIn("7200", joined)

    def test_unknown_name_and_truncated_stream(self):
        with self.assertLogs(self.LOGGER, "WARNING") as logs:
            Qwen3CoderDetector().detect_and_parse(
                _call(command='["ls"]').replace("=f>", "=g>"), [self.tool]
            )
            self._stream("<tool_call>\n<function=f>\n<parameter=timeout>\n5\n")
        joined = "\n".join(logs.output)
        self.assertIn(
            "qwen3_coder tool call name not in tools: name='g' mode=nonstream", joined
        )
        self.assertIn("mode=stream-eos", joined)
        self.assertIn("'command' is a required property", joined)

    def test_conforming_call_logs_nothing(self):
        text = _call(command='["ls", "-la"]', timeout="5")
        with self.assertNoLogs(self.LOGGER, "WARNING"):
            Qwen3CoderDetector().detect_and_parse(text, [self.tool])
            self._stream(text)

    def test_broken_schema_never_raises(self):
        bad = Tool(
            type="function",
            function=Function(
                name="f",
                parameters={"type": "object", "properties": {"x": {"type": 5}}},
            ),
        )
        result = Qwen3CoderDetector().detect_and_parse(_call(x="1"), [bad])
        self.assertEqual(len(result.calls), 1)


class TestQwen3CoderAutoGrammar(unittest.TestCase):
    """tool_choice=auto constrains calls to declared names and typed values in any order."""

    @classmethod
    def setUpClass(cls):
        try:
            import xgrammar as xgr
            from xgrammar.testing import _is_grammar_accept_string
        except ImportError:
            raise unittest.SkipTest("xgrammar not installed")
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        cls.accepts = staticmethod(_is_grammar_accept_string)
        tools = [
            _tool(
                {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "timeout": {"type": "integer", "maximum": 3600},
                    "background": {"type": "boolean"},
                    "mode": {"type": "string", "enum": ["fast", "safe"]},
                    "note": {"type": ["string", "null"]},
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string"},
                            },
                        },
                    },
                },
                ["command"],
            ),
            Tool(
                type="function",
                function=Function(
                    name="noargs", parameters={"type": "object", "properties": {}}
                ),
            ),
        ]
        kind, tag = FunctionCallParser(tools, "qwen3_coder").get_structure_constraint(
            "auto"
        )
        assert kind == "structural_tag"
        cls.grammar = xgr.Grammar.from_structural_tag(tag)

    def ok(self, text):
        return self.accepts(self.grammar, text)

    def test_valid_calls_in_any_order_are_accepted(self):
        call = _call(
            timeout="60",
            background="True",
            command='["ls", "-la"]',
            todos='[{"status": "pending", "content": "x"}]',
            mode="safe",
            note="null",
        )
        self.assertTrue(self.ok("Let me look.\n\n" + call))
        self.assertTrue(self.ok(call + "\n" + _call(command='["pwd"]')))
        self.assertTrue(
            self.ok("<tool_call>\n<function=noargs>\n</function>\n</tool_call>")
        )
        self.assertTrue(self.ok("no tool call at all"))

    def test_out_of_range_number_is_not_cut_short(self):
        """A maximum enforced digit by digit decoded an intended 7200 as 720."""
        self.assertTrue(self.ok(_call(command='["ls"]', timeout="7200")))

    def test_schema_violations_are_rejected(self):
        self.assertFalse(self.ok(_call(command="cat /x")))  # string for array
        self.assertFalse(
            self.ok(_call(command='["ls"]', timeout="1.5"))
        )  # not an integer
        self.assertFalse(self.ok(_call(command='["ls"]', mode="slow")))  # not in enum
        self.assertFalse(
            self.ok(_call(command='["ls"]', extra="1"))
        )  # undeclared parameter
        self.assertFalse(self.ok(_call(timeout="5")))  # required parameter missing
        self.assertFalse(
            self.ok(_call(command='["ls"]').replace("=f>", "=g>"))
        )  # undeclared tool


if __name__ == "__main__":
    unittest.main()
