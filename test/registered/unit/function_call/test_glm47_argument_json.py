import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(properties):
    return Tool(
        type="function",
        function=Function(
            name="f", parameters={"type": "object", "properties": properties}
        ),
    )


def _call(**params):
    body = "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in params.items()
    )
    return f"<tool_call>f{body}</tool_call>"


def _strict_loads(text):
    def reject(name):
        raise ValueError(name)

    return json.loads(text, parse_constant=reject)


def _stream_args(tools, text, step):
    detector = Glm47MoeDetector()
    args = ""
    for i in range(0, len(text), step):
        for call in detector.parse_streaming_increment(text[i : i + step], tools).calls:
            args += call.parameters or ""
    for call in detector.finish(tools).calls:
        args += call.parameters or ""
    return args


class TestGlm47ArgumentJson(unittest.TestCase):
    """Tool-call arguments must be strict JSON, streamed or not."""

    def assert_same_strict_json(self, tool, text):
        result = Glm47MoeDetector().detect_and_parse(text, [tool])
        expected = _strict_loads(result.calls[0].parameters)
        for step in (1, 4, 32):
            self.assertEqual(_strict_loads(_stream_args([tool], text, step)), expected)
        return expected

    def test_python_style_object_value_streams_as_json(self):
        """An object value that opens with '{' but is Python-style was streamed verbatim."""
        tool = _tool({"o": {"type": "object"}, "a": {"type": "array"}})
        text = _call(o="{'a': None, 'b': True}", a="[1, 2,]")
        self.assertEqual(
            self.assert_same_strict_json(tool, text),
            {"o": {"a": None, "b": True}, "a": [1, 2]},
        )

    def test_valid_json_object_value_unchanged(self):
        tool = _tool({"o": {"type": "object"}})
        text = _call(o='{"k": [1, {"x": "y"}], "n": null}')
        self.assertEqual(
            self.assert_same_strict_json(tool, text),
            {"o": {"k": [1, {"x": "y"}], "n": None}},
        )

    def test_non_finite_number_is_not_emitted_as_bare_token(self):
        """json.loads accepted NaN/Infinity and the arguments carried bare NaN."""
        tool = _tool({"x": {"type": "number"}, "y": {"type": "number"}})
        args = self.assert_same_strict_json(tool, _call(x="NaN", y="Infinity"))
        self.assertEqual(args, {"x": "NaN", "y": "Infinity"})


if __name__ == "__main__":
    unittest.main()
