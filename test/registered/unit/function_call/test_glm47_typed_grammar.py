import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _tool(name, parameters):
    return Tool(type="function", function=Function(name=name, parameters=parameters))


def _call(name, **params):
    body = "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in params.items()
    )
    return f"<tool_call>{name}{body}</tool_call>"


class TestGlm47TypedGrammar(unittest.TestCase):
    """The glm47 tool grammar types argument values by schema without forcing argument order."""

    @classmethod
    def setUpClass(cls):
        try:
            import xgrammar as xgr
            from xgrammar.testing import _is_grammar_accept_string
        except ImportError:
            raise unittest.SkipTest("xgrammar not installed")
        cls.accepts = staticmethod(_is_grammar_accept_string)
        defs = {"Sort": {"type": "object", "properties": {"field": {"type": "string"}}}}
        tools = [
            _tool(
                "run",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"}},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 3600},
                        "level": {"type": "array", "items": {"enum": ["a", "b"]}},
                        "sort": {"anyOf": [{"$ref": "#/$defs/Sort"}, {"type": "null"}]},
                        "note": {"type": "string"},
                    },
                    "required": ["command", "timeout"],
                    "additionalProperties": False,
                    "$defs": defs,
                },
            ),
            _tool(
                "rooted",
                {
                    "$ref": "#/$defs/Args",
                    "$defs": {
                        "Args": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                        }
                    },
                },
            ),
            _tool(
                "union",
                {
                    "anyOf": [
                        {"type": "object", "properties": {"x": {"type": "integer"}}},
                        {"type": "object", "properties": {"y": {"type": "boolean"}}},
                    ]
                },
            ),
            _tool(
                "open",
                {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "additionalProperties": True,
                },
            ),
        ]
        kind, ebnf = FunctionCallParser(tools, "glm47").get_structure_constraint(
            "auto", thinking_mode=False
        )
        assert kind == "full_assistant_ebnf"
        cls.grammar = xgr.Grammar.from_ebnf(ebnf)

    def ok(self, text):
        return self.accepts(self.grammar, text)

    def test_typed_values_any_order(self):
        self.assertTrue(
            self.ok(
                _call(
                    "run",
                    note="hi there",
                    timeout="30",
                    sort='{"field": "x", "extra": true}',
                    command='["ls", "-la"]',
                    level='["b", "a"]',
                )
            )
        )
        self.assertTrue(self.ok(_call("run", timeout="5", command="[]", sort="null")))

    def test_required_arguments_are_enforced(self):
        self.assertFalse(self.ok(_call("run", timeout="5")))
        self.assertFalse(self.ok(_call("run", command='["ls"]')))

    def test_types_inside_values(self):
        self.assertFalse(self.ok(_call("run", command="ls -la", timeout="5")))
        self.assertFalse(self.ok(_call("run", command="[1]", timeout="5")))
        self.assertFalse(self.ok(_call("run", command='["ls"]', timeout="5.5")))
        self.assertFalse(
            self.ok(_call("run", command='["ls"]', timeout="5", level='["c"]'))
        )
        self.assertFalse(self.ok(_call("run", command='["ls"]', timeout="5", bad="1")))

    def test_out_of_range_number_is_not_cut_short(self):
        """A maximum enforced digit by digit decoded an intended 7200 as 720."""
        self.assertTrue(self.ok(_call("run", command="[]", timeout="7200")))

    def test_root_ref_and_union_schemas_are_typed(self):
        self.assertTrue(self.ok(_call("rooted", n="3")))
        self.assertFalse(self.ok(_call("rooted", n="three")))
        self.assertTrue(self.ok(_call("union", x="1", y="true")))
        self.assertFalse(self.ok(_call("union", x="one")))

    def test_open_schema_keeps_extra_keys(self):
        self.assertTrue(self.ok(_call("open", n="1", anything="free text")))


if __name__ == "__main__":
    unittest.main()
