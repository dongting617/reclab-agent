"""Registration and composition share the evidence-tool boundary."""
from pathlib import Path
import unittest
from reclab_agent.core.session import function_tools
from reclab_agent.tools import FunctionTool, EvidenceToolbox, Tool, ToolError, ToolRegistry, ToolSpec


ROOT = Path(__file__).absolute().parents[1]


def spec(name="increment"):
    return ToolSpec(name, "Increment a number", {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    })


class ToolRegistryTests(unittest.TestCase):
    def test_combines_existing_evidence_and_new_function(self):
        evidence = EvidenceToolbox(ROOT)
        registry = ToolRegistry().include(evidence)
        registry.register_function(spec(), lambda value: {"value": value + 1})
        self.assertEqual(len(registry.specs), 7)
        self.assertEqual(registry.invoke("increment", {"value": 2}),
                         {"ok": True, "tool": "increment", "result": {"value": 3}})
        self.assertEqual(registry.invoke("list_evidence", {"limit": 1}),
                         evidence.invoke("list_evidence", {"limit": 1}))
        self.assertEqual(len(function_tools(registry)), 7)

    def test_custom_tool_and_function_adapter_share_contract(self):
        class Increment(Tool):
            @property
            def spec(self):
                return spec()

            def run(self, arguments):
                return {"value": arguments["value"] + 1}

        registered = ToolRegistry().register(Increment())
        adapted = ToolRegistry().register(FunctionTool(spec(), lambda value: {"value": value + 1}))
        self.assertEqual(registered.invoke("increment", {"value": 4}),
                         adapted.invoke("increment", {"value": 4}))

    def test_duplicate_registration_is_rejected_without_replacing_handler(self):
        registry = ToolRegistry().register_function(spec(), lambda value: {"value": value + 1})
        with self.assertRaises(ValueError):
            registry.register_function(spec(), lambda value: {"value": 0})
        self.assertEqual(registry.invoke("increment", {"value": 2})["result"], {"value": 3})

    def test_include_conflict_is_atomic_and_keeps_original_envelopes(self):
        registry = ToolRegistry().register_function(spec("inspect_artifact"), lambda value: {})
        with self.assertRaises(ValueError):
            registry.include(EvidenceToolbox(ROOT))
        self.assertEqual([item.name for item in registry.specs], ["inspect_artifact"])

        class ExistingToolbox:
            specs = [spec("old")]

            def invoke(self, name, arguments=None):
                return {"ok": False, "error": {"code": "EXISTING"}, "custom": 123}

        registry.include(ExistingToolbox())
        self.assertEqual(registry.invoke("old"),
                         {"ok": False, "error": {"code": "EXISTING"}, "custom": 123})

    def test_duplicate_names_inside_included_toolbox_are_rejected(self):
        class DuplicateToolbox:
            specs = [spec("same"), spec("same")]

            def invoke(self, name, arguments=None):
                self.fail("must not execute during registration")

        registry = ToolRegistry()
        with self.assertRaises(ValueError):
            registry.include(DuplicateToolbox())
        self.assertEqual(registry.specs, [])

    def test_mutable_schemas_are_not_shared_with_callers(self):
        original = spec()
        registry = ToolRegistry().register_function(original, lambda value: {"value": value})
        original.input_schema["properties"]["value"]["type"] = "string"
        exported = registry.specs
        exported[0].input_schema["properties"].clear()
        self.assertEqual(registry.specs[0].input_schema["properties"]["value"]["type"], "integer")

        class ExistingToolbox:
            specs = [spec("included")]

            def invoke(self, name, arguments=None):
                return {"ok": True}

        toolbox = ExistingToolbox()
        registry.include(toolbox)
        toolbox.specs[0].input_schema["properties"].clear()
        self.assertIn("value", registry.specs[1].input_schema["properties"])

    def test_stable_errors_and_unexpected_exceptions_are_distinguished(self):
        def fail(kind):
            if kind == "tool":
                raise ToolError("OUT_OF_SCOPE", "outside registered scope")
            if kind == "value":
                raise ValueError("invalid value")
            raise RuntimeError("unexpected failure")

        registry = ToolRegistry().register_function(spec("fail"), fail)
        self.assertEqual(registry.invoke("fail", {"kind": "tool"})["error"]["code"], "OUT_OF_SCOPE")
        self.assertEqual(registry.invoke("fail", {"kind": "value"})["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(registry.invoke("fail", {"wrong": 1})["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(registry.invoke("fail", [])["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(registry.invoke("missing")["error"]["code"], "UNKNOWN_TOOL")
        with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
            registry.invoke("fail", {"kind": "unexpected"})


if __name__ == "__main__":
    unittest.main()
