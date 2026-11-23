"""Keep documented readiness aligned with actual Agent tool exposure."""
from pathlib import Path
import unittest
from reclab_agent.tools import RecLabToolbox, tool_catalog
from reclab_agent.core.session import function_tools


ROOT = Path(__file__).absolute().parents[1]


class ToolCatalogTests(unittest.TestCase):
    def setUp(self):
        self.toolbox = RecLabToolbox(ROOT)

    def test_available_inventory_matches_callable_specs(self):
        available = {entry.name for entry in tool_catalog(availability="available")}
        self.assertEqual(available, {spec.name for spec in self.toolbox.specs})
        self.assertEqual(available, {definition["name"] for definition in function_tools(self.toolbox)})
        catalog = tool_catalog()
        self.assertEqual(len(catalog), len({entry.name for entry in catalog}))
        self.assertTrue(all(entry.group and entry.responsibility for entry in catalog))

    def test_planned_inventory_is_not_executable(self):
        planned = tool_catalog(availability="planned")
        self.assertEqual(len(planned), 9)
        for entry in planned:
            with self.subTest(tool=entry.name):
                result = self.toolbox.invoke(entry.name)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")
        with self.assertRaises(ValueError):
            tool_catalog(availability="enabled")


if __name__ == "__main__":
    unittest.main()
