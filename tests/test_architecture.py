"""Keep module ownership and dependency direction clear after reorganization."""
import ast
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / "reclab_agent"


class ArchitectureTests(unittest.TestCase):
    def test_package_root_contains_only_public_and_command_entries(self):
        self.assertEqual({p.name for p in PACKAGE.glob("*.py")},
                         {"__init__.py", "__main__.py", "start.py", "research_cli.py"})
        self.assertFalse((PACKAGE / "research").exists())
        self.assertFalse((PACKAGE / "_execution").exists())

    def test_tools_and_experiments_do_not_depend_on_application_or_workflow(self):
        forbidden = ("reclab_agent.application", "reclab_agent.workflow", "reclab_agent.cli")
        for folder in ("tools", "experiments"):
            for path in (PACKAGE / folder).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                               else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                               else [])
                    for module in modules:
                        with self.subTest(path=path.relative_to(PACKAGE), module=module):
                            self.assertFalse(any(module == x or module.startswith(x + ".")
                                                 for x in forbidden))

    def test_function_tool_definitions_belong_to_tools(self):
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (
                        isinstance(node.func, ast.Name) and node.func.id == "ToolSpec"
                        or isinstance(node.func, ast.Attribute) and node.func.attr == "ToolSpec"):
                    with self.subTest(path=path.relative_to(PACKAGE)):
                        self.assertEqual(path.relative_to(PACKAGE).parts[0], "tools")
