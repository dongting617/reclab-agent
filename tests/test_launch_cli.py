"""The real launch commands remain distinct from the Docker fixture demo."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reclab_agent.research_cli import main


class LaunchCLITests(unittest.TestCase):
    def test_real_run_passes_explicit_approval_and_host_options(self):
        run = Mock(return_value={"status": "completed", "report_path": "fixture/report.json"})
        with patch("reclab_agent.research.launch.run_optimization", run), contextlib.redirect_stdout(io.StringIO()):
            code = main(["run", "--config", "task.json", "--env-file", "agent.env",
                         "--approve-plan", "--windows-ssh"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args, (Path("task.json"),))
        self.assertTrue(run.call_args.kwargs["approve_plan"])
        self.assertTrue(run.call_args.kwargs["windows_ssh"])
        self.assertEqual(run.call_args.kwargs["env_file"], Path("agent.env"))

    def test_llm_text_without_measured_candidate_is_nonzero_exit(self):
        with patch("reclab_agent.research.launch.run_optimization", return_value={"status": "incomplete"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "--config", "task.json", "--approve-plan"]), 2)

    def test_legacy_demo_run_uses_its_supervisor(self):
        supervised = Mock(return_value={"ok": True, "result": {"fixture": True}})
        with patch("reclab_agent.research_cli.run_supervised", supervised), \
                patch("reclab_agent.research.launch.run_optimization") as live, \
                contextlib.redirect_stdout(io.StringIO()):
            code = main(["demo", "run", "--root", "fixture", "--image", "fixture-image"])
        self.assertEqual(code, 0)
        supervised.assert_called_once_with("fixture", "fixture-image")
        live.assert_not_called()

    def test_plan_does_not_check_api_or_launch_jobs(self):
        plan = SimpleNamespace(to_dict=lambda: {"budget": "fixture"}, plan_hash="fixture-hash")
        prepared = (SimpleNamespace(output_directory=Path("results")), plan, {})
        with patch("reclab_agent.research.launch.prepare_launch", return_value=prepared), \
                patch("reclab_agent.research.launch.check_agent_api") as api, \
                patch("reclab_agent.research.launch.run_optimization") as live, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["plan", "--config", "task.json"]), 0)
        self.assertIn('"training_started":false', output.getvalue())
        api.assert_not_called()
        live.assert_not_called()
