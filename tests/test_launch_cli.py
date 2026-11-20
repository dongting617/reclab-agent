"""The real launch commands remain distinct from the Docker fixture demo."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reclab_agent.research_cli import main


class LaunchCLITests(unittest.TestCase):
    def test_real_run_defaults_to_user_summary_and_html_report(self):
        report = {"status": "completed", "user_summary": "已核验候选指标。",
                  "user_report_path": "fixture/report.html", "report_path": "fixture/report.json"}
        with patch("reclab_agent.research.launch.run_optimization", return_value=report), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["run", "--config", "task.json", "--approve-plan"]), 0)
        self.assertIn("已核验候选指标", output.getvalue())
        self.assertIn("report.html", output.getvalue())
        self.assertFalse(output.getvalue().lstrip().startswith("{"))

    def test_real_run_json_option_preserves_structured_result(self):
        report = {"status": "incomplete", "user_report_path": "fixture/report.html"}
        with patch("reclab_agent.research.launch.run_optimization", return_value=report), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["run", "--config", "task.json", "--approve-plan", "--json"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue()), {"ok": False, "result": report})

    def test_saved_report_renders_offline_without_api_or_execution(self):
        with tempfile.TemporaryDirectory(prefix="offline-report-") as temporary:
            root = Path(temporary)
            source = root / "saved.json"
            saved = {"status": "incomplete", "primary_metric": "ndcg@20", "baseline_id": "baseline",
                     "experiments": [], "comparison": [], "test_evaluated": False,
                     "context": {"task_id": "offline-fixture", "task_question": "离线报告"}}
            source.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            output_directory = root / "readable"
            with patch("reclab_agent.research.launch.run_optimization", side_effect=AssertionError("execution during render")), \
                    patch("reclab_agent.research.launch.check_agent_api", side_effect=AssertionError("API during render")), \
                    patch("reclab_agent.research.launch_environment.load_agent_environment",
                          side_effect=AssertionError("credentials during render")), \
                    patch("subprocess.run", side_effect=AssertionError("subprocess during render")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["report", "--input", str(source), "--output-directory", str(output_directory)])
            self.assertEqual(code, 0)
            self.assertTrue((output_directory / "report.html").is_file())
            self.assertTrue((output_directory / "report.md").is_file())
            self.assertIn("report.html", output.getvalue())
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), saved)

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
