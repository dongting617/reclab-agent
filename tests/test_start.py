"""The simple launcher delegates to the same bounded experiment workflow."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from reclab_agent import start
from reclab_agent.contracts import Budget


class SimpleStartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="simple-start-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "reclab.json"
        self.profile.write_text(json.dumps({"task": "configs/task.json", "env_file": ".reclab/agent.env"}),
                                encoding="utf-8")
        self.budget = Budget(max_training_jobs=2, max_job_seconds=600, max_total_job_seconds=1200)
        self.config = SimpleNamespace(
            task=SimpleNamespace(task_id="fixture-task", dataset_id="fixture-data", baseline_id="baseline",
                                 question="Optimize the fixture", budget=self.budget),
            source_directory=self.root / "model", dataset_directory=self.root / "dataset",
            config_name="default", primary_metric="ndcg@20", guardrail_metrics=("recall@20",),
            seeds=(42,), candidate_ids=("candidate-1",), output_directory=self.root / "results",
        )
        self.plan = SimpleNamespace(
            plan_hash="fixture-plan", task_id="fixture-task", dataset_id="fixture-data", baseline_id="baseline",
            primary_metric="ndcg@20", editable_files=("src/model.py",), seeds=(42,),
            candidate_ids=("baseline", "candidate-1"), budget=self.budget,
            to_dict=lambda: {"budget": {"max_training_jobs": 2}, "primary_metric": "ndcg@20"},
        )

    def report(self, status="completed"):
        return {"status": status, "report_path": str(self.root / "results/report.json"),
                "user_report_path": str(self.root / "results/report.html"),
                "user_summary": "本轮实验已完成。" if status == "completed" else "本轮没有有效候选分数。",
                "comparison": [], "experiments": []}

    def test_profile_paths_are_relative_to_project_not_calling_directory(self):
        with patch("os.getcwd", return_value=str(self.root / "unrelated")):
            task, environment = start.load_profile(self.root)
        self.assertEqual(task, self.root / "configs/task.json")
        self.assertEqual(environment, self.root / ".reclab/agent.env")

    def test_explicit_task_and_environment_paths_override_profile(self):
        task, environment = start.load_profile(self.root, config="other/task.json", env_file="other/agent.env")
        self.assertEqual(task, self.root / "other/task.json")
        self.assertEqual(environment, self.root / "other/agent.env")

    def test_profile_rejects_commands_and_credentials_as_configuration_fields(self):
        value = {"task": "configs/task.json", "env_file": ".reclab/agent.env",
                 "command": "credential-never-returned"}
        self.profile.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            start.load_profile(self.root)
        self.assertNotIn("credential-never-returned", str(raised.exception))

    def test_missing_profile_is_an_explicit_configuration_error(self):
        self.profile.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            start.load_profile(self.root)

    def test_plan_never_loads_environment_installs_sdk_or_launches_jobs(self):
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "prepare_launch", return_value=(self.config, self.plan, {})), \
                patch.object(start, "ensure_runtime", side_effect=AssertionError("SDK preparation during plan")), \
                patch.object(start, "run_optimization") as run, \
                patch("reclab_agent.research.launch_environment.load_agent_environment",
                      side_effect=AssertionError("environment read during plan")), \
                patch("subprocess.run", side_effect=AssertionError("subprocess during plan")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = start.main(["--plan"], project_root=self.root)
        self.assertEqual(result, 0)
        run.assert_not_called()
        self.assertTrue(output.getvalue())

    def test_start_delegates_configuration_approval_and_ssh_to_existing_workflow(self):
        run = Mock(return_value=self.report())
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "ensure_runtime", return_value=None), \
                patch.object(start, "run_optimization", run), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = start.main(["--windows-ssh"], project_root=self.root)
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 1)
        self.assertIn(run.call_args.args[0], (self.config, self.root / "configs/task.json"))
        self.assertTrue(run.call_args.kwargs["approve_plan"])
        self.assertTrue(run.call_args.kwargs["windows_ssh"])
        self.assertEqual(run.call_args.kwargs["env_file"], self.root / ".reclab/agent.env")
        self.assertIn("本轮实验已完成", output.getvalue())
        self.assertIn("report.html", output.getvalue())

    def test_runtime_reexecution_returns_child_exit_without_second_launch(self):
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "ensure_runtime", return_value=17), \
                patch.object(start, "run_optimization") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            result = start.main([], project_root=self.root)
        self.assertEqual(result, 17)
        run.assert_not_called()

    def test_already_running_task_is_rejected_before_api_or_experiments(self):
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "ensure_runtime", return_value=None), \
                patch.object(start, "job_lock", return_value=contextlib.nullcontext(False)) as lock, \
                patch.object(start, "run_optimization") as run, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = start.main([], project_root=self.root)
        self.assertEqual(result, 2)
        lock.assert_called_once_with(self.config.output_directory, "optimization-start")
        run.assert_not_called()
        self.assertIn("already running", errors.getvalue())

    def test_current_sdk_is_reused_without_install_or_reexecution(self):
        with patch.object(start.importlib.util, "find_spec", return_value=object()), \
                patch.object(start.importlib.metadata, "version", return_value="3.55.1"), \
                patch.object(start.subprocess, "run", side_effect=AssertionError("SDK reinstall")), \
                patch.object(start.subprocess, "call", side_effect=AssertionError("unnecessary child")):
            self.assertIsNone(start.ensure_runtime(self.root, []))
        self.assertFalse((self.root / ".reclab").exists())

    def test_cached_sdk_reexecution_preserves_literal_arguments_without_install(self):
        runtime = self.root / "cached runtime" / "python"
        state = self.root / ".reclab/controller.json"
        state.parent.mkdir()
        state.write_text(json.dumps({"python": str(runtime)}), encoding="utf-8")
        arguments = ["--config", "config with spaces.json", "--windows-ssh"]
        with patch.object(start.importlib.util, "find_spec", return_value=None), \
                patch.object(start, "_usable_python", return_value=True), \
                patch.object(start.subprocess, "run", side_effect=AssertionError("cached SDK reinstalled")), \
                patch.object(start.subprocess, "call", return_value=7) as child:
            self.assertEqual(start.ensure_runtime(self.root, arguments), 7)
        child.assert_called_once_with([str(runtime), "-B", "-m", "reclab_agent.start", *arguments], cwd=self.root)

    def test_unmeasured_candidate_is_not_a_successful_start_result(self):
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "ensure_runtime", return_value=None), \
                patch.object(start, "run_optimization", return_value=self.report("incomplete")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = start.main([], project_root=self.root)
        self.assertEqual(result, 2)
        self.assertIn("没有有效候选分数", output.getvalue())
        self.assertIn("report.html", output.getvalue())

    def test_unexpected_provider_errors_do_not_echo_credentials_or_tracebacks(self):
        with patch.object(start, "load_task_configuration", return_value=self.config), \
                patch.object(start, "ensure_runtime", return_value=None), \
                patch.object(start, "run_optimization", side_effect=RuntimeError("credential-never-returned")), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = start.main([], project_root=self.root)
        rendered = output.getvalue() + errors.getvalue()
        self.assertEqual(result, 2)
        self.assertNotIn("credential-never-returned", rendered)
        self.assertNotIn("Traceback", rendered)


if __name__ == "__main__":
    unittest.main()
