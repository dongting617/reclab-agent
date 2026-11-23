"""Generic orchestration contracts with synthetic data and a nonexecuting Worker.

No candidate model is imported, no checkpoint is loaded, and no Docker/GPU job
is started. Fixture metrics establish only the pipeline's scoring provenance.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from reclab_agent.domain.contracts import Budget, TaskSpec, content_hash
from reclab_agent.core import ModelTurn
from reclab_agent.execution.config import load_policy
from reclab_agent.data.datasets import create_synthetic_bundle
from reclab_agent.execution.worker import InputTree, WorkerPolicy, ResourceLimits
from reclab_agent.domain.configuration import load_task_configuration
from reclab_agent.application.binding import bind_task_tools, create_task_agent
from reclab_agent.domain.research import PlanReview, ResearchPlan
from reclab_agent.experiments.adapters import TaskAdapterRegistry
from reclab_agent.experiments.runner import ExperimentRunner


FIXTURE_IMAGE = "example/reclab-orchestration-fixture@sha256:" + "a" * 64


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        # Running fixtures reach the deadline without real waits or long polling.
        self.now += max(20.0, seconds)


class FixtureWorkerFactory:
    """Simulate stable Worker jobs; never import or execute source files."""

    def __init__(self, predictions, *, train_status="completed", predict_status="completed",
                 submit_error=False):
        self.predictions = predictions
        self.train_status, self.predict_status = train_status, predict_status
        self.submit_error = submit_error
        self.instances = []
        self.created_jobs = []
        self.jobs = {}

    def __call__(self, state_root, policy):
        factory = self

        class FixtureWorker:
            def __init__(self):
                self.root, self.policy = Path(state_root), policy
                self.specs, self.cancellations, self.collections = [], [], []
                self.capabilities_called = False

            def capabilities(self):
                self.capabilities_called = True
                return {"image": policy.image, "real_container_tested": False,
                        "fixture_only": True}

            def submit(self, spec):
                self.specs.append(spec)
                if factory.submit_error:
                    raise RuntimeError("fixture submission outcome uncertain")
                key = (str(self.root), content_hash(asdict(spec)))
                if key not in factory.jobs:
                    status = factory.train_status if spec.phase == "train" else factory.predict_status
                    job_id = "fixture-job-" + content_hash([*key])[:24]
                    job = {"job_id": job_id, "status": status, "spec": asdict(spec),
                           "exit_code": 0 if status == "completed" else (1 if status == "failed" else None),
                           "reason": None if status == "completed" else "fixture_status",
                           "receipt": {"fixture_only": True},
                           "metrics": {"ndcg@20": 999.0}}
                    factory.jobs[key] = job
                    factory.created_jobs.append(job)
                return copy.deepcopy(factory.jobs[key])

            def poll(self, job_id):
                for (root, _), job in factory.jobs.items():
                    if root == str(self.root) and job["job_id"] == job_id:
                        return copy.deepcopy(job)
                raise KeyError(job_id)

            def cancel(self, job_id):
                self.cancellations.append(job_id)
                return {"job_id": job_id, "status": "stopping", "exit_code": None}

            def collect_predictions(self, job_id):
                job = self.poll(job_id)
                if job["status"] != "completed" or job["spec"]["phase"] != "predict":
                    raise AssertionError("only completed fixture prediction jobs are collectable")
                self.collections.append(job_id)
                return copy.deepcopy(factory.predictions)

        worker = FixtureWorker()
        self.instances.append(worker)
        return worker


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="exp-run-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.model_config = {"model": {"file": "src/model.py", "class": "Model", "kwargs": {}},
                             "max_length": 3, "epochs": 1, "patience": 0, "batch_size": 4,
                             "eval_batch_size": 4, "learning_rate": 0.001, "weight_decay": 0,
                             "grad_clip": 5, "device": "cpu"}
        self.files = {
            "src/model.py": "class Model:\n    pass\n",
            "src/train.py": "raise RuntimeError('fixture code must never execute')\n",
            "src/predict.py": "raise RuntimeError('fixture code must never execute')\n",
            "src/data.py": "readonly_dependency = 1\n",
            "configs/default.json": json.dumps(self.model_config) + "\n",
        }
        for relative, text in self.files.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        (self.bundle.root / "private/test.json").write_text("private test sentinel", encoding="utf-8")
        self.budget = Budget(max_training_jobs=2, max_job_seconds=10, max_total_job_seconds=20)
        task = TaskSpec("runner-fixture", "Validate generic experimental orchestration", self.bundle.dataset_id,
                        "fixture-baseline", budget=self.budget)
        value = {
            "schema_version": 1, "task": task.to_dict(),
            "source": {"directory": "source", "config": "default",
                       "editable_files": ["src/model.py", "configs/default.json"]},
            "data": {"directory": "dataset", "adapter": "next_item_v1", "expected_hash": self.bundle.data_hash},
            "evaluation": {"evaluator": "next_item_ranking_v1", "primary_metric": "ndcg@20",
                           "guardrail_metrics": ["recall@20"]},
            "experiment": {"candidate_ids": ["candidate-1"], "seeds": [42]},
            "agent": {"model_env": "RECLAB_FIXTURE_MODEL", "api_key_env": "RECLAB_FIXTURE_KEY"},
            "execution": {"backend": "docker", "policy": None},
            "output": {"directory": "results"},
        }
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(value), encoding="utf-8")
        self.config = load_task_configuration(task_path)
        self.registry = TaskAdapterRegistry()
        description = self.registry.describe_evaluator(self.config.evaluator_id, self.bundle)
        self.plan = ResearchPlan(
            "runner-plan", task.task_id, "Exercise the fixed experiment pipeline", self.bundle.dataset_id,
            "synthetic-contract-fixture", task.baseline_id, description.protocol_hash, "ndcg@20",
            self.config.editable_files, "same source data and seed", "independent fixture grade is returned",
            "reject a malformed or failed experiment", ("budget exhausted",),
            guardrail_metrics=("recall@20",), candidate_ids=("candidate-1",), seeds=(42,), budget=self.budget,
        )
        self.policy = WorkerPolicy(
            FIXTURE_IMAGE, {"baseline": InputTree.freeze(self.source, "source")},
            {self.bundle.dataset_id: InputTree.freeze(self.bundle.public, "public_data")},
            max_limits=ResourceLimits(1, 256, 32, 20), gpu=False,
        )
        self.predictions = {}
        catalog = self.bundle.manifest["catalog"]
        for user, row in self.bundle.valid.items():
            eligible = [item for item in catalog if item not in set(row["history"]) and item != row["target"]]
            self.predictions[user] = [row["target"], *eligible][:20]
        self.clock = FakeClock()
        self.factory = FixtureWorkerFactory(self.predictions)

    def runner(self, **kwargs):
        return ExperimentRunner(self.config, self.plan, self.policy, registry=self.registry,
                                worker_factory=self.factory, clock=self.clock, sleep=self.clock.sleep, **kwargs)

    def snapshot(self, changed=None):
        files = {**self.files, **(changed or {})}
        directory = self.root / "snapshot"
        for relative, text in files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return {"directory": str(directory), "source_hash": content_hash(files), "files": sorted(files)}

    def execute(self, runner=None, snapshot=None):
        return (runner or self.runner())(snapshot or self.snapshot(), candidate_id="candidate-1", seed=42, max_seconds=10)

    def test_constructor_only_registers_inputs_without_creating_jobs_or_outputs(self):
        self.runner()
        self.assertFalse(self.config.output_directory.exists())
        self.assertEqual(self.factory.instances, [])

    def test_constructor_rejects_output_overlap_without_admitting_a_worker(self):
        paths = (self.source, self.source / "generated", self.bundle.root,
                 self.bundle.root / "generated", self.root)
        for output in paths:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlap"):
                ExperimentRunner(replace(self.config, output_directory=output), self.plan, self.policy,
                                 registry=self.registry, worker_factory=self.factory)
        self.assertEqual(self.factory.instances, [])
        self.assertFalse((self.source / "generated").exists())
        self.assertFalse((self.bundle.root / "generated").exists())

    def test_fixed_pipeline_uses_trained_weights_and_an_independent_score(self):
        snapshot = self.snapshot({"src/model.py": "class Model:\n    width = 8\n"})
        record = self.execute(snapshot=snapshot)
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.failure)
        self.assertEqual(record.source_hash, snapshot["source_hash"])
        self.assertEqual(record.config_hash, content_hash(self.model_config))
        worker = self.factory.instances[0]
        self.assertTrue(worker.capabilities_called)
        self.assertEqual([spec.phase for spec in worker.specs], ["train", "predict"])
        self.assertEqual([spec.seed for spec in worker.specs], [42, 42])
        self.assertEqual(worker.specs[1].model_job_id, record.training_job_id)
        self.assertEqual(worker.collections, [record.prediction_job_id])
        self.assertEqual([spec.limits.seconds for spec in worker.specs], [8, 2])
        self.assertEqual(worker.policy.image, FIXTURE_IMAGE)
        score_path = self.config.output_directory / record.score.source
        report = json.loads(score_path.read_text(encoding="utf-8"))
        self.assertEqual(report["metrics"]["ndcg@20"], 1.0)
        self.assertNotEqual(report["metrics"]["ndcg@20"], 999.0)
        self.assertEqual(report["protocol_hash"], self.plan.protocol_hash)
        self.assertEqual(record.score.sha256, hashlib.sha256(score_path.read_bytes()).hexdigest())

    def test_training_failure_never_submits_prediction_or_scores(self):
        self.factory = FixtureWorkerFactory(self.predictions, train_status="failed")
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        worker = self.factory.instances[0]
        self.assertEqual([spec.phase for spec in worker.specs], ["train"])
        self.assertIsNotNone(record.training_job_id)
        self.assertEqual(worker.collections, [])
        self.assertEqual(list(self.config.output_directory.rglob("grade.json")), [])

    def test_prediction_failure_never_collects_or_scores(self):
        self.factory = FixtureWorkerFactory(self.predictions, predict_status="failed")
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        self.assertIsNotNone(record.training_job_id)
        self.assertIsNotNone(record.prediction_job_id)
        self.assertEqual([spec.phase for spec in self.factory.instances[0].specs], ["train", "predict"])
        self.assertEqual(self.factory.instances[0].collections, [])

    def test_readonly_source_mutation_is_rejected_before_worker_admission(self):
        runner = self.runner()
        snapshot = self.snapshot({"src/data.py": "readonly_dependency = 2\n"})
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.execute(runner=runner, snapshot=snapshot)
        self.assertEqual(self.factory.instances, [])

    def test_source_hash_mismatch_and_missing_dependency_are_rejected(self):
        runner = self.runner()
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, "source_hash"):
            self.execute(runner=runner, snapshot={**snapshot, "source_hash": "b" * 64})
        (Path(snapshot["directory"]) / "src/data.py").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.execute(runner=runner, snapshot=snapshot)
        self.assertEqual(self.factory.instances, [])

    def test_timeout_records_unconfirmed_cancellation_and_blocks_later_calls(self):
        self.factory = FixtureWorkerFactory(self.predictions, train_status="running")
        runner, snapshot = self.runner(), self.snapshot()
        record = self.execute(runner=runner, snapshot=snapshot)
        self.assertEqual(record.status, "timed_out")
        self.assertIsNone(record.score)
        worker = self.factory.instances[0]
        self.assertEqual(len(worker.cancellations), 1)
        fence = self.config.output_directory / "execution-blocked.json"
        observation = json.loads(fence.read_text(encoding="utf-8"))
        self.assertFalse(observation["stop_confirmed"])
        self.assertTrue(observation["continued_watchdog_required"])
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            self.execute(runner=runner, snapshot=snapshot)
        self.assertEqual(len(self.factory.instances), 1)

    def test_uncertain_submission_is_fenced_without_claiming_a_score(self):
        self.factory = FixtureWorkerFactory(self.predictions, submit_error=True)
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        self.assertEqual([spec.phase for spec in self.factory.instances[0].specs], ["train"])
        observation = json.loads((self.config.output_directory / "execution-blocked.json").read_text(encoding="utf-8"))
        self.assertFalse(observation["stop_confirmed"])
        self.assertIsNone(observation["job_id"])

    def test_malformed_predictions_fail_the_independent_grader(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[next(iter(predictions))][0] = 999
        self.factory = FixtureWorkerFactory(predictions)
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        self.assertIn("score", record.failure)
        self.assertFalse((self.config.output_directory / "execution-blocked.json").exists())

    def test_scorer_cannot_modify_collected_predictions_and_claim_a_score(self):
        description = self.registry.describe_evaluator(self.config.evaluator_id, self.bundle)

        def mutating_score(dataset, predictions):
            report = dataset.grade(predictions, "valid")
            ranked = predictions[next(iter(predictions))]
            ranked[0], ranked[1] = ranked[1], ranked[0]
            return report

        self.registry.register_evaluator("mutating_score", lambda _: description, score=mutating_score)
        self.config = replace(self.config, evaluator_id="mutating_score")
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        self.assertIn("scorer modified the collected predictions", record.failure)
        self.assertEqual(len(self.factory.instances[0].collections), 1)
        self.assertEqual(list(self.config.output_directory.rglob("grade.json")), [])

    def test_effective_worker_policy_is_saved_and_can_be_reloaded_for_reconciliation(self):
        snapshot = self.snapshot({"src/model.py": "class Model:\n    width = 16\n"})
        record = self.execute(snapshot=snapshot)
        self.assertEqual(record.status, "completed")
        worker = self.factory.instances[0]
        experiment = self.config.output_directory / "experiments" / record.experiment_id
        policy_path = experiment / "worker-policy.json"
        restored = load_policy(policy_path)
        self.assertEqual(restored.image, FIXTURE_IMAGE)
        self.assertEqual(restored.identity(), worker.policy.identity())
        self.assertEqual(dict(restored.sources), dict(worker.policy.sources))
        self.assertEqual(dict(restored.datasets), dict(worker.policy.datasets))
        source_id = worker.specs[0].source_id
        self.assertEqual(restored.sources[source_id], InputTree.freeze(snapshot["directory"], "source"))
        self.assertNotIn("baseline", restored.sources)
        manifest = json.loads((experiment / "execution.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(manifest["worker_policy"]), policy_path)
        self.assertEqual(Path(manifest["worker_state"]), worker.root)

    def test_keyboard_interrupt_requests_cancellation_fences_and_reraises(self):
        self.factory = FixtureWorkerFactory(self.predictions, train_status="running")
        factory = self.factory

        def interrupted_factory(state_root, policy):
            worker = factory(state_root, policy)

            def interrupted_poll(job_id):
                raise KeyboardInterrupt("synthetic operator interruption")

            worker.poll = interrupted_poll
            return worker

        runner = ExperimentRunner(self.config, self.plan, self.policy, registry=self.registry,
                                  worker_factory=interrupted_factory, clock=self.clock,
                                  sleep=self.clock.sleep)
        with self.assertRaisesRegex(KeyboardInterrupt, "synthetic operator interruption"):
            self.execute(runner=runner)
        worker = self.factory.instances[0]
        self.assertEqual(len(worker.cancellations), 1)
        self.assertEqual(worker.collections, [])
        fence = json.loads((self.config.output_directory / "execution-blocked.json").read_text(encoding="utf-8"))
        self.assertFalse(fence["stop_confirmed"])
        self.assertTrue(fence["continued_watchdog_required"])
        experiment = self.config.output_directory / "experiments" / fence["experiment_id"]
        record = json.loads((experiment / "record.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["score"])
        self.assertIn("KeyboardInterrupt", record["failure"])
        self.assertEqual(record["training_job_id"], worker.cancellations[0])
        manifest = json.loads((experiment / "execution.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(fence["worker_policy"]), Path(manifest["worker_policy"]))

    def test_evaluator_without_a_scoring_callback_cannot_admit_experiments(self):
        registry = TaskAdapterRegistry()
        description = registry.describe_evaluator(self.config.evaluator_id, self.bundle)
        registry.register_evaluator("metadata_only", lambda _: description)
        config = replace(self.config, evaluator_id="metadata_only")
        with self.assertRaisesRegex(ValueError, "scoring callback"):
            ExperimentRunner(config, self.plan, self.policy, registry=registry, worker_factory=self.factory)
        self.assertEqual(self.factory.instances, [])
        self.assertFalse(self.config.output_directory.exists())

    def test_policy_must_register_the_actual_baseline_source(self):
        other_source = self.root / "unregistered-model"
        for relative, text in {**self.files, "src/model.py": "class DifferentModel:\n    pass\n"}.items():
            path = other_source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        policy = replace(self.policy, sources={"foreign": InputTree.freeze(other_source, "source")})
        with self.assertRaisesRegex(ValueError, "original source"):
            ExperimentRunner(self.config, self.plan, policy, registry=self.registry, worker_factory=self.factory)
        self.assertEqual(self.factory.instances, [])

    def test_replay_preserves_worker_state_specs_and_job_identities(self):
        runner, snapshot = self.runner(), self.snapshot()
        first = self.execute(runner=runner, snapshot=snapshot)
        second = self.execute(runner=runner, snapshot=snapshot)
        self.assertEqual(first.experiment_id, second.experiment_id)
        self.assertEqual(first.training_job_id, second.training_job_id)
        self.assertEqual(first.prediction_job_id, second.prediction_job_id)
        self.assertEqual(first.score.sha256, second.score.sha256)
        self.assertEqual(len(self.factory.created_jobs), 2)
        self.assertEqual(self.factory.instances[0].root, self.factory.instances[1].root)
        self.assertEqual([asdict(spec) for spec in self.factory.instances[0].specs],
                         [asdict(spec) for spec in self.factory.instances[1].specs])

    def review(self):
        return PlanReview(self.plan.plan_hash, "approved", "synthetic-test-reviewer",
                          "2026-11-19T00:00:00Z", "Synthetic orchestration fixture only")

    def test_task_binding_supplies_the_generic_tool_and_keeps_dependencies_readonly(self):
        tools = bind_task_tools(self.config, plan=self.plan, review=self.review(), policy=self.policy,
                                registry=self.registry, worker_factory=self.factory)
        self.assertEqual(self.factory.instances, [])
        self.assertEqual(tools.workspace.source_paths, frozenset(self.files))
        self.assertEqual(tools.workspace.editable_paths, frozenset(self.config.editable_files))
        source = tools.invoke("read_source", {"path": "src/data.py"})
        self.assertTrue(source["ok"])
        refused = tools.invoke("apply_patch", {"path": "src/data.py", "old": "1", "new": "2",
                                                "expected_sha256": source["result"]["sha256"]})
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["error"]["code"], "OUTSIDE_PLAN")
        result = tools.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["status"], "completed")
        self.assertEqual([spec.phase for spec in self.factory.instances[0].specs], ["train", "predict"])
        self.assertEqual(len(tools.experiments), 1)

    def test_scripted_agent_edits_calls_fixed_pipeline_and_reads_independent_feedback(self):
        observed = []

        class FixtureAgentBackend:
            calls = 0
            prompt = None

            def respond(backend, *, instructions, input_items, tools):
                backend.calls += 1
                if backend.calls == 1:
                    backend.prompt = input_items[0]["content"]
                    name, arguments = "read_source", {"path": "src/model.py"}
                elif backend.calls == 2:
                    source = json.loads(input_items[-1]["output"])["result"]
                    name, arguments = "apply_patch", {
                        "path": "src/model.py", "old": "    pass\n", "new": "    width = 8\n",
                        "expected_sha256": source["sha256"],
                    }
                elif backend.calls == 3:
                    name, arguments = "run_experiment", {"candidate_id": "candidate-1", "seed": 42}
                elif backend.calls == 4:
                    score = json.loads(input_items[-1]["output"])["result"]["score"]
                    name, arguments = "inspect_artifact", {"path": score["source"],
                                                            "expected_sha256": score["sha256"]}
                elif backend.calls == 5:
                    artifact = json.loads(input_items[-1]["output"])["result"]
                    self.assertTrue(artifact["sha256_matches"])
                    name, arguments = "read_json_values", {"path": artifact["path"],
                                                            "pointers": ["/metrics/ndcg@20"]}
                else:
                    values = json.loads(input_items[-1]["output"])["result"]["values"]
                    observed.append(values["/metrics/ndcg@20"])
                    return ModelTurn([], "Synthetic orchestration fixture complete; no model-quality claim.")
                return ModelTurn([{"type": "function_call", "name": name,
                                   "arguments": json.dumps(arguments),
                                   "call_id": f"runner-fixture-{backend.calls}"}], "")

        backend = FixtureAgentBackend()
        agent = create_task_agent(self.config, plan=self.plan, review=self.review(), backend=backend,
                                  policy=self.policy, registry=self.registry, worker_factory=self.factory)
        self.assertEqual(self.factory.instances, [])
        result = agent.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["budget_usage"]["patch_attempts"], 1)
        self.assertEqual(result["budget_usage"]["experiment_attempts"], 1)
        self.assertEqual(len(result["experiments"]), 1)
        self.assertEqual(result["experiments"][0]["status"], "completed")
        self.assertEqual(observed, [1.0])
        self.assertIn('"source_files"', backend.prompt)
        self.assertIn('"readonly_files"', backend.prompt)
        self.assertIn("src/data.py", backend.prompt)
        self.assertEqual([spec.phase for spec in self.factory.instances[0].specs], ["train", "predict"])


if __name__ == "__main__":
    unittest.main()
