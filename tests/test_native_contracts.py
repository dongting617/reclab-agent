"""Native SSH runtime records and task configuration, without remote execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.contracts import content_hash
from reclab_agent.research.configuration import load_task_configuration
from reclab_agent.research.contracts import EvidenceRef, ExperimentRecord
from tests.test_task_configuration import task_configuration_value


class NativeExperimentContractTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "experiment_id": "exp-1", "plan_hash": "a" * 64,
            "candidate_id": "candidate-1", "source_hash": "b" * 64,
            "config_hash": "c" * 64, "dataset_id": "tiny", "protocol_hash": "d" * 64,
            "image": "example/reclab@sha256:" + "e" * 64, "seed": 2026,
        }
        details = {"python": "3.11.9", "torch": "2.5.1+cu124", "cuda": "12.4",
                   "gpu": ["NVIDIA RTX 4090"], "capabilities": {"cuda_available": True}}
        self.runtime = {"backend": "ssh_tmux", "fingerprint": content_hash(details),
                        "details": details}

    def native(self, **overrides):
        return ExperimentRecord(**{**self.values, "image": None,
                                   "runtime": self.runtime, **overrides})

    def test_existing_docker_record_keeps_the_original_serialized_shape(self):
        record = ExperimentRecord(**self.values)
        expected = {**self.values, "status": "planned", "program_validation": "pending",
                    "training_job_id": None, "prediction_job_id": None,
                    "score": None, "failure": None}
        self.assertEqual(record.to_dict(), expected)
        self.assertIsNone(record.runtime)
        self.assertEqual(ExperimentRecord.from_dict(expected), record)

    def test_native_record_roundtrips_with_runtime_and_no_image(self):
        record = self.native()
        payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        self.assertIsNone(payload["image"])
        self.assertEqual(payload["runtime"], self.runtime)
        self.assertEqual(ExperimentRecord.from_dict(payload), record)

    def test_native_completed_record_still_requires_independent_score_and_jobs(self):
        evidence = EvidenceRef("score.json", "f" * 64, "/metrics/ndcg@20", "valid split")
        completed = self.native(status="completed", program_validation="passed",
                                training_job_id="train-1", prediction_job_id="predict-1",
                                score=evidence)
        self.assertEqual(ExperimentRecord.from_dict(completed.to_dict()), completed)
        with self.assertRaises(ValueError):
            self.native(status="completed", program_validation="passed")

    def test_native_identity_cannot_be_missing_or_disguised_as_a_docker_image(self):
        for image in (None, "ssh_tmux", "python:latest", "sha256:" + "f" * 64):
            with self.subTest(image=image), self.assertRaises(ValueError):
                ExperimentRecord(**{**self.values, "image": image})
        for image in (self.values["image"], "native@sha256:" + "f" * 64):
            with self.subTest(image=image), self.assertRaises(ValueError):
                self.native(image=image)

    def test_runtime_schema_and_fingerprint_are_strict(self):
        invalid = [[], {}, {**self.runtime, "backend": "docker"},
                   {**self.runtime, "details": []}, {**self.runtime, "unexpected": True},
                   {**self.runtime, "fingerprint": "f" * 64},
                   {**self.runtime, "details": {"python": "changed after fingerprinting"}}]
        for key in self.runtime:
            runtime = dict(self.runtime)
            del runtime[key]
            invalid.append(runtime)
        for fingerprint in (None, 123, "abc", "F" * 64, "f" * 63):
            invalid.append({**self.runtime, "fingerprint": fingerprint})
        for runtime in invalid:
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                self.native(runtime=runtime)

    def test_runtime_details_only_accept_finite_json(self):
        for value in (float("nan"), float("inf"), -float("inf"), object(), (1, 2), {1, 2}):
            runtime = {**self.runtime, "details": {"nested": [value]}}
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.native(runtime=runtime)
        with self.assertRaises(ValueError):
            self.native(runtime={**self.runtime, "details": {1: "non-string key"}})
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(ValueError):
            self.native(runtime={**self.runtime, "details": cyclic})


class NativeTaskConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="native-cfg-", dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "task.json"
        self.value = task_configuration_value()

    def load(self, execution):
        value = copy.deepcopy(self.value)
        value["execution"] = execution
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return load_task_configuration(self.path)

    def test_native_connection_path_is_relative_to_task_configuration(self):
        loaded = self.load({"backend": "ssh_tmux", "connection": "connections/autodl.json"})
        self.assertEqual(loaded.execution_backend, "ssh_tmux")
        self.assertIsNone(loaded.execution_policy_path)
        self.assertEqual(loaded.execution_connection_path, self.directory / "connections/autodl.json")
        self.assertFalse(loaded.execution_connection_path.exists())

    def test_native_connection_may_be_an_absolute_literal_path(self):
        path = self.directory / "autodl.json"
        loaded = self.load({"backend": "ssh_tmux", "connection": str(path)})
        self.assertEqual(loaded.execution_connection_path, path)

    def test_docker_configuration_retains_policy_and_has_no_connection(self):
        loaded = self.load({"backend": "docker", "policy": "policy.json"})
        self.assertEqual(loaded.execution_policy_path, self.directory / "policy.json")
        self.assertIsNone(loaded.execution_connection_path)

    def test_native_and_docker_configuration_fields_cannot_be_mixed(self):
        invalid = [
            {"backend": "ssh_tmux"},
            {"backend": "ssh_tmux", "connection": None},
            {"backend": "ssh_tmux", "connection": ""},
            {"backend": "ssh_tmux", "connection": "~/.ssh/config"},
            {"backend": "ssh_tmux", "connection": "bad\npath"},
            {"backend": "ssh_tmux", "connection": "ssh.json", "policy": None},
            {"backend": "ssh_tmux", "connection": "ssh.json", "password": "not-accepted"},
            {"backend": "docker", "policy": None, "connection": "ssh.json"},
            {"backend": "docker", "connection": "ssh.json"},
            {"backend": "other", "connection": "ssh.json"},
        ]
        for execution in invalid:
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                self.load(execution)


if __name__ == "__main__":
    unittest.main()
