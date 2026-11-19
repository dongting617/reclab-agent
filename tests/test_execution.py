"""Worker fault injection tests; fake Docker never counts as real isolation."""
from dataclasses import asdict, replace
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from reclab_agent.datasets import create_synthetic_bundle, popularity_predictions
from reclab_agent.adapters import SourceBackend
from reclab_agent.execution import (CapabilityError, DockerError, DockerWorker,
                                    InputTree, JobSpec, OwnershipError, ResourceLimits,
                                    SubprocessDocker, WorkerPolicy)
from tests.support.docker import FakeDocker, IMAGE, _hold_job_lock


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        for relative, value in {"src/train.py": "print('not executed by unit tests')\n",
                                "src/predict.py": "raise RuntimeError('must remain in Docker')\n",
                                "configs/default.json": "{}"}.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        self.policy = WorkerPolicy(IMAGE, {"baseline": InputTree.freeze(self.source, "source")},
                                   {"tiny": InputTree.freeze(self.bundle.public, "public_data")})
        self.fake, self.now = FakeDocker(), 1000.0
        self.worker = DockerWorker(self.root / "state", self.policy, runner=self.fake, clock=lambda: self.now)
        self.spec = JobSpec("baseline", "tiny")

    def finish(self, job, *, code=0, oom=False, files=None):
        for relative, value in (files or {"model.json": "{}"}).items():
            path = self.worker._output(job["job_id"]) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.fake.containers[job["container"]]["State"].update(Status="exited", ExitCode=code, OOMKilled=oom)
        return self.worker.poll(job["job_id"])

    def prediction(self, raw=None):
        train = self.finish(self.worker.submit(self.spec))
        spec = replace(self.spec, phase="predict", model_job_id=train["job_id"])
        prediction = self.worker.submit(spec)
        values = popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid"))
        prediction = self.finish(prediction, files={"predictions.json": json.dumps(values) if raw is None else raw})
        return prediction, values

    def test_spec_has_no_command_image_or_path_escape(self):
        for fields in ({"source_id": "../../private"}, {"config": "../test"}, {"seed": True},
                       {"phase": "test"}, {"phase": "predict", "model_job_id": None}):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                JobSpec(**{**asdict(self.spec), "limits": self.spec.limits, **fields})
        with self.assertRaises(TypeError):
            JobSpec.from_dict({**asdict(self.spec), "command": "rm -rf /"})
        with self.assertRaises(ValueError):
            replace(self.policy, image="python:latest")
        with self.assertRaises(ValueError):
            self.worker.submit(replace(self.spec, source_id="unknown"))
        self.assertEqual(self.fake.count("create"), 0)

    def test_public_registration_rejects_labels_and_parent_bundle(self):
        with self.assertRaises(ValueError):
            InputTree.freeze(self.bundle.root, "public_data")
        (self.bundle.public / "test.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            InputTree.freeze(self.bundle.public, "public_data")

    def test_input_identity_is_checked_before_launch(self):
        (self.source / "src/train.py").write_text("changed = True", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.worker.submit(self.spec)
        self.assertFalse(self.fake.calls)

    def test_operator_resources_cannot_be_expanded(self):
        for limits in (ResourceLimits(cpus=5), ResourceLimits(memory_mb=16384), ResourceLimits(pids=512), ResourceLimits(seconds=1801)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.worker.submit(replace(self.spec, limits=limits))
        for bad in ({"cpus": float("nan")}, {"seconds": True}, {"memory_mb": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ResourceLimits(**bad)

    def test_unavailable_daemon_fails_closed_without_job_or_host_execution(self):
        self.fake.available = False
        with self.assertRaises(CapabilityError):
            self.worker.submit(self.spec)
        with self.worker._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM docker_jobs").fetchone()[0], 0)
        self.assertFalse(any("train.py" in value for call in self.fake.calls for value in call))

    def test_real_cli_wrapper_is_shell_free_local_and_drops_host_secret_environment(self):
        with patch("reclab_agent._execution.docker.shutil.which", return_value="/usr/bin/docker"), \
             patch("reclab_agent._execution.docker.sys.platform", "linux"), \
             patch("reclab_agent._execution.docker.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="{}", stderr="")) as run, \
             patch.dict(os.environ, {"DOCKER_HOST": "tcp://untrusted:2375", "AWS_SECRET_ACCESS_KEY": "never-forward", "PYTHONPATH": "/untrusted"}):
            runner = SubprocessDocker(self.root / "state")
            runner(["info", "--format", "{{json .}}"])
        args, kwargs = run.call_args
        self.assertEqual(args[0][:3], ["/usr/bin/docker", "--host", "unix:///var/run/docker.sock"])
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(set(kwargs["env"]), {"PATH", "LANG", "DOCKER_CONFIG"})
        self.assertNotIn("never-forward", str(kwargs))

    def test_wrong_digest_and_image_declared_volumes_fail_closed(self):
        self.fake.image_digest = "example/other@sha256:" + "b" * 64
        with self.assertRaises(CapabilityError):
            self.worker.capabilities()
        self.fake.image_digest = IMAGE
        self.fake.volumes = {"/somewhere-writable": {}}
        with self.assertRaises(CapabilityError):
            self.worker.capabilities()

    def test_seccomp_and_gpu_capabilities_are_required(self):
        self.fake.security = []
        with self.assertRaises(CapabilityError):
            self.worker.capabilities()
        self.fake.security = ["name=seccomp,profile=builtin"]
        worker = DockerWorker(self.root / "gpu-state", replace(self.policy, gpu=True), runner=self.fake)
        with self.assertRaises(CapabilityError):
            worker.capabilities()

    def test_seccomp_unconfined_is_not_a_valid_sandbox_capability(self):
        self.fake.security = ["name=seccomp,profile=unconfined"]
        with self.assertRaises(CapabilityError):
            self.worker.capabilities()

    def test_unsupported_cgroup_limits_do_not_silently_degrade(self):
        self.fake.resource_limits = False
        with self.assertRaises(CapabilityError):
            self.worker.submit(self.spec)
        self.assertEqual(self.fake.count("create"), 0)

    def test_supervisor_reconciles_without_client_connection(self):
        job = self.worker.submit(self.spec)
        self.now += 601
        stop = threading.Event()
        seen = []
        def tick(results):
            seen.extend(results)
            stop.set()
        self.worker.serve(stop=stop, on_tick=tick)
        self.assertEqual(seen[0]["status"], "timed_out")
        self.assertEqual(self.fake.count("kill"), 1)

    def test_only_fixed_docker_arguments_and_expected_read_only_mounts(self):
        job = self.worker.submit(self.spec)
        self.assertEqual(job["status"], "running")
        args = next(call for call in self.fake.calls if call[:2] == ["container", "create"])
        for option, value in (("--user", "65532:65532"), ("--network", "none"), ("--cap-drop", "ALL"),
                              ("--security-opt", "no-new-privileges=true"), ("--pull", "never"),
                              ("--memory", "2048m"), ("--memory-swap", "2048m"), ("--pids-limit", "128")):
            self.assertEqual(args[args.index(option) + 1], value)
        self.assertIn("--read-only", args)
        # Docker's local driver defaults to compression, which refuses a
        # single retained log file. Real Engine 29 failed before process start.
        log_options = [args[i + 1] for i, value in enumerate(args) if value == "--log-opt"]
        self.assertIn("max-file=1", log_options)
        self.assertIn("compress=false", log_options)
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
        self.assertEqual(len(mounts), 3)
        self.assertTrue(next(value for value in mounts if "target=/workspace" in value).endswith(",readonly"))
        self.assertTrue(next(value for value in mounts if "target=/data" in value).endswith(",readonly"))
        self.assertNotIn(str(self.bundle.root / "private"), " ".join(args))
        self.assertNotIn("docker.sock", " ".join(args))
        self.assertEqual(args[args.index(IMAGE) + 1:args.index(IMAGE) + 5], ["-E", "-s", "-B", "/workspace/src/train.py"])
        self.assertNotIn("--privileged", args)
        backend = SourceBackend("baseline", ("src/train.py", "src/predict.py"), IMAGE)
        self.assertEqual(tuple(args[args.index(IMAGE) + 4:]), backend.training_command()[2:])

    def test_idempotent_submission_survives_worker_restart(self):
        first = self.worker.submit(self.spec)
        restarted = DockerWorker(self.root / "state", self.policy, runner=self.fake, clock=lambda: self.now)
        second = restarted.submit(self.spec)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)

    def test_concurrent_duplicate_submit_starts_only_one_container(self):
        workers = [DockerWorker(self.root / "state", self.policy, runner=self.fake, clock=lambda: self.now) for _ in range(2)]
        results, errors = [], []
        barrier = threading.Barrier(2)
        def submit(worker):
            try:
                barrier.wait()
                results.append(worker.submit(self.spec))
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=submit, args=(worker,)) for worker in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(len({value["job_id"] for value in results}), 1)
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)

    def test_concurrency_limit_queues_second_job_until_slot_is_free(self):
        first = self.worker.submit(self.spec)
        second = self.worker.submit(replace(self.spec, seed=43))
        self.assertEqual(second["status"], "queued")
        self.assertEqual(self.fake.count("create"), 1)
        self.finish(first)
        self.assertEqual(self.worker.poll(second["job_id"])["status"], "running")
        self.assertEqual(self.fake.count("create"), 2)

    def test_disconnect_after_create_recovers_existing_container(self):
        self.fake.create_disconnect = True
        job = self.worker.submit(self.spec)
        self.assertEqual(job["status"], "reconciling")
        self.assertEqual(self.worker.recover()[0]["status"], "running")
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)

    def test_disconnect_after_start_does_not_start_twice(self):
        self.fake.start_disconnect = True
        job = self.worker.submit(self.spec)
        self.assertEqual(job["status"], "reconciling")
        self.assertEqual(self.worker.recover()[0]["status"], "running")
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)

    def test_rejected_start_is_not_reported_running_and_preserves_container(self):
        self.fake.start_rejected = True
        job = self.worker.submit(self.spec)
        self.assertEqual(job["status"], "reconciling")
        self.fake.start_rejected = False
        self.assertEqual(self.worker.poll(job["job_id"])["status"], "running")
        self.assertEqual(self.fake.count("create"), 1)

    def test_adding_unrelated_registered_source_does_not_change_existing_job(self):
        job = self.worker.submit(self.spec)
        policy = replace(self.policy, sources={**self.policy.sources, "other": self.policy.sources["baseline"]})
        restarted = DockerWorker(self.root / "state", policy, runner=self.fake, clock=lambda: self.now)
        self.assertEqual(restarted.submit(self.spec)["job_id"], job["job_id"])
        self.assertEqual(self.fake.count("create"), 1)

    def test_security_policy_change_requires_restore_but_can_still_cancel(self):
        job = self.worker.submit(self.spec)
        policy = replace(self.policy, max_output_bytes=1_000_000)
        restarted = DockerWorker(self.root / "state", policy, runner=self.fake, clock=lambda: self.now)
        with self.assertRaises(ValueError):
            restarted.poll(job["job_id"])
        self.assertEqual(restarted.cancel(job["job_id"])["status"], "cancelled")

    def test_restart_waits_for_claim_lease_then_recovers(self):
        self.fake.create_disconnect = True
        job = self.worker.submit(self.spec)
        self.worker._update(job["job_id"], status="launching", lease_owner="dead-worker", lease_until=self.now + 60)
        restarted = DockerWorker(self.root / "state", self.policy, runner=self.fake, clock=lambda: self.now)
        self.assertEqual(restarted.poll(job["job_id"])["status"], "launching")
        self.now += 61
        self.assertEqual(restarted.poll(job["job_id"])["status"], "running")
        self.assertEqual(self.fake.count("create"), 1)

    def test_live_paused_launcher_cannot_be_replaced_after_lease_expiry(self):
        paused, release = threading.Event(), threading.Event()
        results, errors = [], []
        original_runner = self.fake
        pause_once = [True]
        def runner(args, timeout=10):
            result = original_runner(args, timeout=timeout)
            if args[:2] == ["container", "inspect"] and result.returncode == 0:
                values = json.loads(result.stdout)
                if values[0]["State"]["Status"] == "created" and pause_once[0]:
                    pause_once[0] = False
                    paused.set()
                    if not release.wait(10):
                        raise DockerError("test launcher pause expired")
            return result
        self.worker.runner = runner
        def submit():
            try:
                results.append(self.worker.submit(self.spec))
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=submit)
        thread.start()
        try:
            self.assertTrue(paused.wait(5))
            with self.worker._connect() as db:
                job_id = db.execute("SELECT job_id FROM docker_jobs").fetchone()[0]
            self.now += 61  # SQLite lease has expired while the old launcher is alive.
            restarted = DockerWorker(self.root / "state", self.policy, runner=self.fake, clock=lambda: self.now)
            self.assertEqual(restarted.poll(job_id)["status"], "launching")
            self.assertEqual(self.fake.count("start"), 0)
        finally:
            release.set()
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(results[0]["status"], "running")
        completed = self.finish(results[0])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)
        self.assertEqual(self.fake.containers[completed["container"]]["State"]["Status"], "exited")

    def test_os_lock_excludes_another_process_and_preserves_cancel_intent(self):
        job = self.worker.submit(self.spec)
        context = multiprocessing.get_context("spawn")
        ready, release = context.Event(), context.Event()
        process = context.Process(target=_hold_job_lock,
                                  args=(str(self.root / "state"), str(self.source), str(self.bundle.public), job["job_id"], ready, release))
        process.start()
        try:
            self.assertTrue(ready.wait(10), "child did not acquire job OS lock")
            before = self.fake.count("inspect")
            self.assertEqual(self.worker.poll(job["job_id"])["status"], "running")
            self.assertEqual(self.fake.count("inspect"), before)
            self.assertEqual(self.worker.cancel(job["job_id"])["status"], "stopping")
            self.assertEqual(self.fake.count("kill"), 0)
        finally:
            release.set()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(self.worker.poll(job["job_id"])["status"], "cancelled")
        self.assertEqual(self.fake.count("kill"), 1)

    def test_unknown_owned_name_is_never_started_killed_or_removed(self):
        self.fake.on_create = lambda container: container["Config"]["Labels"].update({"io.reclab.worker": "somebody-else"})
        with self.assertRaises(OwnershipError):
            self.worker.submit(self.spec)
        with self.worker._connect() as db:
            job_id = db.execute("SELECT job_id FROM docker_jobs").fetchone()[0]
        with self.assertRaises(OwnershipError):
            self.worker.cancel(job_id)
        self.assertEqual(self.fake.count("start"), 0)
        self.assertEqual(self.fake.count("kill"), 0)
        self.assertEqual(self.fake.count("rm"), 0)

    def test_missing_started_container_is_lost_and_not_automatically_relaunched(self):
        job = self.worker.submit(self.spec)
        del self.fake.containers[job["container"]]
        self.assertEqual(self.worker.poll(job["job_id"])["status"], "lost")
        self.worker.submit(self.spec)
        self.assertEqual(self.fake.count("create"), 1)

    def test_transport_error_is_not_misreported_as_stopped_or_missing(self):
        job = self.worker.submit(self.spec)
        self.fake.inspect_disconnect = True
        result = self.worker.poll(job["job_id"])
        self.assertEqual(result["status"], "running")
        self.assertIn("disconnect", result["reason"])

    def test_cancel_confirms_exit_and_terminal_status_is_stable(self):
        job = self.worker.submit(self.spec)
        result = self.worker.cancel(job["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.fake.containers[job["container"]]["State"]["Status"], "exited")
        self.assertEqual(self.worker.cancel(job["job_id"])["status"], "cancelled")
        self.assertEqual(self.fake.count("kill"), 1)

    def test_unconfirmed_cancel_remains_stopping_and_can_be_recovered(self):
        job = self.worker.submit(self.spec)
        self.fake.kill_confirmed = False
        self.assertEqual(self.worker.cancel(job["job_id"])["status"], "stopping")
        self.fake.kill_confirmed = True
        self.assertEqual(self.worker.recover()[0]["status"], "cancelled")

    def test_cancellation_during_creation_prevents_start(self):
        observed = []
        def cancel(container):
            job_id = container["Config"]["Labels"]["io.reclab.job"]
            observed.append(self.worker.cancel(job_id)["status"])
        self.fake.on_create = cancel
        result = self.worker.submit(self.spec)
        self.assertEqual(observed, ["stopping"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.fake.count("start"), 0)

    def test_queued_cancellation_does_not_create_container(self):
        self.worker.submit(self.spec)
        job = self.worker.submit(replace(self.spec, seed=43))
        self.assertEqual(self.worker.cancel(job["job_id"])["status"], "cancelled")
        self.assertEqual(self.fake.count("create"), 1)

    def test_timeout_kills_then_confirms_before_terminal_state(self):
        job = self.worker.submit(replace(self.spec, limits=ResourceLimits(seconds=2)))
        self.now += 3
        result = self.worker.poll(job["job_id"])
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["exit_code"], 137)
        self.assertEqual(self.fake.count("kill"), 1)

    def test_late_completed_exit_is_not_reported_success(self):
        job = self.worker.submit(replace(self.spec, limits=ResourceLimits(seconds=2)))
        self.now += 3
        self.assertEqual(self.finish(job)["status"], "timed_out")

    def test_nonzero_and_oom_are_recorded(self):
        result = self.finish(self.worker.submit(self.spec), code=137, oom=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "oom_killed")
        self.assertEqual(result["exit_code"], 137)

    def test_success_requires_artifacts_and_completed_receipt_is_immutable(self):
        job = self.worker.submit(self.spec)
        self.fake.containers[job["container"]]["State"].update(Status="exited", ExitCode=0)
        self.assertEqual(self.worker.poll(job["job_id"])["status"], "failed")
        self.worker._update(job["job_id"], status="running")
        self.assertEqual(self.worker.get(job["job_id"])["status"], "failed")

    def test_predict_mounts_frozen_training_output_read_only_and_json_is_graded_outside(self):
        job, expected = self.prediction()
        self.assertEqual(job["status"], "completed")
        collected = self.worker.collect_predictions(job["job_id"])
        self.assertEqual(collected, expected)
        self.assertEqual(self.bundle.grade(collected)["metrics"]["users"], 12)
        create = [call for call in self.fake.calls if call[:2] == ["container", "create"]][-1]
        self.assertTrue(any("target=/model,readonly" in value for value in create))
        self.assertEqual(create[-4:], ["--model", "/model", "--split", "valid"])
        backend = SourceBackend("baseline", ("src/train.py", "src/predict.py"), IMAGE)
        self.assertEqual(tuple(create[create.index(IMAGE) + 4:]), backend.prediction_command()[2:])

    def test_prediction_rejects_uncompleted_or_different_training_identity(self):
        training = self.worker.submit(self.spec)
        with self.assertRaises(ValueError):
            self.worker.submit(replace(self.spec, phase="predict", model_job_id=training["job_id"]))
        self.finish(training)
        with self.assertRaises(ValueError):
            self.worker.submit(replace(self.spec, phase="predict", seed=43, model_job_id=training["job_id"]))
        (self.worker._output(training["job_id"]) / "model.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.worker.submit(replace(self.spec, phase="predict", model_job_id=training["job_id"]))

    def test_json_cannot_smuggle_duplicate_keys_nonfinite_or_noninteger_predictions(self):
        for raw in ('{"u":[1],"u":[2]}', '{"u":[NaN]}', '{"u":[true]}', '{"u":"pickle"}', '{}'):
            with self.subTest(raw=raw):
                # Different seeds give independent immutable job identities.
                self.spec = replace(self.spec, seed=self.spec.seed + 1)
                job, _ = self.prediction(raw)
                with self.assertRaises(ValueError):
                    self.worker.collect_predictions(job["job_id"])

    def test_mutated_completed_predictions_are_not_collected(self):
        job, _ = self.prediction()
        (self.worker._output(job["job_id"]) / "predictions.json").write_text('{"u":[1]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.worker.collect_predictions(job["job_id"])

    def test_hardlinked_output_is_rejected(self):
        job = self.worker.submit(self.spec)
        outside = self.root / "not-an-artifact.txt"
        outside.write_text("private", encoding="utf-8")
        try:
            os.link(outside, self.worker._output(job["job_id"]) / "model.json")
        except OSError:
            self.skipTest("hard links not available")
        self.fake.containers[job["container"]]["State"].update(Status="exited", ExitCode=0)
        result = self.worker.poll(job["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("single-link", result["reason"])

    def test_log_collection_is_bounded_and_unknown_job_is_rejected(self):
        job = self.worker.submit(self.spec)
        self.assertEqual(self.worker.logs(job["job_id"]), "candidate log\n")
        with self.assertRaises(KeyError):
            self.worker.logs("unknown-job")


@unittest.skipUnless(sys.platform == "linux" and shutil.which("docker") and os.getenv("RECLAB_DOCKER_TEST_IMAGE"),
                     "real Linux Docker + explicitly preloaded approved digest required; mock tests are not isolation evidence")
class RealDockerSmokeTests(unittest.TestCase):
    def test_real_containers_boundaries_and_external_grading(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary = temporary_directory.name
        root = Path(temporary)
        source = root / "source"
        (source / "src").mkdir(parents=True)
        (source / "configs").mkdir()
        # This is a fixed, reviewed stdlib smoke recipe, not Agent-produced code.
        train = '''import json, os, pathlib, resource, socket
assert os.getuid() == 65532
status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert int(status['NoNewPrivs']) == 1
assert int(status['CapEff'], 16) == 0
assert int(status['Seccomp']) == 2
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (256, 256)
for path in ('/workspace/forbidden', '/data/forbidden', '/etc/reclab-forbidden'):
    try:
        pathlib.Path(path).write_text('unsafe')
    except OSError:
        pass
    else:
        raise AssertionError('write boundary failed: ' + path)
assert not pathlib.Path('/var/run/docker.sock').exists()
for path in ('/root/.ssh/id_rsa', '/data/private/test.json'):
    try:
        pathlib.Path(path).read_bytes()
    except (FileNotFoundError, PermissionError):
        pass
    else:
        raise AssertionError('private content is readable: ' + path)
connection = socket.socket()
connection.settimeout(0.5)
assert connection.connect_ex(('1.1.1.1', 53)) != 0
pathlib.Path('/output/model.json').write_text('{}')
'''
        predict = '''import collections, json, pathlib
data = pathlib.Path('/data')
train = json.loads((data/'train.json').read_text())
valid = json.loads((data/'valid.json').read_text())
catalog = json.loads((data/'manifest.json').read_text())['catalog']
counts = collections.Counter(item for items in train.values() for item in items)
ranking = sorted(catalog, key=lambda item: (-counts[item], item))
assert pathlib.Path('/model/model.json').is_file()
try:
    pathlib.Path('/model/model.json').write_text('unsafe')
except OSError:
    pass
else:
    raise AssertionError('model input is writable')
predictions = {user: [item for item in ranking if item not in row['history']][:20] for user, row in valid.items()}
pathlib.Path('/output/predictions.json').write_text(json.dumps(predictions))
'''
        (source / "src/train.py").write_text(train, encoding="utf-8")
        (source / "src/predict.py").write_text(predict, encoding="utf-8")
        (source / "configs/default.json").write_text("{}", encoding="utf-8")
        bundle = create_synthetic_bundle(root / "data")
        policy = WorkerPolicy(os.environ["RECLAB_DOCKER_TEST_IMAGE"], {"smoke": InputTree.freeze(source, "source")},
                              {"tiny": InputTree.freeze(bundle.public, "public_data")})
        worker = DockerWorker(root / "state", policy)
        jobs = []
        def cleanup():
            for job_id in jobs:
                job = worker.cancel(job_id)
                container = worker._inspect(job)  # Checks exact ownership.
                if container is not None:
                    self.assertIs(container['State']['Running'], False)
                    result = worker.runner(['container', 'rm', '--', container['Id']])
                    self.assertEqual(result.returncode, 0, result.stderr)
        self.addCleanup(cleanup)
        worker.capabilities()  # Failure is a failed opt-in test, never a silent skip.
        spec = JobSpec("smoke", "tiny", limits=ResourceLimits(memory_mb=256, seconds=30))
        def wait(job):
            for _ in range(60):
                if job["status"] in worker.TERMINAL:
                    return job
                time.sleep(0.5)
                job = worker.poll(job["job_id"])
            self.fail("real Docker smoke did not reach terminal state")
        training = worker.submit(spec)
        jobs.append(training['job_id'])
        training = wait(training)
        self.assertEqual(training["status"], "completed", (training, worker.logs(training["job_id"])))
        prediction = worker.submit(replace(spec, phase="predict", model_job_id=training["job_id"]))
        jobs.append(prediction['job_id'])
        prediction = wait(prediction)
        self.assertEqual(prediction["status"], "completed", prediction)
        report = bundle.grade(worker.collect_predictions(prediction["job_id"]))
        self.assertEqual(report["metrics"]["users"], 12)


if __name__ == "__main__":
    unittest.main()
