"""Explicit opt-in real Docker acceptance; mocks never count as this evidence.

RECLAB_REAL_DOCKER=1 requires Linux, the local daemon, and an explicitly preloaded
official Python digest. Once requested, unavailable capabilities FAIL the suite.
Reviewed stdlib programs below are only executed by DockerWorker inside Docker.
"""
from dataclasses import replace
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from reclab_agent.datasets import create_synthetic_bundle, popularity_predictions
from reclab_agent.execution import CommandResult, DockerError, DockerWorker, InputTree, JobSpec, ResourceLimits, WorkerPolicy


_PREPARE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_worker_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("reclab_acceptance_prepare", _PREPARE_PATH)
_PREPARE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPARE)


TRAIN = r'''import collections, errno, json, os, pathlib, resource, socket, subprocess, sys, time
root = pathlib.Path('/output')
config = json.loads(pathlib.Path('/workspace/configs/default.json').read_text())
status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert os.getuid() == os.getgid() == 65532
assert int(status['NoNewPrivs']) == 1
assert int(status['CapEff'], 16) == 0
assert int(status['Seccomp']) == 2
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (256, 256)
denied = {}
for name in ('/workspace/forbidden', '/data/forbidden', '/etc/reclab-forbidden'):
    try:
        pathlib.Path(name).write_text('unsafe')
    except OSError as error:
        assert error.errno in (errno.EROFS, errno.EACCES, errno.EPERM), repr(error)
        denied[name] = error.errno
    else:
        raise AssertionError('write boundary failed: ' + name)
hidden = ['/var/run/docker.sock', '/root/.ssh/id_rsa', '/data/test.json', '/data/private/test.json',
          '/data/../private/test.json', config['host_test_path'], config['host_sentinel_path']]
for name in hidden:
    try:
        with open(name, 'rb') as handle:
            handle.read(1)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        pass
    else:
        raise AssertionError('host/private input visible: ' + name)
assert set(pathlib.Path('/sys/class/net').iterdir()) == {pathlib.Path('/sys/class/net/lo')}
with socket.socket() as connection:
    connection.settimeout(.5)
    assert connection.connect_ex(('1.1.1.1', 53)) != 0
def cgroup_value(v2, v1):
    for name in (v2, v1):
        path = pathlib.Path('/sys/fs/cgroup') / name
        if path.is_file():
            return path.read_text().strip()
    raise AssertionError('cgroup resource file unavailable: ' + v2)
memory = cgroup_value('memory.max', 'memory/memory.limit_in_bytes')
pids = cgroup_value('pids.max', 'pids/pids.max')
try:
    quota, period = map(int, (pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()))
except FileNotFoundError:
    quota = int(cgroup_value('cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_quota_us'))
    period = int(cgroup_value('cpu/cpu.cfs_period_us', 'cpu/cpu.cfs_period_us'))
assert int(memory) == config['memory_bytes']
assert int(pids) == config['pids']
assert abs(quota / period - config['cpus']) < .000001
probe = {'uid': os.getuid(), 'gid': os.getgid(), 'denied_writes': denied,
         'inaccessible_paths': hidden, 'no_new_privs': status['NoNewPrivs'].strip(),
         'cap_eff': status['CapEff'].strip(), 'seccomp': status['Seccomp'].strip(),
         'memory_bytes': int(memory), 'pids_max': int(pids), 'cpu_quota': quota, 'cpu_period': period}
(root/'isolation.json').write_text(json.dumps(probe))
mode = config['mode']
if mode == 'sleep':
    (root/'started.json').write_text('{}')
    time.sleep(60)
elif mode == 'oom':
    blocks = []
    # At most 256 MiB; the actual cgroup limit is only 64 MiB.
    for index in range(32):
        blocks.append(bytearray(8 * 1024 * 1024))
        time.sleep(.01)
    raise AssertionError('memory cgroup did not kill bounded allocation')
elif mode == 'resources':
    children, rejected = [], None
    try:
        # At most 10 direct children, no recursive spawning or fork bomb.
        for index in range(10):
            try:
                children.append(subprocess.Popen([sys.executable, '-E', '-s', '-c', 'import time; time.sleep(10)']))
            except OSError as error:
                assert error.errno == errno.EAGAIN, repr(error)
                rejected = error.errno
                break
        assert rejected == errno.EAGAIN and len(children) < 10
    finally:
        for child in children:
            child.kill()
        for child in children:
            child.wait(timeout=3)
    stat_path = pathlib.Path('/sys/fs/cgroup/cpu.stat')
    if not stat_path.is_file():
        stat_path = pathlib.Path('/sys/fs/cgroup/cpu/cpu.stat')
    before = dict(line.split() for line in stat_path.read_text().splitlines())
    cpu_start, wall_start = time.process_time(), time.monotonic()
    value = 0
    while time.process_time() - cpu_start < .4:
        value += 1
    after = dict(line.split() for line in stat_path.read_text().splitlines())
    cpu_used, wall_used = time.process_time() - cpu_start, time.monotonic() - wall_start
    assert int(after['nr_throttled']) > int(before['nr_throttled'])
    assert wall_used > cpu_used * 1.3
    (root/'resources.json').write_text(json.dumps({'children_started': len(children), 'spawn_errno': rejected,
        'bounded_spawn_attempts': len(children) + 1, 'nr_throttled_delta': int(after['nr_throttled']) - int(before['nr_throttled']),
        'cpu_seconds': cpu_used, 'wall_seconds': wall_used}))
train = json.loads(pathlib.Path('/data/train.json').read_text())
counts = collections.Counter(item for items in train.values() for item in items)
# This is a real, tiny CPU popularity fit, never a neural training or quality claim.
(root/'model.json').write_text(json.dumps({'counts': counts, 'seed': int(os.environ['RECLAB_SEED'])}))
'''


PREDICT = r'''import errno, json, pathlib
root = pathlib.Path('/output')
model = pathlib.Path('/model/model.json')
weights = json.loads(model.read_text())
try:
    model.write_text('unsafe')
except OSError as error:
    assert error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
    denied = error.errno
else:
    raise AssertionError('training output was writable during prediction')
data = pathlib.Path('/data')
valid = json.loads((data/'valid.json').read_text())
catalog = json.loads((data/'manifest.json').read_text())['catalog']
ranking = sorted(catalog, key=lambda item: (-weights['counts'].get(str(item), 0), item))
predictions = {user: [item for item in ranking if item not in row['history']][:20] for user, row in valid.items()}
(root/'predictions.json').write_text(json.dumps(predictions))
(root/'model-readonly.json').write_text(json.dumps({'write_errno': denied}))
# Deliberately false candidate self-report. The trusted scorer must ignore it.
(root/'candidate-metrics.json').write_text(json.dumps({'ndcg@20': 1.0}))
'''


class AcceptancePreparationTests(unittest.TestCase):
    """Local contract checks; explicitly not evidence that a container ran."""
    def test_only_explicit_official_python_digests_are_accepted(self):
        digest = "python@sha256:" + "a" * 64
        self.assertEqual(_PREPARE.approved_digest(digest), digest)
        ecr = "public.ecr.aws/docker/library/python@sha256:" + "b" * 64
        self.assertEqual(_PREPARE.approved_digest(ecr), ecr)
        for value in ("python:3.12-slim", "untrusted/python@sha256:" + "a" * 64,
                      "python@sha256:" + "z" * 64, digest + "\n", "",
                      "public.ecr.aws/docker/library/python:3.12-slim",
                      "public.ecr.aws/untrusted/library/python@sha256:" + "a" * 64,
                      "public.ecr.aws/docker/library/python-extra@sha256:" + "a" * 64,
                      "mirror.example/library/python@sha256:" + "a" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _PREPARE.approved_digest(value)

    def test_reviewed_programs_parse_without_executing_them_on_host(self):
        compile(TRAIN, "reviewed-train.py", "exec")
        compile(PREDICT, "reviewed-predict.py", "exec")

    def test_explicit_opt_in_with_missing_digest_fails_instead_of_skipping(self):
        with patch.dict(os.environ, {"RECLAB_REAL_DOCKER": "1", "RECLAB_DOCKER_TEST_IMAGE": ""}):
            with self.assertRaises(ValueError):
                RealDockerAcceptanceTests.setUpClass()

    def test_ecr_provenance_retains_actual_repository_and_digest(self):
        ecr = "public.ecr.aws/docker/library/python@sha256:" + "b" * 64
        images = [{"RepoDigests": [ecr], "Id": "sha256:" + "c" * 64,
                   "Config": {}, "Os": "linux", "Architecture": "amd64"}]
        # Preparation metadata contract only; this is not a Docker execution.
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(_PREPARE, "docker_json", side_effect=[images, {"OSType": "linux"}]):
            provenance = _PREPARE.prepare(ecr, Path(temporary))
        self.assertEqual(provenance["requested_image"], ecr)
        self.assertEqual(provenance["repo_digests"], [ecr])
        self.assertEqual(provenance["image_repository"], "public.ecr.aws/docker/library/python")
        self.assertEqual(provenance["image_registry"], "public.ecr.aws")
        self.assertEqual(provenance["image_source"], "https://gallery.ecr.aws/docker/library/python")
        self.assertIs(provenance["real_container_tested"], False)

    def test_log_failure_saves_inspection_and_cleans_owned_stopped_container_but_still_fails(self):
        # Exercise only the trusted test harness, not a container or candidate.
        case = RealDockerAcceptanceTests("test_real_cancel_confirms_exit")
        job = {"job_id": "job-reviewed", "container": "reclab-reviewed", "status": "timed_out"}
        labels = {"io.reclab.worker": "owner", "io.reclab.job": "job-reviewed", "io.reclab.spec": "spec"}
        container = {"Id": "a" * 64, "Config": {"Labels": labels},
                     "State": {"Running": False, "Status": "created", "ExitCode": 128,
                               "Error": "failed to initialize logging driver"}}
        def runner(args):
            return (CommandResult(0, "removed") if args[:2] == ["container", "rm"] else
                    CommandResult(1, stderr="compression cannot be enabled when max file count is 1"))
        worker = SimpleNamespace(TERMINAL=DockerWorker.TERMINAL, get=lambda _: job,
                                 _labels=lambda _: labels,
                                 logs=Mock(side_effect=DockerError("failed to collect container logs")),
                                 runner=Mock(side_effect=runner))
        with tempfile.TemporaryDirectory() as temporary:
            case.root, case.worker, case.jobs = Path(temporary), worker, [job["job_id"]]
            case.evidence_errors = []
            case.inspect = lambda _: container
            with self.assertRaisesRegex(AssertionError, "log collection failed"):
                case.tearDown()
            evidence = json.loads((Path(temporary) / "job-reviewed-cleanup.json").read_text())
            self.assertEqual(evidence["inspect"]["State"]["ExitCode"], 128)
            self.assertIn("compression", evidence["log_command_result"]["stderr"])
            self.assertTrue(any(call.args[0] == ["container", "rm", "--", "a" * 64]
                                for call in worker.runner.call_args_list))


@unittest.skipUnless(os.environ.get("RECLAB_REAL_DOCKER"), "real Docker acceptance disabled; enable RECLAB_REAL_DOCKER=1 explicitly")
class RealDockerAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RECLAB_REAL_DOCKER") != "1":
            raise RuntimeError("RECLAB_REAL_DOCKER must equal 1; an invalid opt-in must not pass")
        cls.image = _PREPARE.approved_digest(os.environ.get("RECLAB_DOCKER_TEST_IMAGE", ""))
        base = Path(os.environ.get("RECLAB_ACCEPTANCE_EVIDENCE", ".reclab/worker-acceptance"))
        cls.evidence = base.absolute() / ("run-" + uuid.uuid4().hex)
        cls.provenance = _PREPARE.prepare(cls.image, cls.evidence)
        (base.absolute() / "latest-run.txt").write_text(str(cls.evidence) + "\n", encoding="utf-8")

    def setUp(self):
        self.root = self.evidence / self._testMethodName
        self.root.mkdir()
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        self.sentinel = self.root / "host-private-sentinel.txt"
        self.sentinel.write_text("synthetic non-secret host sentinel", encoding="utf-8")
        self.worker = None
        self.evidence_errors = []

    def make_worker(self, mode="fit", *, limits=None):
        limits = limits or ResourceLimits(cpus=.5, memory_mb=128, pids=16, seconds=30)
        source = self.root / "source"
        (source / "src").mkdir(parents=True)
        (source / "configs").mkdir()
        (source / "src/train.py").write_text(TRAIN, encoding="utf-8")
        (source / "src/predict.py").write_text(PREDICT, encoding="utf-8")
        config = {"mode": mode, "acceptance_run": uuid.uuid4().hex,
                  "memory_bytes": limits.memory_mb * 1024 * 1024, "cpus": limits.cpus, "pids": limits.pids,
                  "host_test_path": str(self.bundle.root.absolute() / "private/test.json"),
                  "host_sentinel_path": str(self.sentinel.absolute())}
        (source / "configs/default.json").write_text(json.dumps(config), encoding="utf-8")
        self.policy = WorkerPolicy(self.image, {"reviewed": InputTree.freeze(source, "source")},
                                   {"fixture": InputTree.freeze(self.bundle.public, "public_data")})
        self.worker = DockerWorker(self.root / "state", self.policy)
        self.worker.capabilities()
        self.spec = JobSpec("reviewed", "fixture", limits=limits)
        self.jobs = []
        return self.worker

    def submit(self, spec=None):
        job = self.worker.submit(spec or self.spec)
        self.jobs.append(job["job_id"])
        return job

    def wait(self, job, seconds=40):
        stop = time.monotonic() + seconds
        while time.monotonic() < stop:
            if job["status"] in self.worker.TERMINAL:
                self.record(job)
                return job
            time.sleep(.2)
            job = self.worker.poll(job["job_id"])
        self.fail("real job did not reach terminal state before acceptance deadline: " + json.dumps(job))

    def inspect(self, job):
        result = self.worker.runner(["container", "inspect", "--", job["container"]])
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)[0]

    def record(self, job, suffix="terminal"):
        container = self.inspect(job)
        value = {"job": job, "inspect": container, "logs": None}
        path = self.root / f"{job['job_id']}-{suffix}.json"
        # Inspection is the primary failure evidence. Save it before attempting
        # logs so a log-driver/CLI failure cannot erase the daemon's real State.
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        try:
            value["logs"] = self.worker.logs(job["job_id"])
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            value["log_error"] = detail
            self.evidence_errors.append(f"{job['job_id']}: log collection failed: {detail}")
            # Read-only diagnostics preserve the CLI's original stderr rather
            # than replacing an actual daemon compatibility problem with a pass.
            try:
                raw = self.worker.runner(["container", "logs", "--tail", "200", "--", job["container"]])
                value["log_command_result"] = {"returncode": raw.returncode,
                                               "stdout": raw.stdout[-64000:], "stderr": raw.stderr[-64000:]}
            except Exception as diagnostic_error:
                value["log_command_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        finally:
            path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        return container

    def assert_actual_limits(self, container, limits, *, predict=False):
        host = container["HostConfig"]
        self.assertEqual(container["Config"]["User"], "65532:65532")
        self.assertEqual(container["Config"]["Image"], self.image)
        self.assertEqual(host["NetworkMode"], "none")
        self.assertIs(host["ReadonlyRootfs"], True)
        self.assertIs(host["Privileged"], False)
        self.assertEqual(host["NanoCpus"], round(limits.cpus * 1_000_000_000))
        self.assertEqual(host["Memory"], limits.memory_mb * 1024 * 1024)
        self.assertEqual(host["MemorySwap"], host["Memory"])
        self.assertEqual(host["PidsLimit"], limits.pids)
        self.assertIn("ALL", host["CapDrop"])
        self.assertIn("no-new-privileges=true", host["SecurityOpt"])
        self.assertEqual(host["IpcMode"], "private")
        self.assertTrue(host["Init"])
        mounts = {entry["Destination"]: entry for entry in container["Mounts"]}
        expected = {"/workspace", "/data", "/output"} | ({"/model"} if predict else set())
        # Docker versions differ in whether --tmpfs is repeated in Mounts or
        # only in HostConfig.Tmpfs. Neither representation permits another bind.
        self.assertEqual({name for name, entry in mounts.items() if entry["Type"] == "bind"}, expected)
        self.assertFalse(set(mounts) - expected - {"/tmp"})
        if "/tmp" in mounts:
            self.assertEqual(mounts["/tmp"]["Type"], "tmpfs")
        for option in ("noexec", "nosuid", "nodev", "size=64m"):
            self.assertIn(option, host["Tmpfs"]["/tmp"].split(","))
        for path in ("/workspace", "/data") + (("/model",) if predict else ()):
            self.assertIs(mounts[path]["RW"], False, path)
        self.assertIs(mounts["/output"]["RW"], True)
        self.assertEqual(mounts["/data"]["Source"], str(self.bundle.public.absolute()))
        self.assertFalse(container["State"]["Running"])

    def tearDown(self):
        if not self.worker:
            return
        # Only this worker's exactly labelled, known containers can be cleaned.
        # Record first; cancel running jobs and require confirmed stop before rm.
        errors = []
        for job_id in dict.fromkeys(self.jobs):
            try:
                job = self.worker.get(job_id)
                if job["status"] not in self.worker.TERMINAL:
                    job = self.worker.cancel(job_id)
                container = self.record(job, "cleanup")
                if container["State"].get("Running"):
                    raise AssertionError("cleanup cannot remove a still-running container")
                if container["Config"].get("Labels", {}) != self.worker._labels(job):
                    # Official image may add unrelated labels; compare exact required keys.
                    labels = container["Config"].get("Labels", {})
                    if any(labels.get(key) != value for key, value in self.worker._labels(job).items()):
                        raise AssertionError("cleanup ownership differs")
                result = self.worker.runner(["container", "rm", "--", container["Id"]])
                if result.returncode:
                    raise AssertionError("stopped container cleanup failed: " + result.stderr)
            except Exception as error:
                errors.append(f"{job_id}: {type(error).__name__}: {error}")
        if errors:
            self.fail("; ".join(dict.fromkeys(errors + self.evidence_errors)))
        if self.evidence_errors:
            self.fail("; ".join(dict.fromkeys(self.evidence_errors)))

    def test_real_fit_predict_boundaries_and_independent_grader(self):
        self.make_worker()
        training = self.wait(self.submit())
        self.assertEqual(training["status"], "completed", training)
        self.assert_actual_limits(self.inspect(training), self.spec.limits)
        model = self.worker._output(training["job_id"]) / "model.json"
        before = hashlib.sha256(model.read_bytes()).hexdigest()
        prediction = self.wait(self.submit(replace(self.spec, phase="predict", model_job_id=training["job_id"])))
        self.assertEqual(prediction["status"], "completed", prediction)
        self.assert_actual_limits(self.inspect(prediction), self.spec.limits, predict=True)
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), before)
        collected = self.worker.collect_predictions(prediction["job_id"])
        self.assertEqual(collected, popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid")))
        report = self.bundle.grade(collected)
        (self.root / "trusted-grade.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self.assertEqual(report["metrics"]["users"], 12)
        self.assertEqual(report["metrics"]["ndcg@20"], 0.0)
        candidate_claim = json.loads((self.worker._output(prediction["job_id"]) / "candidate-metrics.json").read_text())
        self.assertNotEqual(report["metrics"]["ndcg@20"], candidate_claim["ndcg@20"])
        # Frozen receipts reject later host-side mutation; they are not a disk quota.
        (self.worker._output(prediction["job_id"]) / "predictions.json").write_text('{}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.worker.collect_predictions(prediction["job_id"])

    def test_real_duplicate_submit_and_worker_reconstruction_share_one_container(self):
        self.make_worker("sleep")
        first = self.submit()
        before = self.inspect(first)
        self.assertTrue(before["State"]["Running"])
        self.worker = DockerWorker(self.root / "state", self.policy)
        second = self.submit()
        after = self.inspect(second)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(before["Id"], after["Id"])
        self.assertEqual(before["State"]["StartedAt"], after["State"]["StartedAt"])
        found = self.worker.runner(["container", "ls", "--all", "--filter", f"label=io.reclab.worker={self.worker.owner}", "--format", "{{.ID}}"])
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(len(found.stdout.split()), 1)
        self.assertEqual(self.worker.recover()[0]["job_id"], first["job_id"])
        stopped = self.worker.cancel(first["job_id"])
        self.assertEqual(stopped["status"], "cancelled", stopped)
        self.assertFalse(self.record(stopped)["State"]["Running"])

    def test_real_cancel_confirms_exit(self):
        self.make_worker("sleep")
        job = self.submit()
        self.assertTrue(self.inspect(job)["State"]["Running"])
        stopped = self.worker.cancel(job["job_id"])
        self.assertEqual(stopped["status"], "cancelled", stopped)
        self.assertEqual(stopped["exit_code"], 137)
        self.assertFalse(self.record(stopped)["State"]["Running"])

    def test_real_deadline_is_enforced_by_running_supervisor(self):
        import threading
        self.make_worker("sleep", limits=ResourceLimits(cpus=.5, memory_mb=128, pids=16, seconds=4))
        job = self.submit()
        self.assertTrue(self.inspect(job)["State"]["Running"])
        stop = threading.Event()
        errors = []
        def supervise():
            try:
                self.worker.serve(interval=.1, stop=stop)
            except Exception as error:
                errors.append(repr(error))
        thread = threading.Thread(target=supervise, daemon=True)
        thread.start()
        try:
            end = time.monotonic() + 15
            # Only read SQLite: the caller does not poll/reconcile on the supervisor's behalf.
            while time.monotonic() < end and self.worker.get(job["job_id"])["status"] not in self.worker.TERMINAL:
                time.sleep(.2)
        finally:
            stop.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        result = self.worker.get(job["job_id"])
        self.assertEqual(result["status"], "timed_out", result)
        self.assertFalse(self.record(result)["State"]["Running"])

    def test_real_cpu_pid_and_cgroup_limits(self):
        self.make_worker("resources", limits=ResourceLimits(cpus=.25, memory_mb=128, pids=8, seconds=30))
        result = self.wait(self.submit())
        self.assertEqual(result["status"], "completed", result)
        self.assert_actual_limits(self.inspect(result), self.spec.limits)
        resources = json.loads((self.worker._output(result["job_id"]) / "resources.json").read_text())
        self.assertEqual(resources["spawn_errno"], errno.EAGAIN)
        self.assertLess(resources["children_started"], 10)
        self.assertLessEqual(resources["bounded_spawn_attempts"], 10)
        self.assertGreater(resources["nr_throttled_delta"], 0)

    def test_real_independent_watchdog_stops_while_worker_job_lock_is_held(self):
        import threading
        from reclab_agent.watchdog import DockerDeadlineWatchdog
        self.make_worker("sleep", limits=ResourceLimits(cpus=.5, memory_mb=128, pids=16, seconds=5))
        job = self.submit()
        self.assertTrue(self.inspect(job)["State"]["Running"])
        ready, release = threading.Event(), threading.Event()
        errors = []
        def hold_worker_lock():
            try:
                with self.worker._job_lock(job["job_id"]) as acquired:
                    if not acquired:
                        raise AssertionError("test could not hold Worker job lock")
                    ready.set()
                    if not release.wait(20):
                        raise AssertionError("test did not release Worker job lock")
            except Exception as error:
                errors.append(repr(error))
        thread = threading.Thread(target=hold_worker_lock, daemon=True)
        thread.start()
        observation = None
        try:
            self.assertTrue(ready.wait(3))
            watchdog = DockerDeadlineWatchdog(self.root / "state", self.policy)
            stop = time.monotonic() + 15
            while time.monotonic() < stop:
                observations = watchdog.tick()
                observation = next((value for value in observations if value["job_id"] == job["job_id"]), None)
                if observation and observation.get("stop_confirmed"):
                    break
                time.sleep(.2)
            (self.root / "watchdog-confirmation.json").write_text(json.dumps(observation, indent=2, sort_keys=True), encoding="utf-8")
            self.assertIsNotNone(observation, "watchdog did not observe the expired job")
            self.assertTrue(observation["stop_confirmed"], observation)
            self.assertTrue(observation["launcher_busy"], observation)
            self.assertEqual(self.worker.get(job["job_id"])["status"], "stopping")
            self.assertFalse(self.record(self.worker.get(job["job_id"]), "watchdog-stopped")["State"]["Running"])
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        resumed = self.worker.recover()[0]
        self.assertEqual(resumed["status"], "timed_out", resumed)

    def test_real_bounded_memory_allocation_is_oom_killed(self):
        self.make_worker("oom", limits=ResourceLimits(cpus=.5, memory_mb=64, pids=16, seconds=30))
        result = self.wait(self.submit())
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["reason"], "oom_killed", result)
        self.assertEqual(result["exit_code"], 137)
        inspected = self.inspect(result)
        self.assert_actual_limits(inspected, self.spec.limits)
        self.assertIs(inspected["State"]["OOMKilled"], True)


if __name__ == "__main__":
    unittest.main()
