"""Docker-shaped fault injection fixtures; never execute candidate code."""

import hashlib
import json
import threading

from reclab_agent.execution import CommandResult, DockerError, DockerWorker, InputTree, WorkerPolicy


IMAGE = "example/reclab@sha256:" + "a" * 64


class FakeDocker:
    """Only models daemon outcomes and records arguments; does not run candidates."""
    def __init__(self):
        self.calls, self.containers = [], {}
        self.lock = threading.RLock()
        self.available = True
        self.security = ["name=seccomp,profile=builtin"]
        self.resource_limits = True
        self.image_digest, self.volumes = IMAGE, None
        self.create_disconnect = self.start_disconnect = self.inspect_disconnect = False
        self.start_rejected = False
        self.kill_confirmed = True
        self.on_create = None

    def __call__(self, args, timeout=10):
        with self.lock:
            self.calls.append(list(args))
            if not self.available:
                return CommandResult(1, stderr="cannot connect to Docker daemon")
            if args[0] == "info":
                return CommandResult(0, json.dumps({"OSType": "linux", "SecurityOptions": self.security, "Runtimes": {},
                                                    **{key: self.resource_limits for key in ("MemoryLimit", "SwapLimit", "CpuCfsPeriod", "CpuCfsQuota", "PidsLimit")}}))
            if args[:2] == ["image", "inspect"]:
                return CommandResult(0, json.dumps([{"RepoDigests": [self.image_digest], "Config": {"Volumes": self.volumes}}]))
            action = args[1]
            name = args[-1] if action != "create" else args[args.index("--name") + 1]
            if action in {"inspect", "kill"} and name not in self.containers:
                name = next((key for key, value in self.containers.items() if value.get("Id") == name), name)
            if action == "inspect":
                if self.inspect_disconnect:
                    raise DockerError("injected inspect disconnect")
                if name not in self.containers:
                    return CommandResult(1, stderr="Error: No such container: " + name)
                return CommandResult(0, json.dumps([self.containers[name]]))
            if action == "create":
                if name in self.containers:
                    return CommandResult(1, stderr="container name conflict")
                labels = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--label")
                self.containers[name] = {"Id": hashlib.sha256(name.encode()).hexdigest(),
                                         "Config": {"Labels": labels}, "State": {"Status": "created", "ExitCode": 0}}
                if self.on_create:
                    self.on_create(self.containers[name])
                if self.create_disconnect:
                    self.create_disconnect = False
                    raise DockerError("injected disconnect after create")
                return CommandResult(0, "fake-container-id")
            if action == "start":
                if self.start_rejected:
                    return CommandResult(1, stderr="injected Docker start failure")
                self.containers[name]["State"]["Status"] = "running"
                if self.start_disconnect:
                    self.start_disconnect = False
                    raise DockerError("injected disconnect after start")
                return CommandResult(0, name)
            if action == "kill":
                if self.kill_confirmed:
                    self.containers[name]["State"].update(Status="exited", ExitCode=137)
                    return CommandResult(0, name)
                return CommandResult(1, stderr="injected kill failure")
            if action == "logs":
                return CommandResult(0, "candidate log\n")
            raise AssertionError(f"unexpected Docker command: {args}")

    def count(self, action):
        return sum(call[:2] == ["container", action] for call in self.calls)


def _hold_job_lock(state, source, public, job_id, ready, release):
    """Separate trusted test process; never executes candidate source."""
    policy = WorkerPolicy(IMAGE, {"baseline": InputTree.freeze(source, "source")},
                          {"tiny": InputTree.freeze(public, "public_data")})
    worker = DockerWorker(state, policy, runner=FakeDocker())
    with worker._job_lock(job_id) as acquired:
        if acquired:
            ready.set()
            release.wait(15)


class WatchdogFakeDocker(FakeDocker):
    """Docker-shaped immutable IDs; records actual ID-targeted stop calls."""
    def __call__(self, args, timeout=10):
        result = super().__call__(args, timeout=timeout)
        if args[:2] == ["container", "create"]:
            name = args[args.index("--name") + 1]
            container = self.containers[name]
            container["Id"] = hashlib.sha256(name.encode()).hexdigest()
            self.containers[container["Id"]] = container
        return result
