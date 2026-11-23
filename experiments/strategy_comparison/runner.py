"""Experiment-local deadline policy and verified reuse of the prepared baseline."""
from __future__ import annotations
import time
from reclab_agent.domain.contracts import content_hash
from reclab_agent.application.binding import bind_task_tools
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.research import ExperimentRecord
from reclab_agent.application.launch import LaunchError
from reclab_agent.experiments.remote_runner import RemoteExperimentRunner


class ExperimentRemoteRunner(RemoteExperimentRunner):
    """Keep remote 480s/120s limits while excluding control-host suspension.

    The server supervisor still bounds the train/predict process groups. CPU time
    is only the supplemental controller clock; wall time and server durations are
    recorded separately, so a suspended desktop does not discard trained weights.
    """

    def __call__(self, snapshot, *, candidate_id, seed, max_seconds):
        reference_path = self.config.output_directory
        while reference_path.name != "sc" and reference_path != reference_path.parent:
            reference_path = reference_path.parent
        reference_path = reference_path / "baseline-reference.json"
        reference = None
        if candidate_id == self.config.task.baseline_id and reference_path.is_file():
            reference = read_json_strict(reference_path, max_bytes=1_000_000)
            old = read_json_strict(reference["record_path"], max_bytes=1_000_000)
            try:
                ExperimentRecord.from_dict(old)
            except (ValueError, TypeError, KeyError):
                raise LaunchError("prepared baseline record is invalid; refusing implicit retraining") from None
            source, config_hash = self._snapshot(snapshot)
            environment = self.tools.bootstrap()
            connection = self.tools.connection
            details = {"environment": environment, "host": connection.host,
                       "port": connection.port, "remote_root": connection.remote_root}
            expected = "exp-" + content_hash({
                "plan": self.plan.plan_hash, "source": snapshot["source_hash"],
                "source_tree": source.sha256, "public_tree": self.public.sha256,
                "candidate": candidate_id, "seed": seed, "seconds": max_seconds,
                "runtime": content_hash(details),
            })[:32]
            required = {"experiment_id": expected, "plan_hash": self.plan.plan_hash,
                        "source_hash": snapshot["source_hash"], "config_hash": config_hash,
                        "dataset_id": self.dataset.dataset_id, "protocol_hash": self.plan.protocol_hash,
                        "candidate_id": candidate_id, "seed": seed,
                        "training_job_id": expected + "-train",
                        "runtime": {"backend": "ssh_tmux", "fingerprint": content_hash(details), "details": details}}
            if any(old.get(key) != value for key, value in required.items()):
                raise LaunchError("prepared baseline identities differ; refusing to retrain it implicitly")
            job = self.tools.job_status(old["training_job_id"])
            if job.get("status") != "completed" or job.get("exit_code") != 0 or job.get("stop_confirmed") is not True:
                raise LaunchError("prepared baseline training is not confirmed complete")
            print("已核验并复用已完成的基线训练作业，继续预测与评分。", flush=True)
        started = time.monotonic()
        record = super().__call__(snapshot, candidate_id=candidate_id, seed=seed, max_seconds=max_seconds)
        execution_path = self.config.output_directory / "experiments" / record.experiment_id / "execution.json"
        if execution_path.exists():
            execution = read_json_strict(execution_path)
            execution["controller_cpu_seconds"] = execution["elapsed_seconds"]
            execution["elapsed_seconds"] = time.monotonic() - started
            execution["supplemental_controller_clock"] = "process_cpu_excluding_desktop_suspension"
            execution["remote_phase_timeout_seconds"] = {"train": max_seconds * 4 // 5,
                                                          "predict": max_seconds - max_seconds * 4 // 5}
            execution["server_job_seconds"] = sum(
                item.get("finished_at", 0) - item.get("started_at", 0)
                for item in execution["observations"] if item["operation"] == "terminal"
                and item.get("finished_at") is not None and item.get("started_at") is not None)
            if reference is not None:
                execution["reused_baseline_training"] = reference
            self._write(f"experiments/{record.experiment_id}/execution.json", execution)
        return record


def bind_experiment_tools(configuration, *, plan, review, registry, remote_tools=None):
    research = bind_task_tools(configuration, plan=plan, review=review,
                               registry=registry, remote_tools=remote_tools)
    if configuration.execution_backend == "ssh_tmux":
        research._runner = ExperimentRemoteRunner(configuration, plan, remote_tools, registry=registry,
                                                   clock=time.process_time)
    return research
