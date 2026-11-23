"""Durable model/network allowances and the shared task deadline."""
from __future__ import annotations
from pathlib import Path
import time
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.tools.contracts import ToolError
from reclab_agent.domain.configuration import read_json_strict


class RunResources:
    def __init__(self, output, config, *, resume=False):
        self.path = Path(output) / "resources.json"
        self.config = config
        self.project = None
        if resume and self.path.exists():
            self.state = read_json_strict(self.path, max_bytes=100_000)
        else:
            self.state = {"deadline": time.time() + (config.max_wall_seconds or
                config.task.budget.max_total_job_seconds + 60 * config.task.budget.max_model_turns),
                "model_requests": 0, "network_requests": 0}

    def check(self):
        if time.time() >= self.state["deadline"]:
            raise ToolError("BUDGET_EXCEEDED", "task deadline exceeded")

    def save(self):
        atomic_write(self.path, canonical_json(self.state).encode("utf-8"))
        if self.project:
            self.project.event(self.path.parent.name, "resource_reservation", self.state)

    def model_request(self, count):
        self.check()
        self.state["model_requests"] = count
        self.save()

    def network_request(self):
        self.check()
        if self.state["network_requests"] >= self.config.max_network_requests:
            raise ToolError("BUDGET_EXCEEDED", "network request allowance exhausted")
        self.state["network_requests"] += 1
        self.save()
        return min(30, max(.1, self.state["deadline"] - time.time()))
