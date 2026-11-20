"""Host controls shared across rounds of an isolated strategy comparison."""
from __future__ import annotations

import copy
import re
import threading

from reclab_agent.contracts import content_hash
from reclab_agent.files import atomic_write, confined_path, read_regular_file


class ModelBudgetExceeded(RuntimeError):
    """The strategy has used its allowed backend request attempts."""


class CountedBackend:
    """Bound all backend attempts, including calls retried by an Agent session."""

    def __init__(self, backend, max_calls):
        if not callable(getattr(backend, "respond", None)):
            raise TypeError("a model backend is required")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self._backend, self._max_calls = backend, max_calls
        self._calls = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._lock = threading.Lock()

    @property
    def calls(self):
        with self._lock:
            return self._calls

    @property
    def usage(self):
        with self._lock:
            return dict(self._usage)

    def respond(self, **arguments):
        with self._lock:
            if self._calls >= self._max_calls:
                raise ModelBudgetExceeded("strategy model-call budget exhausted")
            self._calls += 1
        # Charge before calling: failed requests consume the same attempt limit.
        turn = self._backend.respond(**arguments)
        usage = getattr(turn, "usage", None)
        if isinstance(usage, dict):
            with self._lock:
                for name in self._usage:
                    value = usage.get(name)
                    if type(value) is int and value >= 0:
                        self._usage[name] += value
        return turn


class RoundResearch:
    """Expose one round while retaining the underlying strategy-wide budgets."""

    def __init__(self, research, candidate_id, seed, seen_source_hashes):
        research.plan.check_scope(candidate_id, tuple(research.workspace.editable_paths), seed)
        self._research = research
        self._candidate_id, self._seed = candidate_id, seed
        self._seen = frozenset(seen_source_hashes)
        self._attempts_before = research.usage["experiment_attempts"]

    def __getattr__(self, name):
        return getattr(self._research, name)

    @property
    def specs(self):
        specs = copy.deepcopy(self._research.specs)
        for spec in specs:
            if spec.name == "run_experiment":
                properties = spec.input_schema["properties"]
                properties["candidate_id"]["enum"] = [self._candidate_id]
                properties["seed"]["enum"] = [self._seed]
        return specs

    @staticmethod
    def _error(code, message):
        return {"ok": False, "tool": "run_experiment", "error": {"code": code, "message": message}}

    def invoke(self, name, arguments=None):
        if name != "run_experiment":
            return self._research.invoke(name, arguments)
        if (not isinstance(arguments, dict) or set(arguments) != {"candidate_id", "seed"}
                or not isinstance(arguments["candidate_id"], str) or type(arguments["seed"]) is not int):
            return self._error("INVALID_ARGUMENTS", "run_experiment requires candidate_id and an integer seed")
        if arguments["candidate_id"] != self._candidate_id or arguments["seed"] != self._seed:
            return self._error("OUTSIDE_ROUND", "use this round's candidate_id and seed")
        if self._research.usage["experiment_attempts"] >= self._attempts_before + 1:
            return self._error("ROUND_EXPERIMENT_LIMIT", "this round has already used its experiment attempt")
        try:
            snapshot = self._research.workspace.snapshot(self._research.snapshots_root)
        except (ValueError, OSError) as error:
            return self._error("INVALID_ARGUMENTS", str(error))
        if snapshot["source_hash"] in self._seen:
            return self._error("DUPLICATE_CANDIDATE", "this source has already been evaluated by this strategy")
        return self._research.invoke(name, arguments)


def _text(path):
    # Match pathlib's universal-newline text reads used by Workspace.source_hash.
    return read_regular_file(path, max_bytes=1_000_000).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def restore_snapshot(research, source_hash):
    """Restore an independently selected source without charging an Agent patch."""
    if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("source_hash must be a lower-case SHA256 digest")
    workspace = research.workspace
    directory = confined_path(research.snapshots_root, source_hash)
    files = {name: _text(confined_path(directory, name)) for name in sorted(workspace.source_paths)}
    if content_hash(files) != source_hash:
        raise ValueError("selected snapshot source hash does not match its contents")
    # Validate everything before writing any editable file. A controller must
    # not silently repair changes to a dependency outside the reviewed scope.
    for name in workspace.source_paths - workspace.editable_paths:
        if _text(confined_path(workspace.root, name)) != files[name]:
            raise ValueError("read-only source dependency changed")
    for name in sorted(workspace.editable_paths):
        atomic_write(confined_path(workspace.root, name), files[name].encode("utf-8"))
    restored = workspace.snapshot(research.snapshots_root)
    if restored["source_hash"] != source_hash:
        raise ValueError("restored workspace does not match the selected snapshot")
    return restored
