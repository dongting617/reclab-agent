"""The recommendation backend boundary is independent of orchestration/providers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from reclab_agent.domain.contracts import identifier


class RecommendationBackend(Protocol):
    backend_id: str

    def inspect(self) -> dict[str, Any]: ...

    def training_command(self, config_name: str = "default") -> tuple[str, ...]: ...

    def prediction_command(self, split: str = "valid", *, config_name: str = "default") -> tuple[str, ...]: ...


@dataclass(frozen=True)
class SourceBackend:
    """Candidate entrypoints run inside Docker, never imported into the controller."""
    backend_id: str
    source_files: tuple[str, ...]
    image: str

    def __post_init__(self):
        identifier(self.backend_id)
        if not self.source_files or "src/train.py" not in self.source_files or "src/predict.py" not in self.source_files:
            raise ValueError("source backend requires src/train.py and src/predict.py")

    def inspect(self) -> dict[str, Any]:
        return {"backend_id": self.backend_id, "source_files": list(self.source_files),
                "image": self.image,
                "contract": "train writes /output; predict reads frozen /model and config, writes /output/predictions.json"}

    def training_command(self, config_name: str = "default") -> tuple[str, ...]:
        identifier(config_name)
        return ("python", "-B", "/workspace/src/train.py", "--data", "/data", "--output", "/output",
                "--config", f"/workspace/configs/{config_name}.json")

    def prediction_command(self, split: str = "valid", *, config_name: str = "default") -> tuple[str, ...]:
        if split != "valid":
            raise ValueError("final test prediction is an explicit trusted freeze operation")
        identifier(config_name)
        return ("python", "-B", "/workspace/src/predict.py", "--data", "/data", "--output", "/output",
                "--config", f"/workspace/configs/{config_name}.json", "--model", "/model", "--split", split)
