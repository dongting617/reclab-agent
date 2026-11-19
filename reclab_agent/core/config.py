"""Small, provider-independent limits shared by all agent strategies."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    model_retries: int = 2

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if type(self.model_retries) is not int or self.model_retries < 0:
            raise ValueError("model_retries cannot be negative")
