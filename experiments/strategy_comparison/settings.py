"""Protocol settings for this experiment; not an Agent configuration extension."""
from __future__ import annotations
from dataclasses import dataclass, fields
from reclab_agent.domain.contracts import Budget
from reclab_agent.domain.configuration import TaskConfiguration

@dataclass(frozen=True)
class ComparisonConfiguration:
    """Equal strategy budgets for independent searches from one baseline."""

    strategies: tuple[str, ...]
    candidates_per_strategy: int
    max_plan_steps: int = 3
    max_reflections: int = 1

    def __post_init__(self) -> None:
        allowed = {"react", "plan_execute", "reflection"}
        if (not isinstance(self.strategies, tuple) or not 2 <= len(self.strategies) <= 3
                or any(not isinstance(value, str) or value not in allowed for value in self.strategies)
                or len(set(self.strategies)) != len(self.strategies)):
            raise ValueError("comparison.strategies must contain two or three unique canonical strategies")
        if type(self.candidates_per_strategy) is not int or not 1 <= self.candidates_per_strategy <= 20:
            raise ValueError("comparison.candidates_per_strategy must be an integer between 1 and 20")
        if type(self.max_plan_steps) is not int or self.max_plan_steps < 1:
            raise ValueError("comparison.max_plan_steps must be a positive integer")
        if type(self.max_reflections) is not int or self.max_reflections < 0:
            raise ValueError("comparison.max_reflections must be a nonnegative integer")

    def validate_budget(self, budget: Budget) -> None:
        if not isinstance(budget, Budget):
            raise ValueError("comparison requires a global Budget")
        strategies = len(self.strategies)
        jobs = 1 + strategies * self.candidates_per_strategy
        if budget.max_training_jobs < jobs:
            raise ValueError("comparison budget must include one baseline and every strategy candidate")
        if budget.max_total_job_seconds < jobs * budget.max_job_seconds:
            raise ValueError("comparison total time must cover every configured experiment limit")
        if budget.max_model_turns < 1 + 2 * strategies:
            raise ValueError("comparison needs one API probe and at least two model turns per strategy")
        if budget.max_patches < strategies:
            raise ValueError("comparison needs at least one patch per strategy")

    def strategy_budget(self, budget: Budget) -> Budget:
        """Allocate identical limits; one global model call is reserved for setup."""
        self.validate_budget(budget)
        strategies = len(self.strategies)
        return Budget(
            max_model_turns=(budget.max_model_turns - 1) // strategies,
            max_patches=budget.max_patches // strategies,
            max_training_jobs=self.candidates_per_strategy,
            max_job_seconds=budget.max_job_seconds,
            max_total_job_seconds=self.candidates_per_strategy * budget.max_job_seconds,
            max_cost_cny=budget.max_cost_cny / strategies,
        )


@dataclass(frozen=True)
class ExperimentConfiguration(TaskConfiguration):
    """Experiment-local allocation; the product TaskConfiguration is unchanged."""
    comparison: ComparisonConfiguration | None = None


def with_comparison(configuration, comparison, **changes):
    values = {field.name: getattr(configuration, field.name) for field in fields(TaskConfiguration)}
    values.update(changes)
    return ExperimentConfiguration(**values, comparison=comparison)
