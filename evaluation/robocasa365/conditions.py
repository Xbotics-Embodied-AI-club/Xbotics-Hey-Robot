"""Experiment conditions over one shared Hey Robot execution stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    instruction: str
    manipulate_call_limit: int | None = None

    def prompt(self, objective: str) -> str:
        return f"{self.instruction}\nGoal: {objective}"


_CONDITIONS = {
    "b0": ExperimentCondition(
        "b0",
        "Use exactly one manipulate call with the complete root goal and "
        "max_steps=300; do not decompose it.",
        manipulate_call_limit=1,
    ),
    "b1": ExperimentCondition(
        "b1",
        "Use normal Hey Robot hierarchical planning and re-observe at option boundaries.",
    ),
    "b2": ExperimentCondition(
        "b2",
        "Follow the frozen oracle pattern: inspect, run manipulate on the complete "
        "root goal, then re-observe and repeat until the environment terminates or "
        "the hard trial budget is exhausted. Do not declare success or failure from "
        "an inconclusive image caption alone.",
    ),
}


def condition_for(name: str) -> ExperimentCondition:
    try:
        return _CONDITIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown condition {name!r}; expected b0, b1 or b2") from exc
