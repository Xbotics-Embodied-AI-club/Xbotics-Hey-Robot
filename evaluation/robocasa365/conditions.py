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
        "Submit the complete root objective as one bounded physical option without "
        "decomposing it.",
        manipulate_call_limit=1,
    ),
    "b1": ExperimentCondition(
        "b1",
        "Use the root goal and the latest scene observation to select exactly one "
        "currently achievable and observable subgoal. Treat execution completion and "
        "world-state evidence as separate facts. Advance only when the latest evidence "
        "covers the complete subgoal, preserve the subgoal identity during a bounded "
        "retry, and otherwise choose a recovery subgoal or report blocked progress. "
        "Do not combine multiple unfinished stages or repeat a stage already supported "
        "by evidence. Continue until the live "
        "environment reports completion or the hard trial budget is exhausted.",
    ),
    "b2": ExperimentCondition(
        "b2",
        "Follow the frozen single-option baseline: use the complete root objective as "
        "the option, refresh evidence after each bounded attempt, and continue until "
        "the environment terminates or the hard trial budget is exhausted. Do not "
        "declare success or failure from inconclusive evidence alone.",
    ),
}


def condition_for(name: str) -> ExperimentCondition:
    try:
        return _CONDITIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown condition {name!r}; expected b0, b1 or b2") from exc
