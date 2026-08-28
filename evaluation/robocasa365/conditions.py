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
        "Do not inspect or rewrite the task. Immediately call manipulate exactly once, "
        "using the complete root Goal text verbatim as task_prompt and max_steps=560. "
        "Do not decompose it or call any other tool.",
        manipulate_call_limit=1,
    ),
    "b1": ExperimentCondition(
        "b1",
        "Emit exactly one tool call per turn. Wait for its result before selecting "
        "the next action; never return parallel or batched tool calls.\n"
        "Use manipulate for every physical action. For each call, copy the complete "
        "Goal text byte-for-byte as task_prompt and use max_steps=560. If its action "
        "budget is exhausted, call manipulate again with that exact same root Goal. "
        "Do not rewrite, shorten, or decompose the Goal. The environment alone "
        "declares final success.",
    ),
}


def condition_for(name: str) -> ExperimentCondition:
    try:
        return _CONDITIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown condition {name!r}; expected b0 or b1") from exc
