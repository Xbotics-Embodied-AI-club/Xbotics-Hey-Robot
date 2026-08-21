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
        "Ground the task in the current observation before selecting a policy "
        "call. Reserve manipulate for contact-sensitive control; use ordinary "
        "robot skills for navigation, open-space transport, release, and pose "
        "adjustment.\n"
        "Treat each uninterrupted manipulate sequence as one policy episode. "
        "If its action budget expires, continue with the same task text before "
        "introducing another physical command. Reposition only after several "
        "attempts show no target interaction. Inspect structured progress after "
        "every action and allow only the environment to declare completion. "
        "A scene caption is optional planning context, not a permission gate for "
        "a vision-action policy: after one unavailable scene caption, do not ask "
        "the same visual question again. Use structured progress and proceed with "
        "the bounded policy call using the official Goal text.",
    ),
}


def condition_for(name: str) -> ExperimentCondition:
    try:
        return _CONDITIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown condition {name!r}; expected b0 or b1") from exc
