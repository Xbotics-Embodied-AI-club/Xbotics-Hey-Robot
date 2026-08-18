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
        "using the complete root Goal text verbatim as task_prompt and max_steps=600. "
        "Do not decompose it or call any other tool.",
        manipulate_call_limit=1,
    ),
    "b1": ExperimentCondition(
        "b1",
        "Use sparse recovery rather than eagerly decomposing a goal that the VLA may "
        "already know end to end. First call manipulate once with the complete root Goal "
        "as its exact task_prompt and max_steps=600. If the live environment has not "
        "completed after that attempt, inspect the current scene and select only the "
        "unfinished semantic remainder. Express recovery in the same short imperative, "
        "single-state-transition style as an Atomic-Seen instruction (for example open, "
        "close, turn on, or place), rather than as a free-form motion plan. For a recovery "
        "call, use a short natural English task_prompt that commands exactly one physical "
        "outcome. Preserve the minimum root-task context needed to disambiguate why and "
        "where to act, and mention an already-achieved state only when it grounds the "
        "remaining action. Prefer forms such as 'With the mug already under the dispenser, "
        "press the coffee machine start button to finish preparing coffee.' Do not copy the "
        "full multi-step root Goal into a recovery prompt, list multiple pending stages, or "
        "add labels such as Root goal / Current state / Current subgoal. Preserve concrete "
        "object identity and source/target spatial relations while leaving motion details "
        "to the VLA; normally use 300-450 steps. Do not create separate physical stages "
        "merely to park the arm, expose controls, improve the camera view, or satisfy "
        "uncertain visual verification. Judge recovery completion from its primary "
        "current-state outcome and do not require the final image to re-prove historical "
        "preconditions. Verification unknown is not automatically failure, but never issue "
        "an identical recovery task_prompt more than once: inspect again and choose a "
        "different remaining semantic outcome, or stop when the task-level hard budget is "
        "exhausted. Continue until the live environment reports completion.",
    ),
    "b3": ExperimentCondition(
        "b3",
        "Use sparse recovery rather than eagerly decomposing a goal that the VLA may "
        "already know end to end. First call manipulate once with the complete root Goal "
        "as its exact task_prompt and max_steps=400. If the live environment has not "
        "completed after that attempt, inspect the current scene and select only the "
        "unfinished semantic remainder. Express recovery in the same short imperative, "
        "single-state-transition style as an Atomic-Seen instruction (for example open, "
        "close, turn on, or place), rather than as a free-form motion plan. For a recovery "
        "call, use a short natural English task_prompt that commands exactly one physical "
        "outcome. Preserve the minimum root-task context needed to disambiguate why and "
        "where to act, and mention an already-achieved state only when it grounds the "
        "remaining action. Do not copy the full multi-step root Goal into a recovery "
        "prompt, list multiple pending stages, or add motion-level instructions. Preserve "
        "concrete object identity and source/target spatial relations while leaving motion "
        "details to the VLA; normally use 300-400 steps. Inspect again after each bounded "
        "attempt and choose the next single unfinished physical outcome. Never issue an "
        "identical recovery task_prompt more than once, and continue until the live "
        "environment reports completion or the task-level hard budget is exhausted.",
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
        raise ValueError(
            f"unknown condition {name!r}; expected b0, b1, b2 or b3"
        ) from exc
