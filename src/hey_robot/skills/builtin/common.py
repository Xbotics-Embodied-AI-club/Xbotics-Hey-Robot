from __future__ import annotations

from typing import Any

from hey_robot.skills.base import SkillSpec


def schema(
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        payload["required"] = list(required)
    return payload


def spec(
    name: str,
    description: str,
    *,
    category: str,
    input_schema: dict[str, Any] | None = None,
    required_resources: tuple[str, ...] = (),
    preconditions: tuple[str, ...] = (),
    success_criteria: tuple[str, ...] = (),
    failure_modes: tuple[str, ...] = (),
    recovery_hints: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    driver_primitives: tuple[str, ...] = (),
    external_capability: str | None = None,
    supported_robots: tuple[str, ...] = ("xlerobot",),
    safety_level: str = "normal",
    timeout_sec: float = 10.0,
    interruptible: bool = True,
    agent_visible: bool = True,
    feedback_mode: str = "status",
    refresh_observation: bool = True,
    capability_type: str | None = None,
    goal_effects: tuple[str, ...] = (),
    evidence_outputs: tuple[str, ...] = (),
    cannot_satisfy: tuple[str, ...] = (),
) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=description,
        category=category,
        input_schema=input_schema or {},
        required_resources=required_resources,
        preconditions=preconditions,
        success_criteria=success_criteria,
        failure_modes=failure_modes,
        recovery_hints=recovery_hints,
        dependencies=dependencies,
        driver_primitives=driver_primitives,
        external_capability=external_capability,
        supported_robots=supported_robots,
        safety_level=safety_level,
        timeout_sec=timeout_sec,
        interruptible=interruptible,
        agent_visible=agent_visible,
        feedback_mode=feedback_mode,
        refresh_observation=refresh_observation,
        capability_type=capability_type,
        goal_effects=goal_effects,
        evidence_outputs=evidence_outputs,
        cannot_satisfy=cannot_satisfy,
    )


async def invoke(ctx: Any, name: str, arguments: dict[str, Any] | None = None) -> Any:
    if ctx.invoke is None:
        raise RuntimeError("skill context invoke is not available")
    return await ctx.invoke(name, dict(arguments or {}))
