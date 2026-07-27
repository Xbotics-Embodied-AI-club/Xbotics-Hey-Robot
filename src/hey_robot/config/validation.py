from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hey_robot.config.model import DeploymentConfig, RobotSpec
from hey_robot.config.robot_inventory import supported_driver_primitives
from hey_robot.skills import Skill, registry_from_config


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


def validate_deployment(config: DeploymentConfig) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    enabled_agents = [
        agent_id for agent_id, agent in config.agents.items() if agent.enabled
    ]
    if len(enabled_agents) > 1:
        issues.append(
            ValidationIssue(
                "error",
                "a deployment supports exactly one enabled autonomous agent; got "
                + ",".join(enabled_agents),
            )
        )
    for agent_id, agent in config.agents.items():
        if agent.robot_id and agent.robot_id not in config.robots:
            issues.append(
                ValidationIssue(
                    "error",
                    f"agent {agent_id} references missing robot {agent.robot_id}",
                )
            )
        if agent.policy_id and agent.policy_id not in config.policies:
            issues.append(
                ValidationIssue(
                    "error",
                    f"agent {agent_id} references missing policy {agent.policy_id}",
                )
            )
    for policy_id, policy in config.policies.items():
        if policy.robot_id not in config.robots:
            issues.append(
                ValidationIssue(
                    "error",
                    f"policy {policy_id} references missing robot {policy.robot_id}",
                )
            )
    issues.extend(_robot_policy_configuration_issues(config))
    issues.extend(_vln_configuration_issues(config))
    issues.extend(_robocasa_configuration_issues(config))
    for path in (
        config.resources.runtime_dir,
        config.resources.media_root,
        config.resources.episodes_root,
    ):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            issues.append(
                ValidationIssue("error", f"cannot create resource path {path}: {exc}")
            )
    if config.skills.mode not in {"production", "bringup"}:
        issues.append(
            ValidationIssue(
                "error",
                f"skills.mode must be 'production' or 'bringup', got {config.skills.mode!r}",
            )
        )
    if config.skills.execution_mode != "local":
        issues.append(
            ValidationIssue(
                "error",
                "skills.execution_mode 只支持 'local'；legacy Skill OS 已移除",
            )
        )
    if not config.skills.modules or any(
        not str(module).startswith("hey_robot.skills.") or ".legacy" in str(module)
        for module in config.skills.modules
    ):
        issues.append(
            ValidationIssue(
                "error",
                "skills.modules 必须使用 native hey_robot.skills.* modules",
            )
        )
    tool_names = config.skills.tool_names
    unknown_implementations = sorted(
        set(config.skills.implementations) - set(tool_names)
    )
    issues.extend(
        ValidationIssue(
            "error",
            f"skills.implementations references non-surface skill {skill_name}",
        )
        for skill_name in unknown_implementations
    )
    if not tool_names:
        issues.append(
            ValidationIssue(
                "error",
                "skills.tools must explicitly list the deployment skill surface",
            )
        )
    try:
        registry = registry_from_config(config)
    except Exception as exc:
        issues.append(ValidationIssue("error", f"failed to load skill modules: {exc}"))
        return issues
    known_tools: list[str] = []
    for skill_name in tool_names:
        try:
            registry.get(skill_name)
        except KeyError:
            issues.append(
                ValidationIssue(
                    "error",
                    f"skills.tools references unknown skill {skill_name}",
                )
            )
            continue
        known_tools.append(skill_name)
    deployment_skills = registry.select(known_tools)
    skill_robots = _skill_robots(config)
    for skill in deployment_skills:
        skill_name = skill.name
        unsupported = _unsupported_robot_families(skill, skill_robots.values())
        if unsupported:
            issues.append(
                ValidationIssue(
                    "error",
                    f"skill {skill_name} supports robots "
                    f"{','.join(skill.supported_robots)}, but deployment has "
                    f"{','.join(unsupported)}",
                )
            )
        issues.extend(
            ValidationIssue(
                "error",
                f"skill {skill_name} requires unavailable model service "
                f"{required_model}",
            )
            for required_model in skill.required_models
            if not _has_model_service_for_skill(config, required_model)
        )
        issues.extend(
            _driver_primitive_issues(
                skill_name,
                (skill,),
                robots=skill_robots,
            )
        )
    return issues


def _robot_policy_configuration_issues(
    config: DeploymentConfig,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for service_id, service in config.model_services.items():
        if service.type != "robot_policy":
            continue
        runtime = str(service.settings.get("runtime") or "")
        if runtime != "lerobot":
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} has unsupported robot policy "
                    f"runtime {runtime!r}",
                )
            )
        issues.extend(
            ValidationIssue(
                "error",
                f"model service {service_id} requires setting {required}",
            )
            for required in (
                "policy_path",
                "policy_device",
                "action_space",
                "action_dimensions",
            )
            if required not in service.settings
        )
        try:
            action_dimensions = int(service.settings.get("action_dimensions") or 0)
        except (TypeError, ValueError):
            action_dimensions = 0
        if "action_dimensions" in service.settings and action_dimensions <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} requires positive action_dimensions",
                )
            )
    return issues


def _vln_configuration_issues(config: DeploymentConfig) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for service_id, service in config.model_services.items():
        if service.type != "vln_planner":
            continue
        backend = str(service.settings.get("backend") or "internvla_n1_dualvln")
        if backend != "internvla_n1_dualvln":
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} has unsupported VLN backend "
                    f"{backend!r}",
                )
            )
        control_mode = str(service.settings.get("control_mode") or "base_action_chunk")
        if control_mode != "base_action_chunk":
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} has unsupported VLN control_mode "
                    f"{control_mode!r}",
                )
            )
        limits = (
            ("base_linear_speed", 0.0, 0.25),
            ("base_angular_speed", 0.0, 0.60),
            ("max_action_chunk_steps", 0.0, 8.0),
            ("system1_replans_per_waypoint", 0.0, 16.0),
            ("discrete_forward_cm", 0.0, 25.0),
            ("discrete_turn_deg", 0.0, 30.0),
        )
        for name, lower, upper in limits:
            value = service.settings.get(name)
            try:
                number = (
                    float(value)
                    if isinstance(value, str | int | float)
                    else float("nan")
                )
            except (TypeError, ValueError):
                number = float("nan")
            if not lower < number <= upper:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"model service {service_id} requires {name} in "
                        f"({lower}, {upper}] for base_action_chunk",
                    )
                )
        try:
            forward_duration_ms = (
                10.0
                * float(service.settings["discrete_forward_cm"])
                / float(service.settings["base_linear_speed"])
            )
            turn_duration_ms = (
                1000.0
                * math.radians(float(service.settings["discrete_turn_deg"]))
                / float(service.settings["base_angular_speed"])
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        else:
            if max(forward_duration_ms, turn_duration_ms) > 1000.0:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"model service {service_id} native VLN action exceeds "
                        "the 1000ms base velocity safety window",
                    )
                )
        if bool(service.settings.get("mock_mode", False)):
            continue
        issues.extend(
            ValidationIssue(
                "error", f"model service {service_id} requires setting {required}"
            )
            for required in ("model_path", "internnav_repo", "media_root")
            if not str(service.settings.get(required) or "").strip()
        )
    return issues


def _robocasa_configuration_issues(
    config: DeploymentConfig,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for service_id, service in config.model_services.items():
        if not (
            service.type == "robot_policy"
            and str(service.settings.get("runtime") or "") == "lerobot"
            and str(service.settings.get("embodiment") or "") == "robocasa"
        ):
            continue
        if tuple(service.provides) != ("manipulate",):
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} must provide only manipulate",
                )
            )
        robot = config.robots.get(service.robot_id)
        if robot is None or robot.robot_family != "robocasa":
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} requires a RoboCasa robot",
                )
            )
        prompt_mode = str(service.settings.get("prompt_mode") or "")
        if prompt_mode not in {"environment_root", "agent_subgoal"}:
            issues.append(
                ValidationIssue(
                    "error",
                    f"model service {service_id} has invalid prompt_mode {prompt_mode!r}",
                )
            )
    for robot_id, robot in config.robots.items():
        if robot.robot_family != "robocasa" or not bool(
            robot.settings.get("managed_backend", False)
        ):
            continue
        matching = [
            service
            for service in config.model_services.values()
            if service.enabled
            and service.robot_id == robot_id
            and service.type == "robot_policy"
            and str(service.settings.get("runtime") or "") == "lerobot"
            and str(service.settings.get("embodiment") or "") == "robocasa"
        ]
        if len(matching) != 1:
            issues.append(
                ValidationIssue(
                    "error",
                    f"managed RoboCasa robot {robot_id} requires exactly one "
                    "LeRobot robot_policy service",
                )
            )
        elif not str(matching[0].settings.get("media_root") or "").strip():
            issues.append(
                ValidationIssue(
                    "error",
                    f"managed RoboCasa robot {robot_id} requires model setting "
                    "media_root",
                )
            )
    return issues


def _has_model_service_for_skill(config: DeploymentConfig, name: str) -> bool:
    return any(
        service.enabled and name in service.provides
        for service in config.model_services.values()
    )


def _skill_robots(config: DeploymentConfig) -> dict[str, RobotSpec]:
    robot_ids = {
        policy.robot_id
        for policy in config.policies.values()
        if policy.enabled and policy.robot_id in config.robots
    }
    if not robot_ids:
        robot_ids = {
            robot_id for robot_id, robot in config.robots.items() if robot.enabled
        }
    return {
        robot_id: config.robots[robot_id]
        for robot_id in sorted(robot_ids)
        if config.robots[robot_id].enabled
    }


def _unsupported_robot_families(
    skill: Skill,
    robots: Iterable[RobotSpec],
) -> list[str]:
    if not skill.supported_robots:
        return []
    supported = set(skill.supported_robots)
    return sorted(
        {robot.robot_family for robot in robots if robot.robot_family not in supported}
    )


def _driver_primitive_issues(
    skill_name: str,
    skills: tuple[Skill, ...],
    *,
    robots: dict[str, RobotSpec],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for skill in skills:
        if not skill.required_actions:
            continue
        for robot_id, robot in robots.items():
            if skill.supported_robots and (
                robot.robot_family not in skill.supported_robots
            ):
                continue
            supported = set(supported_driver_primitives(robot))
            missing = sorted(
                primitive
                for primitive in skill.required_actions
                if primitive not in supported
            )
            if not missing:
                continue
            issues.append(
                ValidationIssue(
                    "error",
                    f"skill {skill_name} requires driver primitives "
                    f"{','.join(missing)} via {skill.name}, but robot {robot_id} "
                    f"({robot.robot_family}/{robot.driver_kind}) does not support them",
                )
            )
    return issues
