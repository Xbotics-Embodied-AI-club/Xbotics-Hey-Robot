from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hey_robot.config.model import DeploymentConfig, RobotSpec
from hey_robot.robots.primitive_inventory import supported_driver_primitives
from hey_robot.skills.base import SkillSpec
from hey_robot.skills.registry import SkillRegistry, registry_from_config


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


def validate_deployment(config: DeploymentConfig) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
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
    if not config.skills.enabled:
        issues.append(
            ValidationIssue(
                "error",
                "skills.enabled must explicitly list the deployment skill surface",
            )
        )
    try:
        registry = registry_from_config(config)
    except Exception as exc:
        issues.append(ValidationIssue("error", f"failed to load skill modules: {exc}"))
        return issues
    catalog = registry.catalog(enabled_only=False)
    for skill_name in config.skills.enabled:
        try:
            contract = catalog.get(skill_name)
        except KeyError:
            issues.append(
                ValidationIssue(
                    "error",
                    f"skills.enabled references unknown skill {skill_name}",
                )
            )
            continue
        if config.skills.mode == "production" and not contract.agent_visible:
            issues.append(
                ValidationIssue(
                    "error",
                    f"skills.enabled must list only semantic skills in production; "
                    f"{skill_name} is implementation-level",
                )
            )
        skill_robots = _skill_robots(config)
        unsupported = _unsupported_robot_families(contract, skill_robots.values())
        if unsupported:
            issues.append(
                ValidationIssue(
                    "error",
                    f"skill {skill_name} supports robots "
                    f"{','.join(contract.supported_robots)}, but deployment has "
                    f"{','.join(unsupported)}",
                )
            )
        if contract.external_capability and not _has_capability(
            config, contract.external_capability
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    f"skill {skill_name} requires unavailable capability "
                    f"{contract.external_capability}",
                )
            )
        for dependency in _skill_dependencies(contract, registry=registry):
            try:
                dependency_contract = catalog.get(dependency)
            except KeyError:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"skill {skill_name} references unknown dependency {dependency}",
                    )
                )
                continue
            if dependency_contract.external_capability and not _has_capability(
                config, dependency_contract.external_capability
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"skill {skill_name} requires unavailable capability "
                        f"{dependency_contract.external_capability}",
                    )
                )
        issues.extend(
            _driver_primitive_issues(
                skill_name,
                (contract, *tuple(_dependency_contracts(contract, registry=registry))),
                robots=skill_robots,
            )
        )
    return issues


def _has_capability(config: DeploymentConfig, name: str) -> bool:
    return any(
        service.enabled and name in service.skill_names
        for service in config.capability_services.values()
    )


def _skill_dependencies(
    contract: SkillSpec, *, registry: SkillRegistry
) -> tuple[str, ...]:
    return tuple(_iter_skill_dependencies(contract, seen=set(), registry=registry))


def _dependency_contracts(
    contract: SkillSpec, *, registry: SkillRegistry
) -> tuple[SkillSpec, ...]:
    catalog = registry.catalog(enabled_only=False)
    contracts: list[SkillSpec] = []
    for dependency in _skill_dependencies(contract, registry=registry):
        try:
            contracts.append(catalog.get(dependency))
        except KeyError:
            continue
    return tuple(contracts)


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
    contract: SkillSpec,
    robots: Iterable[RobotSpec],
) -> list[str]:
    if not contract.supported_robots:
        return []
    supported = set(contract.supported_robots)
    return sorted(
        {robot.robot_family for robot in robots if robot.robot_family not in supported}
    )


def _driver_primitive_issues(
    skill_name: str,
    contracts: tuple[SkillSpec, ...],
    *,
    robots: dict[str, RobotSpec],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for contract in contracts:
        if not contract.driver_primitives:
            continue
        for robot_id, robot in robots.items():
            if contract.supported_robots and (
                robot.robot_family not in contract.supported_robots
            ):
                continue
            supported = set(supported_driver_primitives(robot))
            missing = sorted(
                primitive
                for primitive in contract.driver_primitives
                if primitive not in supported
            )
            if not missing:
                continue
            issues.append(
                ValidationIssue(
                    "error",
                    f"skill {skill_name} requires driver primitives "
                    f"{','.join(missing)} via {contract.name}, but robot {robot_id} "
                    f"({robot.robot_family}/{robot.driver_kind}) does not support them",
                )
            )
    return issues


def _iter_skill_dependencies(
    contract: SkillSpec, *, seen: set[str], registry: SkillRegistry
) -> list[str]:
    dependencies: list[str] = []
    for dependency in contract.dependencies:
        dependency = str(dependency).strip()
        if not dependency or dependency in seen:
            continue
        seen.add(dependency)
        dependencies.append(dependency)
        try:
            child = registry.catalog(enabled_only=False).get(dependency)
        except KeyError:
            continue
        dependencies.extend(
            _iter_skill_dependencies(child, seen=seen, registry=registry)
        )
    return dependencies
