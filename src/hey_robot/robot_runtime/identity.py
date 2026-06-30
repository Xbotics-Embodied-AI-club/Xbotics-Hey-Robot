from __future__ import annotations

from hey_robot.config import DeploymentConfig


def resolve_robot_family(
    config: DeploymentConfig | None,
    robot_id: str | None,
    *,
    fallback: str | None = None,
) -> str | None:
    if config is None or not robot_id:
        return fallback or robot_id
    spec = config.robots.get(robot_id)
    if spec is None:
        return fallback or robot_id
    return spec.robot_family
