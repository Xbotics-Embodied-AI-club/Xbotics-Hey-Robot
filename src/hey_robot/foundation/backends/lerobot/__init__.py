"""LeRobot policy backend with a single production executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.foundation.backends.lerobot.executor import LeRobotPolicyExecutor

__all__ = ["LeRobotPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hey_robot.foundation.backends.lerobot.executor import (
            LeRobotPolicyExecutor,
        )

        return LeRobotPolicyExecutor
    raise AttributeError(name)
