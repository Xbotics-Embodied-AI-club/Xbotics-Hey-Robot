"""LeRobot policy backend with a single production executor."""

from typing import Any

__all__ = ["LeRobotPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hey_robot.foundation.backends.lerobot.executor import (
            LeRobotPolicyExecutor,
        )

        return LeRobotPolicyExecutor
    raise AttributeError(name)
