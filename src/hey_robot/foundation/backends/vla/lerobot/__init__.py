"""LeRobot VLA backends with lazy imports for lightweight sidecars."""

from typing import Any

__all__ = ["LeRobotVLAExecutor", "LeRobotVLAPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hey_robot.foundation.backends.vla.lerobot.executor import (
            LeRobotVLAExecutor,
            LeRobotVLAPolicyExecutor,
        )

        return {
            "LeRobotVLAExecutor": LeRobotVLAExecutor,
            "LeRobotVLAPolicyExecutor": LeRobotVLAPolicyExecutor,
        }[name]
    raise AttributeError(name)
