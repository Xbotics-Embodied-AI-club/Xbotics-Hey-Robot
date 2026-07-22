"""Remote RoboCasa runtime adapter with lazy public imports."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GrpcRoboCasaRuntimeClient": "hey_robot.robot_runtime.robocasa_remote.client",
    "RoboCasaRemoteDriver": "hey_robot.robot_runtime.robocasa_remote.driver",
    "RemoteEpisodeClient": "hey_robot.robot_runtime.robocasa_remote.protocol",
    "RemoteObservation": "hey_robot.robot_runtime.robocasa_remote.protocol",
    "RemoteStep": "hey_robot.robot_runtime.robocasa_remote.protocol",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)
