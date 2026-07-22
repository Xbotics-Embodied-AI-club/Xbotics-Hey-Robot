"""Robot Runtime public API with lazy imports for isolated backends."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RobotCapabilities": "hey_robot.robot_runtime.base",
    "RobotDriver": "hey_robot.robot_runtime.base",
    "RobotDriverContext": "hey_robot.robot_runtime.base",
    "RobotHealth": "hey_robot.robot_runtime.base",
    "ActionBufferEntry": "hey_robot.robot_runtime.control_plane",
    "ControlPlaneDecision": "hey_robot.robot_runtime.control_plane",
    "RobotControlPlane": "hey_robot.robot_runtime.control_plane",
    "DEFAULT_EMBODIMENT_PROFILES": "hey_robot.robot_runtime.embodiments",
    "EmbodimentProfile": "hey_robot.robot_runtime.embodiments",
    "get_embodiment_profile": "hey_robot.robot_runtime.embodiments",
    "resolve_embodiment_profile_name": "hey_robot.robot_runtime.embodiments",
    "LeKiwiDriver": "hey_robot.robot_runtime.lekiwi",
    "RobotManager": "hey_robot.robot_runtime.manager",
    "MockRobotDriver": "hey_robot.robot_runtime.mock",
    "RoboCasaRemoteDriver": "hey_robot.robot_runtime.robocasa_remote",
    "RobotRuntime": "hey_robot.robot_runtime.runtime",
    "RobotRuntimeSnapshot": "hey_robot.robot_runtime.runtime",
    "RobotSafetyError": "hey_robot.robot_runtime.safety",
    "RobotSafetySupervisor": "hey_robot.robot_runtime.safety",
    "SafetyDecision": "hey_robot.robot_runtime.safety",
    "RobotService": "hey_robot.robot_runtime.service",
    "SO101Driver": "hey_robot.robot_runtime.so101",
    "XLeRobotDriver": "hey_robot.robot_runtime.xlerobot",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)
