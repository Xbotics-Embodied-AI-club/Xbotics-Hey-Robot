"""Xiaomi-Robotics-1 policy backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.foundation.backends.xiaomi.executor import XiaomiPolicyExecutor

__all__ = ["XiaomiPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hey_robot.foundation.backends.xiaomi.executor import XiaomiPolicyExecutor

        return XiaomiPolicyExecutor
    raise AttributeError(name)
