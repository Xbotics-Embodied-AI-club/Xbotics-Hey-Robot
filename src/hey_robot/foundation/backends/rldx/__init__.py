"""RLDX-1 policy backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.foundation.backends.rldx.executor import RLDXPolicyExecutor

__all__ = ["RLDXPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from hey_robot.foundation.backends.rldx.executor import RLDXPolicyExecutor

        return RLDXPolicyExecutor
    raise AttributeError(name)
