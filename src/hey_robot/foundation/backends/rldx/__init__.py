"""RLDX-1 policy backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.foundation.backends.rldx.executor import RLDXPolicyExecutor
    from hey_robot.foundation.backends.rldx.option_policy import RLDXChunkPolicy

__all__ = ["RLDXChunkPolicy", "RLDXPolicyExecutor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name == "RLDXChunkPolicy":
            from hey_robot.foundation.backends.rldx.option_policy import RLDXChunkPolicy

            return RLDXChunkPolicy
        from hey_robot.foundation.backends.rldx.executor import RLDXPolicyExecutor

        return RLDXPolicyExecutor
    raise AttributeError(name)
