"""Embodiment-independent coherence for stateful model sessions."""

from __future__ import annotations


class ModelSessionEpoch:
    """Marks a fresh model observation/history seed after physical intervention."""

    def __init__(self) -> None:
        self._value = 0
        self._reseed_required = False

    @property
    def value(self) -> int:
        return self._value

    def mark_physical_intervention(self) -> None:
        self._value += 1
        self._reseed_required = True

    def requires_reseed(self, *, requested: bool = False) -> bool:
        """Return whether the next model rollout needs a fresh context.

        This query is deliberately non-mutating.  A transport failure must not
        silently consume the reset that protects the next successful rollout.
        """
        return requested or self._reseed_required

    def acknowledge_rollout(self) -> None:
        """Record that a model rollout accepted the current physical history."""
        self._reseed_required = False
