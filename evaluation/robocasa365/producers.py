"""Root-task producers used by the common full-system benchmark path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Producer:
    name: str

    def prompt(self, objective: str) -> str:
        return objective


class B0RootTaskProducer(Producer):
    def __init__(self) -> None:
        super().__init__("b0")


class B1AgentProducer(Producer):
    def __init__(self) -> None:
        super().__init__("b1")


class FrozenOraclePlanner(Producer):
    """Deterministic language producer; execution still goes through the Agent."""

    def __init__(self) -> None:
        super().__init__("b2")

    def prompt(self, objective: str) -> str:
        return (
            "Follow this frozen oracle plan through the normal Hey Robot tools. "
            "Do not call the simulator directly. Re-observe after each option. "
            f"Goal: {objective}"
        )


def producer_for(name: str) -> Producer:
    producers = {
        "b0": B0RootTaskProducer,
        "b1": B1AgentProducer,
        "b2": FrozenOraclePlanner,
    }
    try:
        return producers[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unknown producer {name!r}; expected b0, b1 or b2") from exc
