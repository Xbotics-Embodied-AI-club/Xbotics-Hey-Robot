"""Dependency-injection container for robot tools.

Replaces the pattern where tools access ``self.io`` / memory stores / etc.
through ``RobotAgentCore`` being ``self``.  Every tool receives a
:class:`ToolContext` at construction time via its ``create(ctx)`` factory.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hey_robot.agents.runtime.runner import AgentRuntime
    from hey_robot.agents.runtime.state import AgentState
    from hey_robot.autonomy import AutonomyManager
    from hey_robot.memory.runtime import MemoryRuntime
    from hey_robot.skills import RobotSkillCatalog
    from hey_robot.skills.base import SkillCatalog
    from hey_robot.spec import AgentSpec
    from hey_robot.types import AgentIO


@dataclass
class ToolContext:
    """Typed bag of runtime dependencies that tools need.

    Populated by ``RobotAgentCore.__init__`` and refreshed per-turn via
    :meth:`refresh_turn`.
    """

    # Long-lived resources.
    io: AgentIO
    spec: AgentSpec
    memory: MemoryRuntime
    autonomy: AutonomyManager
    skill_catalog: RobotSkillCatalog | None

    # Mutable runtime state.
    runtime_state: AgentState
    pending_skills: dict[str, asyncio.Future] = field(default_factory=dict)
    skill_catalog_runtime: SkillCatalog | None = None

    # Optional per-turn values.
    robot_type: str | None = None
    turn_context: ToolTurnContext | None = None
    runtime: AgentRuntime | None = None
    skill_gateway: Any = None

    # Callable helpers set by RobotAgentCore.
    _current_envelope: Any = None  # Callable[[], Envelope]
    _configured_robot_type: Any = None  # Callable[[], str | None]
    _get_task: Any = None  # Callable[[], str]
    _get_robot_status: Any = None  # Callable[[], str]
    task_runtime: Any = None

    def __post_init__(self) -> None:
        if self.skill_gateway is not None or self.skill_catalog_runtime is None:
            return
        if self.io is None or self.spec is None or self.runtime_state is None:
            return
        if self._current_envelope is None or self._get_task is None:
            return
        from hey_robot.agents.skill_gateway import SkillGateway

        self.skill_gateway = SkillGateway(
            io=self.io,
            spec=self.spec,
            skill_catalog=self.skill_catalog_runtime,
            runtime_state=self.runtime_state,
            pending_skills=self.pending_skills,
            current_envelope=lambda: self._current_envelope(),
            get_task=lambda: self._get_task(),
            recovery_required=lambda: bool(
                self.turn_context and self.turn_context.recovery_required
            ),
            task_runtime=lambda: self.task_runtime,
        )


@dataclass
class ToolTurnContext:
    """Per-turn snapshot refreshed at the start of ``handle_turn()``."""

    snapshot_summary: str = ""
    observation_summary: str = ""
    snapshot: Any = None  # RobotSnapshot
    envelope: Any = None  # Envelope
    recovery_required: bool = False
