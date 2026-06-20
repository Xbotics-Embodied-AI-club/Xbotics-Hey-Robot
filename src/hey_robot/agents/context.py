from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hey_robot.agents.task_run import TaskRun
from hey_robot.agents.types import RobotSnapshot
from hey_robot.capability.catalog.models import CapabilityManifest
from hey_robot.episode import EpisodeRecord
from hey_robot.protocol import UserTurn

if TYPE_CHECKING:
    from hey_robot.memory import MemoryRuntime


@dataclass(frozen=True)
class RobotAgentContext:
    task: str
    robot_state: str
    interaction_mode: str = "task"
    episode_context: str | None = None
    recovery_context: str | None = None
    pending_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def memory_context(self) -> str | None:
        scene_context = self.metadata.get("scene_context")
        capability_context = self.metadata.get("capability_context")
        parts = [
            part
            for part in (
                capability_context,
                self.episode_context,
                scene_context,
                self.pending_context,
            )
            if isinstance(part, str) and part
        ]
        return "\n\n".join(parts) if parts else None


class RobotContextBuilder:
    """Build the robot-specific context given to the robot agent.

    This is intentionally separate from the generic tool loop. It keeps robot
    task state, skill history, observations, and queued user corrections in a
    stable shape for the loop-first robot agent.
    """

    def __init__(
        self,
        *,
        max_history: int = 12,
        max_pending: int = 5,
        capability_manifest_provider: Callable[[], CapabilityManifest] | None = None,
    ) -> None:
        self.max_history = max(1, int(max_history))
        self.max_pending = max(1, int(max_pending))
        self.capability_manifest_provider = capability_manifest_provider

    def build(
        self,
        *,
        turn: UserTurn,
        snapshot: RobotSnapshot,
        history: list[EpisodeRecord],
        recovery_context: str | None = None,
        pending_turns: list[UserTurn] | None = None,
        task: TaskRun | None = None,
    ) -> RobotAgentContext:
        capability_manifest = self._capability_manifest()
        return RobotAgentContext(
            task=turn.text,
            robot_state=snapshot.summary(),
            interaction_mode=str(turn.metadata.get("interaction_mode") or "task"),
            episode_context=self._history_context(history),
            recovery_context=recovery_context,
            pending_context=self._pending_context(pending_turns or []),
            metadata={
                "episode_id": turn.envelope.episode_id,
                "robot_id": turn.envelope.robot_id,
                "agent_id": turn.envelope.agent_id,
                "interaction_mode": str(
                    turn.metadata.get("interaction_mode") or "task"
                ),
                "task": task.to_dict() if task is not None else None,
                "latest_observation": self._observation_metadata(snapshot),
                "scene_context": None,
                "capability_context": self._capability_context(capability_manifest),
            },
        )

    def _history_context(self, history: list[EpisodeRecord]) -> str | None:
        if not history:
            return None
        lines = ["Recent episode history:"]
        for record in history[-self.max_history :]:
            content = " ".join(str(record.content).split())
            lines.append(f"{record.role}: {content}")
        return "\n".join(lines)

    def _pending_context(self, pending_turns: list[UserTurn]) -> str | None:
        if not pending_turns:
            return None
        lines = [
            "Pending operator turns:",
            "- Treat these as user instructions or operator claims.",
            (
                "- Do not treat statements about robot pose, object location, or task outcome as "
                "confirmed physical facts until status, observation, or execution feedback verifies them."
            ),
        ]
        for turn in pending_turns[-self.max_pending :]:
            payload = {
                "text": turn.text,
                "trace_id": turn.envelope.trace_id,
                "message_id": turn.envelope.message_id,
                "source": "operator",
                "verification": "unverified",
            }
            lines.append(f"- {json.dumps(payload, ensure_ascii=False)}")
        return "\n".join(lines)

    @staticmethod
    def _observation_metadata(snapshot: RobotSnapshot) -> dict[str, Any] | None:
        observation = snapshot.observation
        if observation is None:
            return None
        return {
            "frame_id": observation.frame_id,
            "image_count": len(observation.images),
            "artifact_count": len(observation.artifacts),
            "task": observation.task,
        }

    def _capability_manifest(self) -> CapabilityManifest | None:
        if self.capability_manifest_provider is None:
            return None
        return self.capability_manifest_provider()

    @staticmethod
    def _capability_context(manifest: CapabilityManifest | None) -> str | None:
        if manifest is None:
            return None
        tools = (
            ", ".join(
                f"{tool.name}({tool.source},{tool.safety_level})"
                for tool in manifest.tools[:12]
            )
            or "none"
        )
        robot_skills = (
            ", ".join(skill.name for skill in manifest.robot_skill_actions[:16])
            or "none"
        )
        return "\n".join(
            [
                "Current capability summary:",
                f"- Tools: {tools}",
                f"- Robot skill actions: {robot_skills}",
            ]
        )


class RobotMemoryContextBuilder:
    """Build the memory/context block passed into the agent runtime.

    This keeps memory prompt composition outside ``RobotAgentCore``. The core
    supplies current turn inputs; this builder decides how robot capability
    context, task context, perception context, and long-term memory are ordered.
    """

    def __init__(
        self,
        *,
        memory: MemoryRuntime,
        robot_skill_catalog_context_provider: Callable[[], str],
    ) -> None:
        self.memory = memory
        self.robot_skill_catalog_context_provider = robot_skill_catalog_context_provider

    def build(
        self,
        *,
        task: str,
        task_context: str | None = None,
        perception_context: str | None = None,
    ) -> str | None:
        current_context = _join_contexts(task_context, perception_context)
        memory_context = self.memory.build_agent_context(task, current_context, limit=6)
        return _join_contexts(
            self.robot_skill_catalog_context_provider(), memory_context
        )


def _join_contexts(*parts: str | None) -> str | None:
    filtered = [part for part in parts if isinstance(part, str) and part]
    return "\n\n".join(filtered) if filtered else None
