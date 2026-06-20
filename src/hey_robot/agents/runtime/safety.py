from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from hey_robot.agents.runtime.grounding import is_perception_evidence_record
from hey_robot.agents.runtime.hooks import ToolHookContext
from hey_robot.skills.registry import load_skill_registry


class RobotSafetyHook:
    """Pre-tool safety gate for robot actions.

    It is intentionally small and deterministic. The hook blocks non-read-only
    tools when the latest status reports an emergency stop or safety stop.
    """

    def __init__(
        self,
        status_provider: Callable[[], Mapping[str, Any] | None],
    ) -> None:
        self.status_provider = status_provider

    async def before_tool(self, context: ToolHookContext) -> None:
        if context.tool.read_only:
            return
        status = dict(self.status_provider() or {})
        if context.tool.name == "request_capability":
            self._check_status_snapshot(context, status)
            active_flags = [
                key
                for key in (
                    "emergency_stop",
                    "estop",
                    "safety_stop",
                    "collision_detected",
                    "protective_stop",
                )
                if bool(status.get(key))
            ]
            if active_flags:
                raise RuntimeError(
                    f"robot safety gate blocked tool {context.tool.name}: active flags={','.join(active_flags)}"
                )
            self._check_skill_intent(context.arguments)
            self._check_consecutive_motion_without_perception(context.arguments, status)

    async def after_tool(self, _context: ToolHookContext, _result: str) -> None:
        return None

    async def on_tool_error(self, _context: ToolHookContext, _error: Exception) -> None:
        return None

    def _check_status_snapshot(
        self,
        context: ToolHookContext,
        status: Mapping[str, Any],
    ) -> None:
        if not status:
            raise RuntimeError(
                f"robot safety gate blocked tool {context.tool.name}: missing robot status snapshot"
            )

        frame_id = status.get("frame_id")
        if frame_id is None:
            raise RuntimeError(
                f"robot safety gate blocked tool {context.tool.name}: missing status frame_id"
            )

        if int(frame_id) < 0:
            raise RuntimeError(
                f"robot safety gate blocked tool {context.tool.name}: invalid status frame_id={frame_id}"
            )

    def _check_skill_intent(self, arguments: Mapping[str, Any]) -> None:
        objective = str(arguments.get("objective") or "").strip()
        if not objective:
            raise RuntimeError("robot safety gate blocked empty skill objective")
        if len(objective) > 160:
            raise RuntimeError("robot safety gate blocked overlong skill objective")
        lowered = objective.lower()
        blocked_tokens = (";", "\n", " and then ", " then ", " afterwards ")
        if any(token in lowered for token in blocked_tokens):
            raise RuntimeError(
                "robot safety gate blocked compound skill objective; issue one atomic skill"
            )

    def _check_consecutive_motion_without_perception(
        self,
        arguments: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> None:
        capability = str(arguments.get("capability") or "").strip()
        if not capability:
            return
        try:
            spec = load_skill_registry().catalog(enabled_only=False).get(capability)
        except KeyError:
            return
        if spec.safety_level != "motion":
            return
        recent = status.get("recent_tool_calls")
        if not isinstance(recent, list) or not recent:
            return
        seen_motion = False
        for item in reversed(recent):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            args = item.get("arguments")
            success = bool(item.get("success"))
            if not isinstance(args, dict):
                args = {}
            if is_perception_evidence_record(name, args, success=success):
                return
            if (
                success
                and name == "request_capability"
                and str(args.get("capability") or "").strip() == capability
            ):
                seen_motion = True
                break
        if seen_motion:
            raise RuntimeError(
                "robot safety gate blocked tool request_capability: "
                f"consecutive {capability} requires fresh perception evidence"
            )
