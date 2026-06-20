from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hey_robot.agents.runtime.grounding import needs_perception_grounding
from hey_robot.agents.scene_runtime import assess_observation_freshness
from hey_robot.agents.types import AgentTurnInput

if TYPE_CHECKING:
    from hey_robot.agents.core import RobotAgentCore
    from hey_robot.config import AgentSpec


@dataclass(frozen=True)
class AgentTurnPolicy:
    interaction_mode: str
    allowed_tools: set[str] | None
    scene_freshness: dict[str, Any]
    requires_active_perception: bool


class RobotTurnPolicy:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def build(self, payload: AgentTurnInput) -> AgentTurnPolicy:
        interaction_mode = self.interaction_mode(payload.turn)
        assessment = dict(payload.turn.metadata.get("_scene_freshness") or {})
        requires_active_perception = False
        if not payload.block_actuation:
            if assessment:
                requires_active_perception = bool(
                    assessment.get("status") != "not_required"
                )
            else:
                requires_active_perception = self.turn_needs_fresh_perception(
                    payload.turn.text,
                    interaction_mode=interaction_mode,
                )
                if requires_active_perception:
                    max_age_sec = float(
                        self.spec.settings.get("active_perception_max_age_sec", 15.0)
                    )
                    assessment = assess_observation_freshness(
                        payload.snapshot.observation,
                        robot_id=payload.snapshot.robot_id,
                        max_age_sec=max_age_sec,
                    ).to_dict()
        return AgentTurnPolicy(
            interaction_mode=interaction_mode,
            allowed_tools=(
                self.recovery_allowed_tools()
                if payload.block_actuation
                else self.allowed_tools(
                    payload.turn.text, interaction_mode=interaction_mode
                )
            ),
            scene_freshness=assessment,
            requires_active_perception=requires_active_perception,
        )

    async def collect_perception_context(
        self,
        *,
        core: RobotAgentCore,
        payload: AgentTurnInput,
        policy: AgentTurnPolicy,
    ) -> str | None:
        if payload.block_actuation or not policy.requires_active_perception:
            return None
        assessment = policy.scene_freshness
        if not bool(assessment.get("needs_refresh")):
            result = json.dumps(
                {
                    "tool": "request_perception",
                    "evidence_status": "ok",
                    "freshness": "current",
                    "evidence": {
                        "status": "ok",
                        "frame_id": assessment.get("frame_id"),
                        "image_count": assessment.get("image_count", 0),
                        "summary": "Fresh visual observation is already available in the current robot snapshot.",
                    },
                    "result": "Fresh visual observation is already available.",
                },
                ensure_ascii=False,
            )
            core.runtime.state.add_tool_call(
                "request_perception",
                {
                    "question": payload.turn.text,
                    "freshness": "current",
                    "source": "active_perception",
                },
                result,
                success=True,
            )
            return (
                "Active perception gate reused fresh current observation before deciding "
                f"(frame={assessment.get('frame_id')}, images={assessment.get('image_count', 0)}):\n"
                f"{result}"
            )
        question = payload.turn.text
        try:
            result = await core.request_perception(question=question, freshness="fresh")
        except Exception as exc:
            result = json.dumps(
                {
                    "tool": "request_perception",
                    "evidence_status": "degraded",
                    "result": f"Active perception failed: {type(exc).__name__}: {exc}",
                    "evidence": {"status": "caption_failed", "summary": str(exc)},
                },
                ensure_ascii=False,
            )
        core.runtime.state.add_tool_call(
            "request_perception",
            {"question": question, "freshness": "fresh", "source": "active_perception"},
            result,
            success='"evidence_status": "ok"' in result,
        )
        return (
            "Active perception evidence collected before deciding because current visual evidence was "
            f"{assessment.get('status') or 'not fresh'} ({assessment.get('reason') or 'unknown reason'}):\n"
            f"{result}"
        )

    @staticmethod
    def interaction_mode(turn) -> str:
        return (
            str((getattr(turn, "metadata", {}) or {}).get("interaction_mode") or "task")
            .strip()
            .lower()
        )

    @staticmethod
    def is_status_query(text: str) -> bool:
        normalized = " ".join(str(text or "").lower().split())
        markers = (
            "battery",
            "power",
            "status",
            "state",
            "progress",
            "电池",
            "电量",
            "状态",
            "现在怎么样",
            "你在干什么",
            "机械臂",
            "夹爪",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def is_action_request(text: str) -> bool:
        normalized = " ".join(str(text or "").lower().split())
        markers = (
            "open",
            "close",
            "move",
            "home",
            "turn",
            "stop",
            "inspect",
            "look",
            "pick",
            "place",
            "grasp",
            "forward",
            "backward",
            "left",
            "right",
            "打开",
            "关闭",
            "移动",
            "回到",
            "回去",
            "恢复",
            "复位",
            "重置",
            "调整",
            "姿态",
            "关节",
            "转",
            "向左",
            "向右",
            "停止",
            "看看",
            "观察",
            "跟随",
            "跟着",
            "拿",
            "抓",
            "放",
            "前进",
            "后退",
            "左转",
            "右转",
            "夹爪",
        )
        return any(marker in normalized for marker in markers)

    def turn_needs_fresh_perception(self, text: str, *, interaction_mode: str) -> bool:
        if interaction_mode == "chat":
            return needs_perception_grounding(text)
        if needs_perception_grounding(text):
            return True
        normalized = " ".join(str(text or "").lower().split())
        action_markers = (
            "pick",
            "place",
            "grasp",
            "approach",
            "navigate",
            "find",
            "拿",
            "抓",
            "放",
            "找",
            "靠近",
            "导航",
            "移动到",
        )
        return any(marker in normalized for marker in action_markers)

    def allowed_tools(self, text: str, *, interaction_mode: str) -> set[str] | None:
        if self.is_action_request(text):
            return None
        if self.is_status_query(text):
            return {"get_robot_status"}
        if interaction_mode == "chat":
            if needs_perception_grounding(text):
                return {"request_perception", "get_robot_status", "search_memory"}
            return {"get_robot_status", "search_memory"}
        if interaction_mode == "debug":
            return None
        return None

    @staticmethod
    def recovery_allowed_tools() -> set[str]:
        return {
            "get_task_context",
            "get_robot_status",
            "search_memory",
            "wait",
        }
