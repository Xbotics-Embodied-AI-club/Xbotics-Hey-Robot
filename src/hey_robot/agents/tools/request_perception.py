import json
from typing import cast

from hey_robot.agents.skill_gateway import SkillGateway, SkillGatewayRequest, WaitPolicy
from hey_robot.agents.tools.base import Tool, tool_parameters
from hey_robot.agents.tools.context import ToolContext
from hey_robot.agents.tools.schema import StringSchema, tool_parameters_schema

_VALID_MODALITIES = {"vision", "image", "camera"}
_VALID_SCOPES = {"current_scene", "front", "front_view", "execution_result"}
_VALID_WAIT_POLICIES = {"wait_result", "wait_acceptance", "return_handle"}


@tool_parameters(
    tool_parameters_schema(
        modality=StringSchema("vision, image, or camera"),
        scope=StringSchema("current_scene, front, or execution_result"),
        freshness=StringSchema("fresh or cached"),
        question=StringSchema("What to look for"),
        wait_policy=StringSchema(
            "wait_result, wait_acceptance, or return_handle",
            enum=["wait_result", "wait_acceptance", "return_handle"],
        ),
    )
)
class RequestPerceptionTool(Tool):
    name = "request_perception"
    description = "Request observe-only grounded perception evidence from the robot."
    safety_level = "observe"
    exclusive = True
    resources = ("robot.perception", "camera")
    timeout_sec = 30.0
    result_policy = "require_final_answer"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._io = ctx.io
        self._spec = ctx.spec
        self._runtime_state = ctx.runtime_state
        self._current_envelope = ctx._current_envelope
        self._gateway = cast(SkillGateway | None, getattr(ctx, "skill_gateway", None))

    @classmethod
    def create(cls, ctx: ToolContext):
        return cls(ctx)

    async def execute(
        self,
        modality: str = "vision",
        scope: str = "current_scene",
        freshness: str = "fresh",
        question: str = "",
        wait_policy: str = "wait_result",
    ) -> str:
        if modality and modality.lower() not in _VALID_MODALITIES:
            raise ValueError(f"unsupported perception modality: {modality}")
        if scope and scope.lower() not in _VALID_SCOPES:
            raise ValueError(f"unsupported perception scope: {scope}")
        normalized_wait_policy = wait_policy or "wait_result"
        if normalized_wait_policy not in _VALID_WAIT_POLICIES:
            raise ValueError(f"unknown wait_policy: {wait_policy}")

        skill_name = "inspect_scene"
        objective = (
            question or self._runtime_state.task or "inspect current scene"
        ).strip()

        # Use internal skill submission (lightweight, no LLM round-trip)
        tc = self._ctx.turn_context if hasattr(self, "_ctx") else None
        snapshot = tc.snapshot if tc else None
        baseline_frame_id = (
            snapshot.observation.frame_id if snapshot and snapshot.observation else None
        )

        skill_result = await self._submit_internal_skill(
            skill_name,
            objective,
            wait_policy=cast(WaitPolicy, normalized_wait_policy),
        )
        if skill_result is not None:
            return skill_result

        query_scene_evidence = getattr(self._io, "query_scene_evidence", None)
        if query_scene_evidence is None:
            evidence = {
                "status": "caption_failed",
                "frame_id": None,
                "image_count": 0,
                "summary": "Scene evidence query is not available in this agent IO.",
                "confidence": None,
                "objects": [],
                "risks": ["scene evidence query unavailable"],
                "next_observation_hint": "Use an AgentIO implementation that provides query_scene_evidence.",
                "source": "request_perception",
                "metadata": {"baseline_frame_id": baseline_frame_id},
            }
        else:
            scene_timeout = float(
                self._spec.settings.get("scene_evidence_timeout_sec", 2.0)
            )
            scene_evidence = await query_scene_evidence(
                robot_id=self._current_envelope().robot_id,
                question=objective,
                baseline_frame_id=baseline_frame_id,
                freshness=freshness or "fresh",
                timeout_sec=scene_timeout,
            )
            evidence = scene_evidence.to_dict()

        return json.dumps(
            {
                "tool": "request_perception",
                "evidence_status": "ok"
                if evidence.get("status") == "ok"
                else "degraded",
                "modality": modality or "vision",
                "scope": scope or "current_scene",
                "freshness": freshness or "fresh",
                "evidence": evidence,
                "result": evidence.get("summary") or "",
            },
            ensure_ascii=False,
        )

    async def _submit_internal_skill(
        self, skill_name: str, objective: str, *, wait_policy: WaitPolicy
    ) -> str | None:
        """Submit an internal perception skill directly."""
        if self._gateway is None:
            raise RuntimeError("skill gateway is not configured")
        if wait_policy == "return_handle":
            return await self._gateway.submit(
                SkillGatewayRequest(
                    capability=skill_name,
                    objective=objective,
                    slots={"question": objective},
                    wait_policy="return_handle",
                    result_prefix="perception",
                )
            )
        if wait_policy == "wait_acceptance":
            return await self._gateway.submit(
                SkillGatewayRequest(
                    capability=skill_name,
                    objective=objective,
                    slots={"question": objective},
                    wait_policy="wait_acceptance",
                    result_prefix="perception",
                )
            )
        await self._gateway.submit(
            SkillGatewayRequest(
                capability=skill_name,
                objective=objective,
                slots={"question": objective},
                wait_policy="wait_result",
                result_prefix="perception",
            )
        )
        return None
