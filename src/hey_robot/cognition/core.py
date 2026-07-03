from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, cast

from hey_robot.cognition.command_router import CommandRouter, RoutedCommand
from hey_robot.cognition.core_builder import RobotAgentCoreBuilder
from hey_robot.cognition.execution_feedback import (
    ExecutionFeedbackEvaluator,
    ImageResolver,
)
from hey_robot.cognition.io import AgentIO
from hey_robot.cognition.memory import MemoryRuntime
from hey_robot.cognition.memory_context import RobotMemoryContextBuilder
from hey_robot.cognition.runtime import AgentRuntimeInput
from hey_robot.cognition.runtime.grounding import is_perception_skill_name
from hey_robot.cognition.runtime.response_policy import (
    looks_like_internal_agent_protocol,
)
from hey_robot.cognition.scene_evidence import reusable_scene_evidence_result
from hey_robot.cognition.skill_gateway import (
    SkillGateway,
    SkillGatewayRequest,
    WaitPolicy,
)
from hey_robot.cognition.skill_state import SkillStateMachine
from hey_robot.cognition.task_safety import evaluate_user_task
from hey_robot.cognition.tool_binding import bind_agent_tools
from hey_robot.cognition.types import AgentCoreResult, AgentTurnInput, RobotSnapshot
from hey_robot.config import AgentSpec
from hey_robot.foundation.catalog.loader import SkillSurfaceLoader
from hey_robot.foundation.catalog.models import SkillSurfaceManifest
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import Envelope, SkillIntent, SkillResult
from hey_robot.providers import ReasoningProvider
from hey_robot.robot_runtime.identity import resolve_robot_family
from hey_robot.user_reply import present_tool_result_for_user

logger = HeyRobotLogger(name="core")


class RobotAgentCore:
    """Protocol-native robot agent core.

    The core owns the model provider and tools. It does not own bus, channel, episodes, or
    robot drivers; those are supplied through AgentIO and RobotSnapshot.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        spec: AgentSpec,
        io: AgentIO,
        media_resolver: ImageResolver | None = None,
        provider: ReasoningProvider | None = None,
        feedback_evaluator: ExecutionFeedbackEvaluator | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.spec = spec
        self.io = io
        self.media_resolver = media_resolver
        self.builder = RobotAgentCoreBuilder(agent_id=agent_id, spec=spec)
        self.provider: ReasoningProvider = provider or self.builder.build_provider(
            "agent"
        )
        self.skill_catalog = self.builder.configured_skill_catalog()
        self.skill_state = SkillStateMachine()
        self.last_submitted_skill_id: str | None = None
        self._turn_submitted_skill_id: str | None = None
        self.last_feedback_summary: str | None = None
        self.last_next_hint: str | None = None
        self.runtime = self.builder.build_runtime(
            self.provider,
            status_snapshot_provider=self._status_snapshot_for_safety,
        )
        self.feedback_evaluator = (
            feedback_evaluator or self.builder.build_feedback_evaluator()
        )
        self.command_router = CommandRouter()
        self._pending_skills: dict[str, asyncio.Future[str]] = {}
        self._current_tool_call_start = 0
        self._last_contact_envelope: Envelope | None = None
        from hey_robot.cognition.autonomy import AutonomyManager

        self.autonomy = AutonomyManager(
            max_events=int(self.spec.settings.get("autonomy", {}).get("max_events", 50))
            if isinstance(self.spec.settings.get("autonomy"), dict)
            else 50,
            default_goal=(
                self.spec.settings.get("autonomy", {}).get("default_goal")
                if isinstance(self.spec.settings.get("autonomy"), dict)
                else None
            ),
        )
        self.memory = MemoryRuntime.from_path(
            self.builder.memory_path(), autonomy=self.autonomy
        )
        self.runtime.memory = self.memory
        self.memory_context_builder = RobotMemoryContextBuilder(
            memory=self.memory,
            robot_skill_catalog_context_provider=self._robot_skill_catalog_context,
        )
        self.skill_gateway = SkillGateway(
            io=self.io,
            spec=self.spec,
            skill_catalog=self.skill_catalog,
            runtime_state=self.runtime.state,
            pending_skills=self._pending_skills,
            current_envelope=self._current_envelope,
            get_task=lambda: self.runtime.state.task,
            on_submit=self._observe_submitted_skill,
            recovery_required=lambda: bool(getattr(self, "_recovery_required", False)),
            task_runtime=getattr(self.io, "task_runtime", None),
        )
        self._tool_context: Any = (
            None  # ToolContext 鈥?populated by _register_tools when class-based
        )
        self._register_tools()
        self.skill_surface = SkillSurfaceLoader(
            tools=self.runtime.tools, robot_skills=self.skill_catalog
        )

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def skill_surface_manifest(self) -> SkillSurfaceManifest:
        return self.skill_surface.build(robot_type=self._configured_robot_type())

    async def handle_turn(self, payload: AgentTurnInput) -> AgentCoreResult:
        self.bind_turn_context(payload)
        self._refresh_tool_context()
        self._turn_submitted_skill_id = None
        tool_call_start = len(self.runtime.state.tool_calls)
        self._current_tool_call_start = tool_call_start
        logger.debug(
            f"开始处理 turn：agent={self.agent_id} task_len={len(payload.turn.text)} "
            f"robot={payload.turn.envelope.robot_id} mode={self.spec.settings.get('mode', 'agent')}"
        )
        safety_decision = evaluate_user_task(
            payload.turn.text,
            channel=payload.turn.envelope.channel,
            settings=self.spec.settings,
        )
        if not safety_decision.allowed:
            return AgentCoreResult(
                reply_text=safety_decision.reply or safety_decision.reason,
                skill_submitted=False,
                task_finished=True,
                tool="task_safety",
                metadata={
                    "stop_reason": "task_safety_blocked",
                    "safety_rule": safety_decision.rule,
                    "safety_reason": safety_decision.reason,
                },
            )

        routed_command = self.command_router.route(payload.turn.text)
        if routed_command is not None:
            return await self._handle_routed_command(payload, routed_command)

        memory_context = self.memory_context_builder.build(
            task=payload.turn.text,
            task_context=payload.memory_context,
            perception_context=payload.perception_context,
        )
        result = await self.runtime.step(
            AgentRuntimeInput(
                task=payload.turn.text,
                images=self._snapshot_images(payload.snapshot),
                robot_state=payload.snapshot.summary(),
                robot_status=(
                    asdict(payload.snapshot.status)
                    if payload.snapshot.status is not None
                    else None
                ),
                memory_context=memory_context,
                autonomy_context=self.autonomy.prompt_context() or None,
                last_feedback=self.last_feedback_summary,
                next_hint=self._next_hint(),
                skill_in_progress=not self.skill_state.snapshot.is_terminal,
                recovery_context=payload.recovery_context,
                allowed_tools=payload.allowed_tools,
            )
        )
        logger.debug(
            f"runtime 执行完成：agent={self.agent_id} tool={result.tool} "
            f"task_finished={result.task_finished} stop_reason={result.stop_reason}"
        )
        reply_text = self._reply_text_from_runtime_result(result)
        if (
            reply_text is None
            and result.stop_reason == "max_iterations_after_tool_result"
            and result.tool == "request_skill"
        ):
            reply_text = self._safe_tool_result_reply(result.result)
        if reply_text is None and result.stop_reason in {
            "max_iterations",
            "empty_response",
            "provider_error",
            "model_timeout",
            "turn_budget_exhausted",
            "internal_protocol_response",
            "invalid_tool_protocol",
        }:
            reply_text = self._fallback_reply_for_unfinished_turn(result)
        task_finished = bool(result.task_finished)
        if (
            not task_finished
            and reply_text is not None
            and not bool(getattr(result, "task_evaluation_applied", False))
        ):
            task_finished = self._final_response_finishes_task(
                result, tool_call_start=tool_call_start
            )
        execution_failure = self._latest_execution_failure(tool_call_start)
        if reply_text is not None and execution_failure:
            reply_text = f"动作执行未成功：{execution_failure}"
            task_finished = False
        logger.info(
            f"turn 完成判定：agent={self.agent_id} tool={result.tool} "
            f"stop_reason={result.stop_reason} reply_len={len(reply_text or '')} "
            f"task_finished={task_finished} "
            f"skill_calls_in_turn={len(self.runtime.state.tool_calls) - tool_call_start}"
        )
        return AgentCoreResult(
            reply_text=reply_text,
            skill_submitted=False,
            task_finished=task_finished,
            tool=result.tool,
            metadata={
                "tool": result.tool,
                "args": result.args,
                "result": result.result,
                "skill_id": self._turn_submitted_skill_id,
                "stop_reason": result.stop_reason,
            },
        )

    async def _handle_routed_command(
        self, payload: AgentTurnInput, command: RoutedCommand
    ) -> AgentCoreResult:
        self.runtime.state.task = payload.turn.text
        try:
            result_text = await self.skill_gateway.submit(
                SkillGatewayRequest(
                    skill=command.skill,
                    objective=command.objective,
                    slots=command.slots,
                    interrupt=command.interrupt,
                    wait_policy=command.wait_policy,
                    metadata={
                        **dict(payload.turn.metadata or {}),
                        "command_router": True,
                    },
                    result_prefix="command",
                    enforce_motion_guards=command.skill
                    not in {"stop_motion", "reset_posture"},
                    confirmed=True,
                )
            )
        except Exception as exc:
            return AgentCoreResult(
                reply_text=f"指令没有成功下发：{exc}",
                skill_submitted=False,
                task_finished=False,
                tool="request_skill",
                metadata={
                    "tool": "request_skill",
                    "skill": command.skill,
                    "stop_reason": "command_router_failed",
                    "error": str(exc),
                },
            )
        return AgentCoreResult(
            reply_text=command.reply_text,
            skill_submitted=True,
            task_finished=False,
            tool="request_skill",
            metadata={
                "tool": "request_skill",
                "args": {
                    "skill": command.skill,
                    "objective": command.objective,
                    "slots": command.slots,
                    "interrupt": command.interrupt,
                    "wait_policy": command.wait_policy,
                },
                "result": result_text,
                "skill_id": self._turn_submitted_skill_id,
                "stop_reason": "command_router",
            },
        )

    def _latest_execution_failure(self, tool_call_start: int) -> str | None:
        for record in reversed(self.runtime.state.tool_calls[tool_call_start:]):
            if record.name != "request_skill" or not record.success:
                continue
            parsed = self._parse_agent_feedback(record.result)
            if parsed is None or parsed.get("subgoal_success") is not False:
                return None
            return str(
                parsed.get("failure_reason")
                or parsed.get("summary")
                or "机器人未确认动作成功"
            ).strip()
        return None

    def _final_response_finishes_task(
        self, result: Any, *, tool_call_start: int
    ) -> bool:
        if result.tool != "final_response" or result.stop_reason != "text_response":
            return False
        if looks_like_internal_agent_protocol(str(result.result or "")):
            return False
        skill_calls = [
            record
            for record in self.runtime.state.tool_calls[tool_call_start:]
            if record.name == "request_skill" and record.success
        ]
        if not skill_calls:
            logger.debug("final_response 无 skill 调用，视为任务完成")
            return True
        decision = self._latest_feedback_allows_task_completion(
            tool_call_start, final_response=True
        )
        logger.debug(
            f"latest_feedback 判定：decision={decision} skill_count={len(skill_calls)}"
        )
        return decision

    def _latest_feedback_allows_task_completion(
        self, tool_call_start: int, *, final_response: bool = False
    ) -> bool:
        for record in reversed(self.runtime.state.tool_calls[tool_call_start:]):
            if record.name != "request_skill" or not record.success:
                continue
            skill = str(record.arguments.get("skill") or "").strip()
            parsed = self._parse_agent_feedback(record.result)
            if parsed is None:
                logger.debug(
                    f"feedback 无法解析（非标准格式），允许完成。skill={skill}"
                )
                return True
            is_perception = is_perception_skill_name(skill)
            logger.debug(
                f"feedback 解析成功：skill={skill} "
                f"task_success={parsed.get('task_success')} "
                f"subgoal_success={parsed.get('subgoal_success')} "
                f"recommended_action={parsed.get('recommended_action')} "
                f"is_perception={is_perception} final_response={final_response}"
            )
            if parsed.get("task_success") is False:
                result = (
                    final_response
                    and parsed.get("subgoal_success") is True
                    and bool(skill)
                )
                logger.debug(f"task_success=False 分支：返回 {result}")
                return result
            result = str(parsed.get("recommended_action") or "").lower() != "continue"
            logger.debug(f"recommended_action 分支：返回 {result}")
            return result
        logger.debug("没有找到有效的 request_skill 记录，返回 False")
        return False

    def observe_skill_result(
        self, skill_id: str, status: str, error: str | None = None
    ) -> None:
        turn = getattr(self, "_current_turn", None)
        self.skill_state.observe_result(
            SkillResult(
                envelope=turn.envelope if turn is not None else Envelope(),
                skill_id=skill_id,
                status=status,
                success=status == "completed",
                error=error,
            ),
        )

    def _observe_submitted_skill(self, skill: SkillIntent) -> None:
        self.last_submitted_skill_id = skill.skill_id
        self._turn_submitted_skill_id = skill.skill_id
        self.skill_state.submit(skill)

    def _register_tools(self) -> None:
        """Register tools via auto-discovery from the ``agents.tools`` package."""
        self._tool_context = bind_agent_tools(self)

    async def request_skill(
        self,
        skill: str,
        objective: str,
        slots: dict[str, Any] | None = None,
        interrupt: bool = False,
        wait_policy: str = "wait_result",
        confirmed: bool = False,
    ) -> str:
        turn = getattr(self, "_current_turn", None)
        turn_metadata = dict(getattr(turn, "metadata", {}) or {})
        duplicate_perception = self._successful_perception_result_this_turn(skill)
        if duplicate_perception is not None:
            logger.info(f"reuse perception result in current turn: skill={skill}")
            return duplicate_perception
        return await self.skill_gateway.submit(
            SkillGatewayRequest(
                skill=skill,
                objective=objective,
                slots=slots,
                interrupt=interrupt,
                wait_policy=cast(WaitPolicy, wait_policy),
                metadata=turn_metadata,
                confirmed=confirmed,
            )
        )

    def _successful_perception_result_this_turn(self, skill: str) -> str | None:
        return reusable_scene_evidence_result(
            self.runtime.state.tool_calls[self._current_tool_call_start :],
            skill,
            parse_feedback=self._parse_agent_feedback,
        )

    def resolve_skill(self, skill_id: str, result_text: str) -> bool:
        future = self._pending_skills.get(skill_id)
        if future is not None and not future.done():
            future.set_result(result_text)
            return True
        return False

    def is_waiting_for_skill(self, skill_id: str) -> bool:
        future = self._pending_skills.get(skill_id)
        return future is not None and not future.done()

    def get_robot_status(self) -> str:
        return getattr(self, "_current_snapshot_summary", "no robot snapshot")

    def get_observation_summary(self) -> str:
        return getattr(self, "_current_observation_summary", "no observation")

    async def request_perception(
        self,
        modality: str = "vision",
        scope: str = "current_scene",
        freshness: str = "fresh",
        question: str = "",
    ) -> str:
        if modality and modality.lower() not in {"vision", "image", "camera"}:
            raise ValueError(f"unsupported perception modality: {modality}")
        if scope and scope.lower() not in {
            "current_scene",
            "front",
            "front_view",
            "execution_result",
        }:
            raise ValueError(f"unsupported perception scope: {scope}")
        skill_name = "inspect_scene"
        objective = (
            question or self.runtime.state.task or "inspect current scene"
        ).strip()
        snapshot = getattr(self, "_current_snapshot", None)
        baseline_frame_id = (
            snapshot.observation.frame_id if snapshot and snapshot.observation else None
        )
        await self.request_skill(
            skill_name,
            objective=objective,
            slots={"question": objective},
            interrupt=False,
        )
        evidence = await self._query_scene_evidence_dict(
            question=objective,
            baseline_frame_id=baseline_frame_id,
            freshness=freshness or "fresh",
            source="request_perception",
        )
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

    async def _query_scene_evidence_dict(
        self,
        *,
        question: str,
        baseline_frame_id: int | None,
        freshness: str,
        source: str,
    ) -> dict[str, Any]:
        query_scene_evidence = getattr(self.io, "query_scene_evidence", None)
        if query_scene_evidence is None:
            return {
                "status": "caption_failed",
                "frame_id": None,
                "image_count": 0,
                "summary": "",
                "confidence": None,
                "objects": [],
                "risks": ["scene evidence query unavailable"],
                "next_observation_hint": "Use an AgentIO implementation that provides query_scene_evidence.",
                "source": source,
                "metadata": {
                    "baseline_frame_id": baseline_frame_id,
                    "question": question,
                },
            }
        scene_timeout = float(self.spec.settings.get("scene_evidence_timeout_sec", 2.0))
        scene_evidence = await query_scene_evidence(
            robot_id=self._current_envelope().robot_id,
            question=question,
            baseline_frame_id=baseline_frame_id,
            freshness=freshness,
            timeout_sec=scene_timeout,
        )
        evidence: dict[str, Any] = scene_evidence.to_dict()
        evidence["source"] = source
        evidence["metadata"] = {
            **dict(evidence.get("metadata") or {}),
            "baseline_frame_id": baseline_frame_id,
            "question": question,
        }
        return evidence

    @staticmethod
    def _parse_agent_feedback(text: str) -> dict[str, Any] | None:
        stripped = (text or "").strip()
        prefix = "Execution feedback for skill "
        if not stripped.startswith(prefix):
            return None
        parsed: dict[str, Any] = {}
        for line in stripped.splitlines()[1:]:
            line = line.strip()
            if not line.startswith("- "):
                continue
            key, sep, value = line[2:].partition(":")
            if not sep:
                continue
            parsed[key.strip()] = RobotAgentCore._parse_feedback_value(value.strip())
        return parsed

    @staticmethod
    def _parse_feedback_value(value: str) -> Any:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _clean_feedback_text(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        for marker in ("; robot_state=", "\n", "\r"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        generic = {
            "base moved",
            "inspect_scene completed",
            "stop_motion completed",
            "move_base completed",
            "turn_base completed",
            "reset_posture completed",
            "set_gripper completed",
            "gripper opening set",
            "gripper closed",
            "gripper opened",
        }
        if cleaned.lower() in generic:
            return ""
        return cleaned.rstrip("。.")

    @staticmethod
    def _format_number(value: int | float) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def bind_turn_context(self, payload: AgentTurnInput) -> None:
        self._current_turn = payload.turn
        self._last_contact_envelope = payload.turn.envelope
        self._current_snapshot = payload.snapshot
        self._recovery_required = payload.block_actuation
        self._current_snapshot_summary = payload.snapshot.summary()
        obs = payload.snapshot.observation
        self._current_observation_summary = (
            f"frame_id={obs.frame_id} images={len(obs.images)} task={obs.task}"
            if obs is not None
            else "no observation"
        )

    def _refresh_tool_context(self) -> None:
        """Update the per-turn snapshot on the class-based tool context."""
        ctx = self._tool_context
        if ctx is None:
            return
        from hey_robot.cognition.tools.context import ToolTurnContext

        ctx.turn_context = ToolTurnContext(
            snapshot_summary=self._current_snapshot_summary,
            observation_summary=self._current_observation_summary,
            snapshot=self._current_snapshot,
            envelope=self._current_envelope()
            if hasattr(self, "_current_turn") and self._current_turn is not None
            else None,
            recovery_required=bool(getattr(self, "_recovery_required", False)),
        )

    def _robot_skill_catalog_context(self) -> str:
        skills = self.skill_catalog.list()
        if not skills:
            return ""
        lines = [
            "Robot skill catalog for request_skill.skill:",
            "- Choose request_skill.skill exactly from this catalog. Do not invent skill names.",
            "- Use the skill description and input_schema to choose arguments.",
        ]
        for skill in skills:
            required = (
                skill.input_schema.get("required")
                if isinstance(skill.input_schema, dict)
                else None
            )
            required_text = f" required={required}" if required else ""
            resources = (
                f" resources={list(skill.required_resources)}"
                if skill.required_resources
                else ""
            )
            lines.append(
                f"- {skill.name}: {skill.description}{required_text}{resources}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback_reply_for_unfinished_turn(result: Any) -> str:
        reason = str(result.result or result.reason or result.stop_reason or "").strip()
        if result.stop_reason == "provider_error":
            return f"模型服务这次没有成功返回可用结果：{reason}"
        if result.stop_reason == "model_timeout":
            return "模型服务这次没有在限定时间内返回可用结果。当前不会继续执行新的动作，请稍后重试。"
        if result.stop_reason == "turn_budget_exhausted":
            return "这次任务处理超时，当前不会继续执行新的动作。建议先重新观察或重试。"
        if result.stop_reason in {
            "internal_protocol_response",
            "invalid_tool_protocol",
        }:
            return "我已经收到上一步执行结果，但还没有形成可靠的最终结论，会继续根据最新观测推进任务。"
        return (
            "这次没有形成可靠的动作或最终答复，当前不会继续执行新的动作。请稍后重试。"
        )

    @staticmethod
    def _safe_tool_result_reply(text: str) -> str | None:
        cleaned = str(text or "").strip()
        if not cleaned or looks_like_internal_agent_protocol(cleaned):
            return None
        return cleaned

    def _reply_text_from_runtime_result(self, result: Any) -> str | None:
        if result.stop_reason != "text_response":
            return None
        text = str(result.result or "").strip()
        if result.tool == "final_response":
            if text and not self._looks_like_internal_final_response(text):
                return text
            return self._fallback_reply_for_unfinished_turn(result)
        return present_tool_result_for_user(
            tool=str(result.tool or ""),
            args=dict(result.args or {}),
            result=text,
            success=result.tool_success,
        ) or self._fallback_reply_for_unfinished_turn(result)

    @staticmethod
    def _looks_like_internal_final_response(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().split())
        return normalized.startswith(("用户说", "用户表示")) or (
            "回顾一下之前的进展" in normalized
        )

    def _current_envelope(self):
        return self._current_turn.envelope

    def _next_hint(self) -> str | None:
        return self.last_next_hint

    def _status_snapshot_for_safety(self) -> dict[str, Any] | None:
        snapshot = getattr(self, "_current_snapshot", None)
        if snapshot is None or snapshot.status is None:
            return None
        status = snapshot.status
        recent_tool_calls = [
            {
                "name": record.name,
                "arguments": dict(record.arguments),
                "success": bool(record.success),
            }
            for record in self.runtime.state.tool_calls[-12:]
        ]
        return {
            "frame_id": status.frame_id,
            "state": status.state,
            "error": status.error,
            "recent_tool_calls": recent_tool_calls,
            **status.metrics,
        }

    def _snapshot_images(self, snapshot: RobotSnapshot) -> list[Any]:
        send_images = self.spec.settings.get("send_images_on_turn", False)
        if not send_images:
            return []
        observation = snapshot.observation
        if observation is None or self.media_resolver is None:
            return []
        return self.media_resolver.resolve_images(observation.images[:4])

    def _configured_robot_type(self) -> str | None:
        override = self.spec.settings.get(
            "robot_skill_catalog_type"
        ) or self.spec.settings.get("embodiment_type")
        if override:
            return str(override)
        config = self.spec.settings.get("_deployment_config")
        if config is None or self.spec.robot_id is None:
            return self.spec.robot_id
        return resolve_robot_family(
            config, self.spec.robot_id, fallback=self.spec.robot_id
        )
