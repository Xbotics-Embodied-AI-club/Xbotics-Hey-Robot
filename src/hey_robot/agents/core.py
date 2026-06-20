from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from hey_robot.agents.autonomy import AutonomyManager
from hey_robot.agents.execution_feedback import (
    DefaultExecutionFeedbackEvaluator,
    ExecutionFeedbackEvaluator,
    ImageResolver,
    VisionExecutionFeedbackEvaluator,
    image_resolver_from_root,
)
from hey_robot.agents.io import AgentIO
from hey_robot.agents.memory_context import RobotMemoryContextBuilder
from hey_robot.agents.runtime import AgentRuntime, AgentRuntimeInput, ToolRegistry
from hey_robot.agents.runtime.grounding import is_perception_skill_name
from hey_robot.agents.runtime.prompts import (
    AgentPromptTemplates,
    load_agent_prompt_templates,
)
from hey_robot.agents.runtime.response_policy import looks_like_internal_agent_protocol
from hey_robot.agents.runtime.safety import RobotSafetyHook
from hey_robot.agents.skill_gateway import SkillGateway, SkillGatewayRequest, WaitPolicy
from hey_robot.agents.skill_state import SkillStateMachine
from hey_robot.agents.task_safety import evaluate_user_task
from hey_robot.agents.tool_binding import bind_agent_tools
from hey_robot.agents.turn_modes import decide_turn_mode
from hey_robot.agents.types import AgentCoreResult, AgentTurnInput, RobotSnapshot
from hey_robot.capability.catalog.loader import CapabilityLoader
from hey_robot.capability.catalog.models import CapabilityManifest
from hey_robot.capability.catalog.policy import CapabilityPolicySet
from hey_robot.capability.catalog.resolver import CapabilityResolver
from hey_robot.config import AgentSpec
from hey_robot.logging import HeyRobotLogger
from hey_robot.memory import MemoryRuntime
from hey_robot.protocol import Envelope, SkillIntent, SkillResult
from hey_robot.providers import ReasoningProvider, build_provider
from hey_robot.providers.base import UnconfiguredReasoningProvider
from hey_robot.robots.identity import resolve_robot_family
from hey_robot.skills.base import SkillCatalog
from hey_robot.skills.registry import registry_from_config
from hey_robot.templates.loader import TemplateStore

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
        self.provider: ReasoningProvider = provider or self._build_provider("agent")
        self.skill_catalog = self._configured_skill_catalog()
        self.skill_state = SkillStateMachine()
        self.last_submitted_skill_id: str | None = None
        self._turn_submitted_skill_id: str | None = None
        self.last_feedback_summary: str | None = None
        self.last_next_hint: str | None = None
        self.runtime = self._build_runtime()
        self.feedback_evaluator = feedback_evaluator or self._build_feedback_evaluator()
        self._pending_skills: dict[str, asyncio.Future[str]] = {}
        self._last_contact_envelope: Envelope | None = None
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
            self._memory_path(), autonomy=self.autonomy
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
        self.capabilities = CapabilityLoader(
            tools=self.runtime.tools, robot_skills=self.skill_catalog
        )

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def capability_manifest(self) -> CapabilityManifest:
        return self.capabilities.build(robot_type=self._configured_robot_type())

    async def handle_turn(self, payload: AgentTurnInput) -> AgentCoreResult:
        self.bind_turn_context(payload)
        self._refresh_tool_context()
        self._turn_submitted_skill_id = None
        tool_call_start = len(self.runtime.state.tool_calls)
        logger.debug(
            f"开始处理 turn：agent={self.agent_id} task_len={len(payload.turn.text)} "
            f"robot={payload.turn.envelope.robot_id} mode={self.spec.settings.get('mode', 'agent')}"
        )
        mode = decide_turn_mode(self.spec, payload.turn)
        if mode.is_direct:
            return await self._direct_turn(payload)

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
        reply_text = result.result if result.stop_reason == "text_response" else None
        if (
            reply_text is None
            and result.stop_reason == "max_iterations_after_tool_result"
            and result.tool == "request_capability"
        ):
            reply_text = self._safe_tool_result_reply(result.result)
        if reply_text is None and result.stop_reason in {
            "max_iterations",
            "empty_response",
            "provider_error",
            "internal_protocol_response",
            "invalid_tool_protocol",
        }:
            reply_text = self._fallback_reply_for_unfinished_turn(result)
        task_finished = bool(result.task_finished)
        if not task_finished and reply_text is not None:
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
            f"capability_calls_in_turn={len(self.runtime.state.tool_calls) - tool_call_start}"
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

    def _latest_execution_failure(self, tool_call_start: int) -> str | None:
        for record in reversed(self.runtime.state.tool_calls[tool_call_start:]):
            if record.name != "request_capability" or not record.success:
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
        capability_calls = [
            record
            for record in self.runtime.state.tool_calls[tool_call_start:]
            if record.name == "request_capability" and record.success
        ]
        if not capability_calls:
            logger.debug("final_response 无 capability 调用，视为任务完成")
            return True
        decision = self._latest_feedback_allows_task_completion(
            tool_call_start, final_response=True
        )
        logger.debug(
            f"latest_feedback 判定：decision={decision} "
            f"capability_count={len(capability_calls)}"
        )
        return decision

    def _latest_feedback_allows_task_completion(
        self, tool_call_start: int, *, final_response: bool = False
    ) -> bool:
        for record in reversed(self.runtime.state.tool_calls[tool_call_start:]):
            if record.name != "request_capability" or not record.success:
                continue
            capability = str(record.arguments.get("capability") or "").strip()
            parsed = self._parse_agent_feedback(record.result)
            if parsed is None:
                logger.debug(
                    f"feedback 无法解析（非标准格式），允许完成。capability={capability}"
                )
                return True
            is_perception = is_perception_skill_name(capability)
            logger.debug(
                f"feedback 解析成功：capability={capability} "
                f"task_success={parsed.get('task_success')} "
                f"subgoal_success={parsed.get('subgoal_success')} "
                f"recommended_action={parsed.get('recommended_action')} "
                f"is_perception={is_perception} final_response={final_response}"
            )
            if parsed.get("task_success") is False:
                result = (
                    final_response
                    and parsed.get("subgoal_success") is True
                    and bool(capability)
                )
                logger.debug(f"task_success=False 分支：返回 {result}")
                return result
            result = str(parsed.get("recommended_action") or "").lower() != "continue"
            logger.debug(f"recommended_action 分支：返回 {result}")
            return result
        logger.debug("没有找到有效的 request_capability 记录，返回 False")
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

    async def _direct_turn(self, payload: AgentTurnInput) -> AgentCoreResult:
        if payload.block_actuation:
            return AgentCoreResult(
                reply_text="Recovery is required before submitting a new skill.",
                skill_submitted=False,
                tool="wait",
                metadata={"recovery_required": True},
            )
        skill = await self.skill_gateway.submit_direct(
            objective=payload.turn.text,
            slots={"objective": payload.turn.text, "interrupt": False},
            metadata=dict(payload.turn.metadata),
        )
        return AgentCoreResult(
            reply_text="Skill submitted. Waiting for execution feedback.",
            skill_submitted=True,
            tool="request_capability",
            metadata={"skill_id": skill.skill_id, "status": "skill_issued"},
        )

    def _build_runtime(self) -> AgentRuntime:
        safety_cfg = self.spec.settings.get("safety", {})
        safety_enabled = (
            bool(safety_cfg.get("enabled", False))
            if isinstance(safety_cfg, dict)
            else False
        )
        capability_policy = CapabilityPolicySet.from_dict(
            self.spec.settings.get("capability_policy")
        ).for_mode(str(self.spec.settings.get("mode", "agent")))
        tool_registry = ToolRegistry()
        return AgentRuntime(
            self.provider,
            max_iterations=int(self.spec.settings.get("max_iterations", 8)),
            provider_timeout_sec=float(
                self.spec.settings.get("provider_timeout_sec", 300.0)
            ),
            tool_registry=tool_registry,
            permission_mode=self.spec.settings.get("permission_mode", "autonomous"),
            hooks=[RobotSafetyHook(self._status_snapshot_for_safety)]
            if safety_enabled
            else [],
            capability_resolver=CapabilityResolver(
                tool_registry, policy=capability_policy
            ),
            prompt_templates=self._load_prompt_templates(),
        )

    def _load_prompt_templates(self) -> AgentPromptTemplates:
        return load_agent_prompt_templates(
            template_root=self._template_root(),
            soul_path=self.spec.settings.get("soul_path"),
        )

    def _template_root(self) -> str | Path:
        config = self.spec.settings.get("_deployment_config")
        runtime_dir = (
            Path(config.resources.runtime_dir)
            if config is not None
            else Path("runtime")
        )
        return self.spec.settings.get("template_root") or runtime_dir / "templates"

    def _memory_path(self) -> Path:
        config = self.spec.settings.get("_deployment_config")
        runtime_dir = (
            Path(config.resources.runtime_dir)
            if config is not None
            else Path("runtime")
        )
        return Path(
            self.spec.settings.get("long_term_memory_path")
            or runtime_dir / "memory" / "long_term.jsonl"
        )

    def _build_provider(self, purpose: str) -> ReasoningProvider:
        if purpose == "agent" and self._direct_mode():
            return UnconfiguredReasoningProvider(
                "planner provider is not used in direct mode"
            )
        config = self.spec.settings.get("_deployment_config")
        if config is None:
            raise ValueError(
                f"agent [{self.agent_id}] requires an explicit {purpose} provider configuration; "
                "deterministic fallback has been removed from runtime"
            )
        return build_provider(config, self.agent_id, purpose=purpose)

    def _build_feedback_evaluator(self) -> ExecutionFeedbackEvaluator:
        cfg = self.spec.settings.get("execution_feedback") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        templates = TemplateStore(self._template_root())
        media_root = str(
            cfg.get("media_root")
            or self.spec.settings.get("media_root")
            or "runtime/media"
        )
        resolver = image_resolver_from_root(media_root)
        backend = str(cfg.get("backend", "status")).lower()
        vision_backend: ExecutionFeedbackEvaluator | None = None
        if backend in {"provider", "vlm", "vision", "scene"}:
            vision_backend = VisionExecutionFeedbackEvaluator(
                self._build_provider("feedback"),
                image_resolver=resolver,
                templates=templates,
            )
        return DefaultExecutionFeedbackEvaluator(
            status_backend=backend, vision_backend=vision_backend
        )

    def _register_tools(self) -> None:
        """Register tools via auto-discovery from the ``agents.tools`` package."""
        self._tool_context = bind_agent_tools(self)

    async def request_capability(
        self,
        capability: str,
        objective: str,
        slots: dict[str, Any] | None = None,
        interrupt: bool = False,
        wait_policy: str = "wait_result",
        confirmed: bool = False,
    ) -> str:
        turn = getattr(self, "_current_turn", None)
        turn_metadata = dict(getattr(turn, "metadata", {}) or {})
        return await self.skill_gateway.submit(
            SkillGatewayRequest(
                capability=capability,
                objective=objective,
                slots=slots,
                interrupt=interrupt,
                wait_policy=cast(WaitPolicy, wait_policy),
                metadata=turn_metadata,
                confirmed=confirmed,
            )
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
        await self.request_capability(
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
        from hey_robot.agents.tools.context import ToolTurnContext

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
            "Robot capability catalog for request_capability.capability:",
            "- Choose request_capability.capability exactly from this catalog. Do not invent capability names.",
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

    def _configured_skill_catalog(self) -> SkillCatalog:
        config = self.spec.settings.get("_deployment_config")
        enabled_only = bool(getattr(getattr(config, "skills", None), "enabled", ()))
        if config is None:
            enabled_only = False
        catalog = registry_from_config(config).catalog(
            enabled_only=enabled_only,
        )
        mode = (
            getattr(getattr(config, "skills", None), "mode", "production")
            if config is not None
            else "production"
        )
        if mode == "production" and enabled_only:
            catalog = catalog.semantic_skills()
        return catalog

    @staticmethod
    def _fallback_reply_for_unfinished_turn(result: Any) -> str:
        reason = str(result.result or result.reason or result.stop_reason or "").strip()
        if result.stop_reason == "provider_error":
            return f"模型服务这次没有成功返回可用结果：{reason}"
        if result.stop_reason in {
            "internal_protocol_response",
            "invalid_tool_protocol",
        }:
            return "我已经收到上一步执行结果，但还没有形成可靠的最终结论，会继续根据最新观测推进任务。"
        return "我这次没有成功形成可执行动作或最终答复，请换一种更具体的说法再试。"

    @staticmethod
    def _safe_tool_result_reply(text: str) -> str | None:
        cleaned = str(text or "").strip()
        if not cleaned or looks_like_internal_agent_protocol(cleaned):
            return None
        return cleaned

    def _current_envelope(self):
        return self._current_turn.envelope

    def _direct_mode(self) -> bool:
        return str(self.spec.settings.get("mode", "agent")).lower() == "direct"

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
