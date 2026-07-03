from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hey_robot.cognition.execution_feedback import (
    DefaultExecutionFeedbackEvaluator,
    ExecutionFeedbackEvaluator,
    VisionExecutionFeedbackEvaluator,
    image_resolver_from_root,
)
from hey_robot.cognition.runtime import AgentRuntime, ToolRegistry
from hey_robot.cognition.runtime.prompts import (
    AgentPromptTemplates,
    load_agent_prompt_templates,
)
from hey_robot.cognition.runtime.safety import RobotSafetyHook
from hey_robot.config import AgentSpec
from hey_robot.foundation.catalog.policy import ToolPolicySet
from hey_robot.foundation.catalog.resolver import ToolPolicyResolver
from hey_robot.providers import ReasoningProvider, build_provider
from hey_robot.skill_os.base import SkillCatalog
from hey_robot.skill_os.registry import registry_from_config
from hey_robot.templates.loader import TemplateStore


@dataclass(frozen=True)
class RobotAgentCoreBuilder:
    """Builds long-lived dependencies for RobotAgentCore."""

    agent_id: str
    spec: AgentSpec

    def build_provider(self, purpose: str) -> ReasoningProvider:
        config = self.spec.settings.get("_deployment_config")
        if config is None:
            raise ValueError(
                f"agent [{self.agent_id}] requires an explicit {purpose} provider configuration; "
                "deterministic fallback has been removed from runtime"
            )
        return build_provider(config, self.agent_id, purpose=purpose)

    def build_runtime(
        self,
        provider: ReasoningProvider,
        *,
        status_snapshot_provider: Callable[[], dict[str, Any] | None],
    ) -> AgentRuntime:
        safety_cfg = self.spec.settings.get("safety", {})
        safety_enabled = (
            bool(safety_cfg.get("enabled", False))
            if isinstance(safety_cfg, dict)
            else False
        )
        tool_policy = ToolPolicySet.from_dict(
            self.spec.settings.get("tool_policy")
        ).for_mode(str(self.spec.settings.get("mode", "agent")))
        tool_registry = ToolRegistry()
        return AgentRuntime(
            provider,
            max_iterations=int(self.spec.settings.get("max_iterations", 8)),
            provider_timeout_sec=float(
                self.spec.settings.get("provider_timeout_sec", 300.0)
            ),
            turn_timeout_sec=float(self.spec.settings.get("turn_timeout_sec", 120.0)),
            tool_registry=tool_registry,
            permission_mode=self.spec.settings.get("permission_mode", "autonomous"),
            hooks=[RobotSafetyHook(status_snapshot_provider)] if safety_enabled else [],
            tool_policy_resolver=ToolPolicyResolver(tool_registry, policy=tool_policy),
            prompt_templates=self.load_prompt_templates(),
        )

    def load_prompt_templates(self) -> AgentPromptTemplates:
        return load_agent_prompt_templates(
            template_root=self.template_root(),
            soul_path=self.spec.settings.get("soul_path"),
        )

    def template_root(self) -> str | Path:
        config = self.spec.settings.get("_deployment_config")
        runtime_dir = (
            Path(config.resources.runtime_dir)
            if config is not None
            else Path("runtime")
        )
        return self.spec.settings.get("template_root") or runtime_dir / "templates"

    def memory_path(self) -> Path:
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

    def build_feedback_evaluator(self) -> ExecutionFeedbackEvaluator:
        cfg = self.spec.settings.get("execution_feedback") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        templates = TemplateStore(self.template_root())
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
                self.build_provider("feedback"),
                image_resolver=resolver,
                templates=templates,
            )
        return DefaultExecutionFeedbackEvaluator(
            status_backend=backend, vision_backend=vision_backend
        )

    def configured_skill_catalog(self) -> SkillCatalog:
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
