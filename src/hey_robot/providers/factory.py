from __future__ import annotations

import os
from typing import Any

from hey_robot.config import DeploymentConfig
from hey_robot.providers.base import GenerationSettings
from hey_robot.providers.fallback_provider import (
    FallbackCandidate,
    FallbackReasoningProvider,
)
from hey_robot.providers.openai_compat_provider import OpenAICompatReasoningProvider
from hey_robot.providers.registry import find_provider
from hey_robot.providers.static_feedback import DeterministicExecutionFeedbackReasoner
from hey_robot.providers.types import ReasoningProvider


def build_provider(
    config: DeploymentConfig, agent_id: str, *, purpose: str = "planner"
) -> ReasoningProvider:
    agent = config.agents[agent_id]
    provider_config = _purpose_config(agent.settings, purpose)
    provider = _build_provider(provider_config, purpose=purpose)
    fallback_configs = list(provider_config.get("fallback_models", []) or [])
    if not fallback_configs:
        return provider
    fallbacks = []
    for item in fallback_configs:
        fallback_provider = _build_provider(item, purpose=purpose)
        fallbacks.append(
            FallbackCandidate(
                provider=fallback_provider, model=fallback_provider.get_default_model()
            )
        )
    return FallbackReasoningProvider(provider, fallbacks)


def _purpose_config(settings: dict[str, Any], purpose: str) -> dict[str, Any]:
    providers = settings.get("providers")
    if isinstance(providers, dict):
        if purpose == "agent":
            planner_cfg = providers.get("planner")
            if isinstance(planner_cfg, dict):
                return planner_cfg
        purpose_cfg = providers.get(purpose)
        if isinstance(purpose_cfg, dict):
            return purpose_cfg
        return {}
    return {}


def _build_provider(cfg: dict[str, Any], *, purpose: str) -> ReasoningProvider:
    provider_type = str(cfg.get("type") or cfg.get("provider") or "").lower()
    if not provider_type:
        raise ValueError(f"missing provider config for purpose={purpose}")
    if provider_type == "deterministic":
        if purpose == "feedback":
            return DeterministicExecutionFeedbackReasoner()
        raise ValueError(
            "deterministic planner has been removed; configure a real planner provider"
        )
    if provider_type in {"local", "none"}:
        raise ValueError(f"unsupported provider for purpose={purpose}: {provider_type}")
    generation = GenerationSettings(
        temperature=float(cfg.get("temperature", 0.1)),
        max_tokens=int(cfg.get("max_tokens", 2048)),
        reasoning_effort=cfg.get("reasoning_effort"),
    )
    if provider_type in {
        "openai",
        "openai_compat",
        "openai-compatible",
        "openrouter",
        "deepseek",
        "dashscope",
        "ark",
        "doubao",
        "minimax",
        "custom",
    }:
        model = _resolve_model(cfg)
        configured_api_base = _resolve_configured_api_base(cfg)
        provider = find_provider(
            provider_type,
            model=model,
            api_base=configured_api_base,
        )
        api_base = configured_api_base or provider.default_api_base
        headers = {
            **provider.default_headers,
            **dict(cfg.get("extra_headers", {}) or {}),
        }
        return OpenAICompatReasoningProvider(
            model=model,
            api_key=_resolve_api_key(cfg),
            api_base=api_base,
            generation=generation,
            extra_headers=headers or None,
            extra_body=dict(cfg.get("extra_body", {}) or {}) or None,
            use_responses_api=cfg.get("use_responses_api"),
            provider_name=provider.name,
            supports_required_tool_choice=provider.name != "deepseek",
        )
    raise ValueError(f"unsupported provider: {provider_type}")


def _resolve_model(cfg: dict[str, Any]) -> str:
    model = str(cfg.get("model") or "").strip()
    if model:
        return model
    model_env = str(cfg.get("model_env") or "").strip()
    if model_env:
        resolved = os.environ.get(model_env, "").strip()
        if resolved:
            return resolved
        raise ValueError(f"model provider env var is empty: {model_env}")
    raise ValueError("model provider requires `model` or `model_env`")


def _resolve_api_key(cfg: dict[str, Any]) -> str:
    api_key = str(cfg.get("api_key") or "").strip()
    if api_key:
        return api_key
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    if api_key_env:
        resolved = os.environ.get(api_key_env, "").strip()
        if resolved:
            return resolved
        raise ValueError(f"model provider env var is empty: {api_key_env}")
    raise ValueError("model provider requires `api_key` or `api_key_env`")


def _resolve_configured_api_base(cfg: dict[str, Any]) -> str | None:
    base = str(cfg.get("api_base") or cfg.get("base_url") or "").strip()
    if base:
        return base
    for env_name in ("api_base_env", "base_url_env"):
        env_key = str(cfg.get(env_name) or "").strip()
        if env_key:
            resolved = os.environ.get(env_key, "").strip()
            if resolved:
                return resolved
            raise ValueError(f"model provider env var is empty: {env_key}")
    return None
