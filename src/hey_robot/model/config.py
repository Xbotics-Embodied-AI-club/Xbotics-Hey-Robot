from __future__ import annotations

import os
from typing import Any

from hey_robot.config import DeploymentConfig
from hey_robot.model.client import ModelClient


def create_model_client(
    config: DeploymentConfig, agent_id: str, *, purpose: str = "planner"
) -> ModelClient:
    settings = config.agents[agent_id].settings
    models = settings.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"missing models config for purpose={purpose}")
    key = "planner" if purpose == "agent" else purpose
    model_config = models.get(key)
    if not isinstance(model_config, dict):
        raise ValueError(f"missing model config for purpose={purpose}")
    _reject_legacy_options(model_config)
    return ModelClient(
        model=_value_or_env(model_config, "model"),
        api_key=_value_or_env(model_config, "api_key"),
        base_url=_optional_value_or_env(model_config, "base_url"),
        temperature=float(model_config.get("temperature", 0.1)),
        max_tokens=int(model_config.get("max_tokens", 2048)),
        reasoning_effort=model_config.get("reasoning_effort"),
        extra_headers=dict(model_config.get("extra_headers", {}) or {}) or None,
        extra_body=dict(model_config.get("extra_body", {}) or {}) or None,
    )


def _value_or_env(config: dict[str, Any], name: str) -> str:
    value = str(config.get(name) or "").strip()
    if value:
        return value
    env_name = str(config.get(f"{name}_env") or "").strip()
    if not env_name:
        raise ValueError(f"model config requires `{name}` or `{name}_env`")
    resolved = os.environ.get(env_name, "").strip()
    if not resolved:
        raise ValueError(f"model config env var is empty: {env_name}")
    return resolved


def _optional_value_or_env(config: dict[str, Any], name: str) -> str | None:
    value = str(config.get(name) or "").strip()
    if value:
        return value
    env_name = str(config.get(f"{name}_env") or "").strip()
    if not env_name:
        return None
    resolved = os.environ.get(env_name, "").strip()
    if not resolved:
        raise ValueError(f"model config env var is empty: {env_name}")
    return resolved


def _reject_legacy_options(config: dict[str, Any]) -> None:
    removed = {
        "type",
        "provider",
        "api_base",
        "api_base_env",
        "use_responses_api",
        "strict_tools",
        "fallback_models",
    }.intersection(config)
    if removed:
        names = ", ".join(sorted(removed))
        raise ValueError(f"removed model config options: {names}")
