from __future__ import annotations

import pytest

from hey_robot.config import AgentSpec, DeploymentConfig
from hey_robot.model import ModelClient, create_model_client


def _deployment(model_config: dict) -> DeploymentConfig:
    return DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={"models": {"planner": model_config}},
            )
        }
    )


def test_create_model_client_resolves_explicit_env_names(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODEL", "model-1")
    monkeypatch.setenv("TEST_KEY", "key-1")
    monkeypatch.setenv("TEST_BASE_URL", "https://example.invalid/v1")

    client = create_model_client(
        _deployment(
            {
                "model_env": "TEST_MODEL",
                "api_key_env": "TEST_KEY",
                "base_url_env": "TEST_BASE_URL",
                "temperature": 0.0,
                "max_tokens": 512,
            }
        ),
        "main",
        purpose="agent",
    )

    assert isinstance(client, ModelClient)
    assert client.model == "model-1"
    assert client.base_url == "https://example.invalid/v1"
    assert client.temperature == 0.0
    assert client.max_tokens == 512


def test_model_config_does_not_fall_back_to_global_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "global-key")

    with pytest.raises(ValueError, match="api_key"):
        create_model_client(_deployment({"model": "model-1"}), "main")


def test_missing_or_empty_config_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EMPTY_MODEL", "")

    with pytest.raises(ValueError, match="env var is empty"):
        create_model_client(
            _deployment(
                {
                    "model_env": "EMPTY_MODEL",
                    "api_key": "key",
                }
            ),
            "main",
        )


@pytest.mark.parametrize("legacy_name", ["type", "provider", "api_base"])
def test_legacy_provider_options_fail_explicitly(legacy_name: str) -> None:
    config = {"model": "model-1", "api_key": "key", legacy_name: "legacy"}

    with pytest.raises(ValueError, match="removed model config options"):
        create_model_client(_deployment(config), "main")
