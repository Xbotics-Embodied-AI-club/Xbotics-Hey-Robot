from __future__ import annotations

import pytest

from hey_robot.config import AgentSpec, DeploymentConfig
from hey_robot.providers import (
    OpenAICompatReasoningProvider,
    ReasoningMessage,
    ReasoningResponse,
    ReasoningToolCall,
    build_provider,
)
from hey_robot.providers.openai_compat_provider import (
    _to_openai_message,
    _validate_required_tool_call,
)


def test_build_provider_resolves_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("UNIT_TEST_MODEL", "env-model-123")
    monkeypatch.setenv("UNIT_TEST_API_KEY", "test-key")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model_env": "UNIT_TEST_MODEL",
                            "api_key_env": "UNIT_TEST_API_KEY",
                        }
                    }
                },
            )
        }
    )

    provider = build_provider(config, "main", purpose="planner")

    assert isinstance(provider, OpenAICompatReasoningProvider)
    assert provider.get_default_model() == "env-model-123"


def test_required_tool_choice_rejects_text_only_provider_response() -> None:
    response = _validate_required_tool_call(
        ReasoningResponse(
            content="```tool_call request_capability({})```", finish_reason="stop"
        ),
        "required",
        [{"type": "function", "function": {"name": "request_capability"}}],
    )

    assert response.finish_reason == "error"
    assert response.error_kind == "tool_protocol"


def test_explicit_empty_provider_key_does_not_fall_back_to_openai_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    provider = OpenAICompatReasoningProvider(
        model="deepseek-chat", api_key="", api_base="https://api.deepseek.com"
    )

    with pytest.raises(ValueError, match="api_key"):
        provider._client_or_create()


def test_build_provider_resolves_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("UNIT_TEST_MODEL", "env-model-123")
    monkeypatch.setenv("UNIT_TEST_API_KEY", "test-key")
    monkeypatch.setenv("UNIT_TEST_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model_env": "UNIT_TEST_MODEL",
                            "api_key_env": "UNIT_TEST_API_KEY",
                            "base_url_env": "UNIT_TEST_BASE_URL",
                        }
                    }
                },
            )
        }
    )

    provider = build_provider(config, "main", purpose="planner")

    assert isinstance(provider, OpenAICompatReasoningProvider)
    assert provider.api_base == "https://example.invalid/v1"


def test_build_provider_does_not_use_global_model_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "global-model")
    monkeypatch.setenv("OPENAI_API_KEY", "global-key")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={"providers": {"planner": {"type": "openai_compat"}}},
            )
        }
    )

    with pytest.raises(ValueError, match="model"):
        build_provider(config, "main", purpose="planner")


def test_build_provider_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "global-key")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model": "unit-test-model",
                        }
                    }
                },
            )
        }
    )

    with pytest.raises(ValueError, match="api_key"):
        build_provider(config, "main", purpose="planner")


def test_deepseek_provider_disables_required_tool_choice(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "deepseek",
                            "model_env": "DEEPSEEK_MODEL",
                            "api_key_env": "DEEPSEEK_API_KEY",
                            "base_url_env": "DEEPSEEK_BASE_URL",
                        }
                    }
                },
            )
        }
    )

    provider = build_provider(config, "main", purpose="planner")

    assert isinstance(provider, OpenAICompatReasoningProvider)
    assert provider.supports_required_tool_choice is False


def test_assistant_history_preserves_tool_calls() -> None:
    message = ReasoningMessage(
        role="assistant",
        content="",
        tool_calls=[
            ReasoningToolCall(
                id="call_1",
                name="request_capability",
                arguments={
                    "capability": "camera_inspect",
                    "objective": "capture front view",
                    "interrupt": False,
                },
            )
        ],
    )

    payload = _to_openai_message(message)

    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["id"] == "call_1"
    assert payload["tool_calls"][0]["function"]["name"] == "request_capability"


def test_build_provider_deterministic_feedback_purpose() -> None:
    from hey_robot.providers.static_feedback import (
        DeterministicExecutionFeedbackReasoner,
    )

    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={"providers": {"feedback": {"type": "deterministic"}}},
            )
        }
    )
    provider = build_provider(config, "main", purpose="feedback")
    assert isinstance(provider, DeterministicExecutionFeedbackReasoner)


def test_build_provider_deterministic_planner_rejected() -> None:
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={"providers": {"planner": {"type": "deterministic"}}},
            )
        }
    )

    with pytest.raises(ValueError, match="deterministic planner has been removed"):
        build_provider(config, "main", purpose="planner")


def test_build_provider_unsupported_type_raises() -> None:
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={"providers": {"planner": {"type": "unknown_provider_x"}}},
            )
        }
    )
    with pytest.raises(ValueError, match="unsupported provider"):
        build_provider(config, "main", purpose="planner")


def test_resolve_model_env_var_empty_raises(monkeypatch) -> None:
    monkeypatch.setenv("EMPTY_MODEL", "")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model_env": "EMPTY_MODEL",
                            "api_key": "test-key",
                        }
                    }
                },
            )
        }
    )
    with pytest.raises(ValueError, match="model provider env var is empty"):
        build_provider(config, "main", purpose="planner")


def test_resolve_api_key_env_var_empty_raises(monkeypatch) -> None:
    monkeypatch.setenv("EMPTY_KEY", "")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model": "test-model",
                            "api_key_env": "EMPTY_KEY",
                        }
                    }
                },
            )
        }
    )
    with pytest.raises(ValueError, match="model provider env var is empty"):
        build_provider(config, "main", purpose="planner")


def test_resolve_api_base_env_var_empty_raises(monkeypatch) -> None:
    monkeypatch.setenv("UNIT_TEST_MODEL", "test-model")
    monkeypatch.setenv("UNIT_TEST_API_KEY", "test-key")
    monkeypatch.setenv("EMPTY_BASE", "")
    config = DeploymentConfig(
        agents={
            "main": AgentSpec(
                type="robot_agent",
                settings={
                    "providers": {
                        "planner": {
                            "type": "openai_compat",
                            "model_env": "UNIT_TEST_MODEL",
                            "api_key_env": "UNIT_TEST_API_KEY",
                            "base_url_env": "EMPTY_BASE",
                        }
                    }
                },
            )
        }
    )
    with pytest.raises(ValueError, match="model provider env var is empty"):
        build_provider(config, "main", purpose="planner")
