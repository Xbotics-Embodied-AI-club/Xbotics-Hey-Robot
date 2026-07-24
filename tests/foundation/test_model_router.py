from __future__ import annotations

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.clients import ModelServiceRegistry, RegistryModelRouter


async def test_registry_model_router_adapts_existing_model_clients() -> None:
    config = DeploymentConfig.from_dict(
        {
            "model_services": {
                "vla": {
                    "type": "mock",
                    "enabled": True,
                    "robot_id": "mock0",
                    "provides": ["manipulate"],
                    "timeout_sec": 123,
                    "settings": {
                        "summary": "policy result",
                        "result_metrics": {"action": "open"},
                    },
                }
            }
        }
    )
    registry = ModelServiceRegistry(config)
    router = RegistryModelRouter(registry)

    result = await router.infer(
        "manipulate",
        {"task_prompt": "open drawer"},
        run_id="run-1",
        robot_id="mock0",
    )
    await router.cancel("run-1")

    client = registry.clients["vla"]
    assert result.success is True
    assert result.summary == "policy result"
    assert result.data == {"action": "open"}
    assert client.executed[0].intent.name == "manipulate"  # type: ignore[attr-defined]
    assert client.executed[0].timeout_sec == 123  # type: ignore[attr-defined]
    assert client.cancelled == ["run-1"]  # type: ignore[attr-defined]


async def test_registry_model_router_reports_missing_capability() -> None:
    router = RegistryModelRouter(ModelServiceRegistry(DeploymentConfig.from_dict({})))

    result = await router.infer("missing", {}, run_id="run-1", robot_id="mock0")

    assert result.success is False
    assert result.failure_mode == "model_service_unavailable"
