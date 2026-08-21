from __future__ import annotations

from typing import Any

import pytest

from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.clients import local
from hey_robot.foundation.clients.models import ServiceInvocationRequest
from hey_robot.protocol import Envelope, SkillIntent


class _Executor:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.cancelled = False

    def health(self) -> dict[str, Any]:
        return {"online": True, "loaded": True, "metrics": {"device": "cpu"}}

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {"success": True, "summary": "acted", "metrics": {"steps": 2}}

    def cancel(self) -> None:
        self.cancelled = True


def _spec(runtime: str = "rldx") -> ModelServiceSpec:
    return ModelServiceSpec("robot_policy", "robocasa", settings={"runtime": runtime})


def _request() -> ServiceInvocationRequest:
    intent = SkillIntent(
        Envelope(episode_id="episode"),
        "skill",
        "task",
        "skill",
        "manipulate",
        {"x": 1},
        "goal",
    )
    return ServiceInvocationRequest("policy", intent, 2.0, {"task_prompt": "goal"})


async def test_local_client_uses_an_isolated_executor_without_network_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor()
    monkeypatch.setattr(local, "_build_executor", lambda *_: executor)
    client = local.LocalFoundationClient("policy", _spec())

    health = await client.health()
    result = await client.execute(_request())
    await client.cancel("skill")

    assert health.online
    assert health.loaded
    assert health.robot_id == "robocasa"
    assert result.success
    assert result.metrics == {"steps": 2}
    assert executor.payload == {
        "skill_name": "manipulate",
        "episode_id": "episode",
        "objective": "goal",
        "arguments": {"task_prompt": "goal"},
    }
    assert executor.cancelled


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            ModelServiceSpec("robot_policy", "r", settings={"runtime": "unknown"}),
            "unsupported local robot policy",
        ),
        (ModelServiceSpec("unknown", "r"), "unsupported local foundation"),
    ],
)
def test_local_executor_factory_rejects_unknown_backend_types(
    spec: ModelServiceSpec, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        local._build_executor("policy", spec)
