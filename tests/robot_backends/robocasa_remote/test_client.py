from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hey_robot.robot_backends.robocasa_remote.client import GrpcRoboCasaRuntimeClient


def _observation() -> SimpleNamespace:
    return SimpleNamespace(
        trial_id="trial",
        frame_id=4,
        state=[1, 2],
        images=[],
        task="OpenDrawer",
        done=False,
        metadata=None,
    )


class _Stub:
    async def get_health(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            online=True, loaded=True, busy=False, error_message="", metrics=None
        )

    async def begin_trial(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return _observation()

    async def observe(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return _observation()

    async def step(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            observation=_observation(),
            done=True,
            status="completed",
            actions_executed=3,
            chunks_executed=1,
            progress=None,
            diagnostics=None,
            error_message="",
        )

    async def step_native(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(observation=_observation(), done=False, progress=None)

    async def read_truth(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            done=True, official_success=True, frame_id=4, metrics=None
        )

    async def end_trial(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(ended=True)


_Stub.GetHealth = _Stub.get_health
_Stub.BeginTrial = _Stub.begin_trial
_Stub.Observe = _Stub.observe
_Stub.Step = _Stub.step
_Stub.StepNative = _Stub.step_native
_Stub.ReadTruth = _Stub.read_truth
_Stub.EndTrial = _Stub.end_trial


async def test_remote_client_projects_all_runtime_operations_without_transport_logic() -> (
    None
):
    client = GrpcRoboCasaRuntimeClient("grpc://runtime:9092", token="secret")  # noqa: S106
    client._stub = _Stub()

    assert await client.health() == {
        "online": True,
        "loaded": True,
        "busy": False,
        "error": None,
        "metrics": {},
    }
    begun = await client.begin_trial(trial_id="trial", task="OpenDrawer", seed=1)
    observed = await client.observe()
    option = await client.run_option(session_id="s", instruction="open", max_actions=8)
    native = await client.step_native(action=[0.0] * 12, expected_frame_id=4)

    assert begun.frame_id == observed.frame_id == 4
    assert option.done
    assert option.actions_executed == 3
    assert native.status == "native"
    assert await client.read_truth() == {
        "done": True,
        "official_success": True,
        "frame_id": 4,
        "metrics": {},
    }
    assert await client.end_trial()
    assert client._metadata() == (("authorization", "Bearer secret"),)


def test_remote_client_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GrpcRoboCasaRuntimeClient("grpc://")
