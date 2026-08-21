from __future__ import annotations

from typing import Any

from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.skills import SkillContext
from hey_robot.skills.vla.option import (
    VLAOptionRequest,
    VLAOptionResult,
    VLAOptionRunner,
    _actions_from_model_data,
    _coalesce_native_action_chunk,
    _execution_diagnostics,
    _normalize_action,
    _observation_payload,
    _with_diagnostics,
)


def test_model_action_normalization_supports_chunks_primitives_and_native_values() -> (
    None
):
    chunk = {
        "action_chunk": {
            "actions": [
                {
                    "name": "embodiment_native_action",
                    "arguments": {"values": [0.0] * 12},
                },
                {"action": "look", "speed": 1},
                "invalid",
            ]
        }
    }

    assert [action["name"] for action in _actions_from_model_data(chunk)] == [
        "embodiment_native_action",
        "look",
    ]
    assert _actions_from_model_data({"primitive": {"action": "open", "amount": 2}}) == [
        {"name": "open", "arguments": {"amount": 2}}
    ]
    assert _actions_from_model_data({"values": [1.0] * 12}) == [
        {"name": "embodiment_native_action", "arguments": {"values": [1.0] * 12}}
    ]
    assert _actions_from_model_data({}) == []
    assert _normalize_action("bad") is None


def test_native_chunk_coalescing_preserves_non_native_actions() -> None:
    native = [
        {
            "name": "embodiment_native_action",
            "arguments": {"values": [float(index)] * 12},
        }
        for index in range(2)
    ]

    coalesced = _coalesce_native_action_chunk(native)

    assert coalesced[0][1] == 2
    assert coalesced[0][0]["arguments"]["values"] == [[0.0] * 12, [1.0] * 12]
    assert _coalesce_native_action_chunk([{"name": "look", "arguments": {}}])[0][1] == 1
    assert (
        len(_coalesce_native_action_chunk([*native, {"name": "look", "arguments": {}}]))
        == 3
    )
    assert (
        len(
            _coalesce_native_action_chunk(
                [{"name": "embodiment_native_action", "arguments": {"values": [0]}}] * 2
            )
        )
        == 2
    )


def test_option_evidence_helpers_keep_observation_and_diagnostics_structured() -> None:
    observation = RobotObservation(
        Envelope(timestamp=3.0),
        7,
        images=[ImageRef("media://frame", camera="front")],
        proprioception=[0.0] * 16,
        raw={"execution_diagnostics": {"reward": 1}},
    )
    payload = _observation_payload(observation)
    base = VLAOptionResult(
        True, "done", True, True, "environment_done", 1, 7, ({"a": 1},), ()
    )
    enriched = _with_diagnostics(base, {"before": 1}, {"after": 2})

    assert payload["frame_id"] == 7
    assert payload["images"][0]["uri"] == "media://frame"
    assert _execution_diagnostics(observation) == {"reward": 1}
    assert _execution_diagnostics(None) == {}
    assert enriched.model_outputs[-1]["harness_diagnostics"] == {
        "before": {"before": 1},
        "after": {"after": 2},
    }
    skill = enriched.to_skill_result()
    assert skill.data["subgoal_status"] == "achieved"
    assert skill.data["steps_used"] == 0


class _ObservingRobot:
    async def observe(self, robot_id: str, **_kwargs):
        return RobotObservation(
            Envelope(robot_id=robot_id), 1, proprioception=[0.0] * 16
        )


class _FailedModels:
    async def infer(self, *_args: Any, **_kwargs: Any) -> Any:
        from hey_robot.foundation.clients.models import ModelInferenceResult

        return ModelInferenceResult(
            False, "policy offline", failure_mode="unavailable", error="offline"
        )


async def test_option_runner_returns_structured_failures_before_any_physical_action() -> (
    None
):
    request = VLAOptionRequest("open drawer", 2)
    runner = VLAOptionRunner()

    no_model = await runner.run(SkillContext("run", "task", "robot"), request)
    no_robot = await runner.run(
        SkillContext("run", "task", "robot", models=_FailedModels()),
        request,  # type: ignore[arg-type]
    )
    failed_model = await runner.run(
        SkillContext(
            "run", "task", "robot", robot=_ObservingRobot(), models=_FailedModels()
        ),  # type: ignore[arg-type]
        request,
    )

    assert no_model.failure_mode == "model_service_unavailable"
    assert no_robot.failure_mode == "robot_client_unavailable"
    assert failed_model.failure_mode == "unavailable"
    assert failed_model.error == "offline"
