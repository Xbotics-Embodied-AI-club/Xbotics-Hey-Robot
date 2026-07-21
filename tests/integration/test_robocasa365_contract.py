from __future__ import annotations

import asyncio
import json

import numpy as np

from evaluation.robocasa365.full_system_benchmark import _write_worker_event_artifacts
from evaluation.robocasa365.worker.episode_manager import EpisodeManager
from evaluation.robocasa365.worker.vla_option_executor import (
    OptionRequest,
    VLAOptionExecutor,
    _clip_action_to_space,
    _policy_task_prompt,
    _PolicyBundle,
)
from hey_robot.foundation.clients.models import ServiceInvocationResult
from hey_robot.skill_os import SkillRuntime, load_skill_registry
from hey_robot.skill_os.context import SkillContext


class _AuthContext:
    def __init__(self, token: str) -> None:
        self.token = token
        self.aborted: tuple[object, str] | None = None

    def invocation_metadata(self):
        return (("authorization", self.token),)

    async def abort(self, code, detail) -> None:
        self.aborted = (code, detail)


def _observation() -> dict[str, object]:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "agent_pos": np.zeros((16,), dtype=np.float32),
        "pixels": {"camera1": frame, "camera2": frame, "camera3": frame},
    }


class _ActionSpace:
    low = np.full((12,), -1.0, dtype=np.float32)
    high = np.full((12,), 1.0, dtype=np.float32)

    def contains(self, action) -> bool:
        value = np.asarray(action)
        return (
            value.shape == (12,)
            and bool(np.all(value >= self.low))
            and bool(np.all(value <= self.high))
        )


class _Env:
    action_space = _ActionSpace()

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def step(self, action):
        assert len(action) == 12
        self.steps += 1
        return _observation(), 0.0, False, False, {"is_success": False}

    def close(self) -> None:
        self.closed = True


def test_episode_manager_is_the_only_environment_owner() -> None:
    env = _Env()
    manager = EpisodeManager(
        allowed_tasks=frozenset({"CloseFridge"}),
        env_factory=lambda _spec: (env, _observation()),
    )
    trial = manager.begin_trial(manager.new_spec(task="CloseFridge", seed=1000))

    outcome = manager.step(np.zeros((12,), dtype=np.float32), expected_frame_id=0)

    assert outcome.frame_id == 1
    assert manager.observe() is trial
    assert trial.frame_id == 1
    assert manager.end_trial() is True
    assert manager.end_trial() is False
    assert env.closed is True


def test_policy_action_is_clipped_at_the_vla_adapter_boundary() -> None:
    raw = np.zeros((12,), dtype=np.float32)
    raw[3] = 1.25
    raw[6] = -1.5

    action, clipped = _clip_action_to_space(raw, _ActionSpace())

    assert clipped is True
    assert action[3] == 1.0
    assert action[6] == -1.0
    assert raw[3] == 1.25


def test_worker_ledger_writes_one_canonical_option_record(tmp_path) -> None:
    truth = {
        "metrics": {
            "events": [
                {"kind": "model_service_option", "skill_id": "option-1"},
                {"kind": "action", "frame_id": 1},
            ]
        }
    }

    _write_worker_event_artifacts(tmp_path, truth)

    options = [
        json.loads(line)
        for line in (tmp_path / "options.jsonl").read_text().splitlines()
    ]
    assert options == [{"kind": "model_service_option", "skill_id": "option-1"}]


def test_pi052_uses_environment_root_task_instead_of_agent_option_label() -> None:
    env = _Env()
    env.task_description = "Close the fridge door."
    trial = type("Trial", (), {"env": env})()

    prompt = _policy_task_prompt(trial, "PushFridgeDoor")

    assert prompt == "Close the fridge door."


def test_option_executor_advances_the_shared_episode_without_exposing_truth() -> None:
    class Policy:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def select_action(self, _sample):
            return np.zeros((1, 12), dtype=np.float32)

    manager = EpisodeManager(
        allowed_tasks=frozenset({"CloseFridge"}),
        env_factory=lambda _spec: (_Env(), _observation()),
    )
    manager.begin_trial(manager.new_spec(task="CloseFridge", seed=1000))
    policy = Policy()
    runner = VLAOptionExecutor(
        manager=manager,
        environ={"ROBOCASA_OPTION_HORIZON": "2"},
        policy_loader=lambda path, device: _PolicyBundle(
            policy_path=path,
            policy_type="fake",
            device=device,
            input_features={},
            policy=policy,
            preprocessor=lambda sample: sample,
            postprocessor=lambda action: action,
        ),
    )
    runner.prepare()

    result = runner.run(
        OptionRequest(
            skill_id="option-1",
            option_command="Close the fridge.",
        )
    )

    assert result.success is True
    assert policy.reset_count == 1
    assert manager.current_trial().frame_id == 2
    assert result.metrics["before_frame_id"] == 0
    assert result.metrics["after_frame_id"] == 2
    assert "root_task_success" not in result.metrics
    assert "episode_success" not in result.metrics

    second = runner.run(
        OptionRequest(skill_id="option-2", option_command="Keep closing it.")
    )
    assert second.success is True
    assert policy.reset_count == 1
    assert manager.current_trial().frame_id == 4


def test_agent_visible_skill_accepts_only_option_command() -> None:
    class Services:
        def __init__(self) -> None:
            self.calls = []

        async def call(self, name, arguments):
            self.calls.append((name, arguments))
            return ServiceInvocationResult(
                success=True,
                status="completed",
                summary="bounded option ended",
                metrics={
                    "option_state": "boundary_reached",
                    "requires_reobservation": True,
                },
            )

    async def run_once() -> None:
        services = Services()
        registry = load_skill_registry(enabled=("robocasa_option",))
        skill = registry.get("robocasa_option").spec
        assert set(skill.input_schema["properties"]) == {"option_command"}
        assert skill.input_schema["additionalProperties"] is False
        runtime = SkillRuntime(registry)
        result = await runtime.execute(
            "robocasa_option",
            {"option_command": "Close the fridge."},
            context_factory=lambda invoke: SkillContext(
                model_services=services, invoke=invoke
            ),
        )
        assert result.success is True
        assert services.calls == [
            ("robocasa_option", {"option_command": "Close the fridge."})
        ]

    asyncio.run(run_once())


def test_rollout_skill_is_removed() -> None:
    registry = load_skill_registry()
    assert "robocasa_rollout" not in registry.names()


def test_runtime_roles_require_distinct_credentials() -> None:
    from evaluation.robocasa365.worker.runtime_server import RoboCasaRuntimeService

    async def run_once() -> None:
        evaluator_token = "evaluator" + "-credential"
        data_token = "data" + "-credential"
        runtime = RoboCasaRuntimeService(
            evaluator_token=evaluator_token, data_token=data_token
        )
        evaluator = _AuthContext(f"Bearer {evaluator_token}")
        data = _AuthContext(f"Bearer {data_token}")
        await runtime._authorize(evaluator, role="evaluator")
        await runtime._authorize(data, role="data")
        await runtime._authorize(data, role="evaluator")
        assert data.aborted is not None

    asyncio.run(run_once())
