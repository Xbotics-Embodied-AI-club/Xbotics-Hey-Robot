from __future__ import annotations

import asyncio
import base64
import io
import json

import numpy as np
from PIL import Image

from evaluation.robocasa365.conditions import condition_for
from evaluation.robocasa365.full_system_benchmark import (
    _find_trial_task,
    _option_records,
    _parser as trial_parser,
    _write_evaluator_action_artifact,
)
from hey_robot.config import ModelServiceSpec
from hey_robot.foundation.backends.lerobot import LeRobotPolicyExecutor
from hey_robot.foundation.clients.models import ModelInferenceResult
from hey_robot.protocol import Envelope, ImageRef, RobotObservation
from hey_robot.robocasa_backend.episode_manager import EpisodeManager
from hey_robot.robot_api import RobotActionResult
from hey_robot.skills import (
    SkillCommand,
    SkillContext,
    SkillRunner,
    load_skill_registry,
)
from hey_robot.skills.resources import ResourceManager


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


def test_evaluator_ledger_writes_actions_without_overwriting_options(tmp_path) -> None:
    truth = {
        "metrics": {
            "events": [
                {"kind": "model_service_option", "skill_id": "option-1"},
                {"kind": "action", "frame_id": 1},
            ]
        }
    }

    _write_evaluator_action_artifact(tmp_path, truth)

    actions = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text().splitlines()
    ]
    assert actions == [{"frame_id": 1, "kind": "action"}]
    assert not (tmp_path / "options.jsonl").exists()
    assert not (tmp_path / "model_service_events.jsonl").exists()


def test_option_records_come_from_skill_os_manipulate_events() -> None:
    snapshots = [
        {
            "skills": [
                {
                    "name": "manipulate",
                    "timeline": [{"envelope": {"chat_id": "trial-1"}}],
                }
            ]
        }
    ]
    assert len(_option_records(snapshots, trial_id="trial-1")) == 1


def test_option_records_accept_current_gateway_run_payload() -> None:
    snapshots = [
        {
            "skills": [
                {
                    "name": "manipulate",
                    "skill": "manipulate",
                    "envelope": {"chat_id": "trial-1"},
                    "phase": "completed",
                },
                {
                    "name": "inspect_scene",
                    "skill": "inspect_scene",
                    "envelope": {"chat_id": "trial-1"},
                    "phase": "completed",
                },
            ]
        }
    ]

    assert _option_records(snapshots, trial_id="trial-1") == [snapshots[0]["skills"][0]]


def test_flat_condition_has_an_executable_single_option_limit() -> None:
    assert condition_for("b0").manipulate_call_limit == 1
    assert condition_for("b1").manipulate_call_limit is None


def test_trial_defaults_to_live_environment_objective() -> None:
    args = trial_parser().parse_args(
        [
            "--task",
            "KettleBoiling",
            "--output-dir",
            "runtime/test-trial",
        ]
    )

    assert args.objective is None


def test_agent_task_is_correlated_by_the_submitted_condition_prompt() -> None:
    objective = condition_for("b2").prompt("Close the fridge.")
    task = {"objective": objective, "created_at": 11.0, "status": "active"}
    assert _find_trial_task([task], objective=objective, started=10.0) is task


def _encoded_observation(frame_id: int = 0) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "frame_id": frame_id,
        "proprioception": [0.0] * 16,
        "images": [
            {"camera": camera, "data": encoded}
            for camera in ("camera1", "camera2", "camera3")
        ],
    }


def test_policy_executor_returns_one_action_without_owning_environment() -> None:
    class Runtime:
        policy_type = "fake"

        def __init__(self) -> None:
            self.reset_count = 0
            self.tasks = []

        def reset(self, seed=None) -> None:
            del seed
            self.reset_count += 1

        def select_action(self, _observation, task):
            self.tasks.append(task)
            action = np.zeros((12,), dtype=np.float32)
            action[3] = 1.25
            return action

        def cancel(self):
            pass

        def close(self):
            pass

    runtime = Runtime()
    runner = LeRobotPolicyExecutor(
        "robocasa365",
        _robocasa_policy_spec(),
        runtime_loader=lambda *_args: runtime,
    )
    request = {
        "skill_name": "manipulate",
        "episode_id": "trial-1",
        "arguments": {
            "task_prompt": "Close the fridge.",
            "seed": 1000,
            "observation": {
                **_encoded_observation(),
                "raw": {"policy_task": "Close the fridge door."},
            },
        },
    }
    result = runner.execute(request)

    assert result["success"] is True
    assert runtime.reset_count == 1
    action = result["metrics"]["policy_result"]["actions"][0]["arguments"]
    assert action["values"][3] == 1.0
    assert action["raw_values"][3] == 1.25
    assert result["metrics"]["action_clipped"] is True
    assert result["metrics"]["policy_result"]["raw"]["expected_frame_id"] == 0
    assert runtime.tasks == ["Close the fridge door."]

    second = runner.execute(request)
    assert second["success"] is True
    assert runtime.reset_count == 1


def test_agent_subgoal_change_resets_policy_action_queue() -> None:
    class Runtime:
        policy_type = "fake"

        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self, seed=None) -> None:
            del seed
            self.reset_count += 1

        def select_action(self, _observation, _task):
            return np.zeros((12,), dtype=np.float32)

        def cancel(self):
            pass

        def close(self):
            pass

    runtime = Runtime()
    runner = LeRobotPolicyExecutor(
        "robocasa365",
        _robocasa_policy_spec(prompt_mode="agent_subgoal"),
        runtime_loader=lambda *_args: runtime,
    )
    request = {
        "skill_name": "manipulate",
        "episode_id": "trial-1",
        "arguments": {
            "task_prompt": "Pick up the kettle.",
            "observation": _encoded_observation(),
        },
    }

    assert runner.execute(request)["success"] is True
    assert runner.execute(request)["success"] is True
    request["arguments"]["task_prompt"] = "Place the kettle on the burner."
    assert runner.execute(request)["success"] is True

    assert runtime.reset_count == 2


def _robocasa_policy_spec(**settings) -> ModelServiceSpec:
    return ModelServiceSpec(
        type="robot_policy",
        robot_id="robocasa365",
        provides=("manipulate",),
        settings={
            "runtime": "lerobot",
            "policy_path": "fake",
            "policy_device": "cpu",
            "embodiment": "robocasa",
            "action_space": "robocasa_12d",
            "action_dimensions": 12,
            "prompt_mode": "environment_root",
            **settings,
        },
    )


def test_generic_manipulate_executes_native_action_option() -> None:
    class Models:
        def __init__(self) -> None:
            self.calls = []

        async def infer(
            self, capability, request, *, run_id, robot_id, timeout_sec=None
        ):
            del run_id, robot_id, timeout_sec
            self.calls.append((capability, request))
            return ModelInferenceResult(
                success=True,
                summary="one action",
                data={"values": [0.0] * 12},
            )

        async def cancel(self, run_id) -> None:
            del run_id

    class Robot:
        def __init__(self, observations) -> None:
            self.observations = observations
            self.calls = []

        async def observe(self, robot_id, *, after_frame_id=None, timeout_sec=None):
            del after_frame_id, timeout_sec
            del robot_id
            return self.observations[0]

        async def execute(
            self, robot_id, action, arguments, *, run_id, expected_frame_id=None
        ):
            del robot_id, run_id
            self.calls.append((action, arguments, expected_frame_id))
            frame_id = self.observations[0].frame_id + 1
            self.observations[0] = make_observation(frame_id)
            return RobotActionResult(True, "applied", frame_id=frame_id)

        async def capabilities(self, robot_id):
            del robot_id
            raise NotImplementedError

        async def stop(self, robot_id, *, reason):
            del robot_id, reason

    class Events:
        def __init__(self) -> None:
            self.items = []

        async def emit(self, event) -> None:
            self.items.append(event)

    refs = [
        ImageRef(uri=f"media://{camera}", camera=camera)
        for camera in ("camera1", "camera2", "camera3")
    ]

    def make_observation(frame_id):
        return RobotObservation(
            envelope=Envelope(episode_id="trial-1", robot_id="robocasa365"),
            frame_id=frame_id,
            images=refs,
            proprioception=[0.0] * 16,
            task="CloseFridge",
            raw={"trial_id": "trial-1", "policy_task": "Close the fridge door."},
        )

    async def run_once() -> None:
        models = Models()
        observations = [make_observation(0)]
        robot = Robot(observations)
        registry = load_skill_registry(("hey_robot.skills.builtins",))
        skill = registry.get("manipulate")
        assert "task_prompt" in skill.parameters["properties"]
        assert "robot_control" in skill.resources
        runner = SkillRunner(
            registry,
            resources=ResourceManager(),
            events=Events(),
            context_factory=lambda invoke: SkillContext(
                run_id=invoke.run_id,
                task_id=invoke.task_id,
                robot_id=invoke.robot_id,
                robot=robot,
                models=models,
            ),
        )
        result = await runner.execute(
            SkillCommand(
                envelope=Envelope(episode_id="trial-1", robot_id="robocasa365"),
                run_id="trial-1",
                task_id="task-1",
                robot_id="robocasa365",
                name="manipulate",
                arguments={"task_prompt": "Close the fridge.", "max_steps": 50},
            )
        )
        assert result.success is True
        assert len(models.calls) == 50
        assert len(robot.calls) == 50
        assert models.calls[0][0] == "manipulate"
        assert models.calls[0][1]["task_prompt"] == "Close the fridge."
        assert models.calls[0][1]["policy_session_id"] == "trial-1"
        assert robot.calls[0][0] == "embodiment_native_action"
        assert result.data["termination_reason"] == "max_steps"

    asyncio.run(run_once())


def test_rollout_skill_is_removed() -> None:
    registry = load_skill_registry(("hey_robot.skills.builtins",))
    names = {skill.name for skill in registry.list()}
    assert "robocasa_rollout" not in names
    assert "robocasa_option" not in names


def test_runtime_roles_require_distinct_credentials() -> None:
    from hey_robot.robocasa_backend.runtime_server import (
        RoboCasaRuntimeService,
    )

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
