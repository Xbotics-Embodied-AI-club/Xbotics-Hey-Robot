"""Administrative smoke test for the frame-level RoboCasa Runtime service."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import grpc
from hey_robot.model_service.v1 import model_service_pb2, model_service_pb2_grpc

from hey_robot.robocasa_runtime.v1 import (
    robocasa_runtime_pb2,
    robocasa_runtime_pb2_grpc,
)


async def run(target: str, task: str, seed: int) -> dict[str, Any]:
    channel = grpc.aio.insecure_channel(target.removeprefix("grpc://"))
    runtime = robocasa_runtime_pb2_grpc.RoboCasaRuntimeStub(channel)
    models = model_service_pb2_grpc.ModelServiceStub(channel)
    episode_id = ""
    try:
        health_before = await runtime.GetHealth(
            robocasa_runtime_pb2.HealthRequest(), timeout=30
        )
        created = await runtime.CreateEpisode(
            robocasa_runtime_pb2.CreateEpisodeRequest(task=task, seed=seed),
            timeout=180,
        )
        observation = created.observation
        episode_id = observation.episode_id
        blocked = await models.ExecuteSkill(
            model_service_pb2.ExecuteSkillRequest(
                skill_id="runtime-smoke-exclusion",
                skill_name="robocasa_rollout",
            ),
            timeout=10,
        )
        observed = await runtime.Observe(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id), timeout=30
        )
        stepped = await runtime.Step(
            robocasa_runtime_pb2.StepRequest(
                episode_id=episode_id,
                action=[0.0] * 12,
                expected_frame_id=observed.frame_id,
            ),
            timeout=30,
        )
        reset = await runtime.Reset(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id), timeout=180
        )
        closed = await runtime.CloseEpisode(
            robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id), timeout=30
        )
        episode_id = ""
        health_after = await runtime.GetHealth(
            robocasa_runtime_pb2.HealthRequest(), timeout=30
        )
        return {
            "health_before": _health(health_before),
            "create": _observation(observation),
            "task_rollout_exclusion": {
                "success": blocked.success,
                "failure_mode": blocked.failure_mode,
            },
            "observe_frame_id": observed.frame_id,
            "step": {
                "frame_id": stepped.observation.frame_id,
                "reward": stepped.reward,
                "done": stepped.done,
                "success": stepped.success,
            },
            "reset": {
                "frame_id": reset.frame_id,
                "done": reset.done,
                "success": reset.success,
            },
            "closed": closed.closed,
            "health_after": _health(health_after),
        }
    finally:
        if episode_id:
            await runtime.CloseEpisode(
                robocasa_runtime_pb2.EpisodeRequest(episode_id=episode_id), timeout=30
            )
        await channel.close()


def _health(response) -> dict[str, Any]:
    return {
        "online": response.online,
        "loaded": response.loaded,
        "busy": response.busy,
        "error": response.error_message or None,
    }


def _observation(response) -> dict[str, Any]:
    return {
        "episode_id": response.episode_id,
        "frame_id": response.frame_id,
        "state_dimensions": len(response.state),
        "cameras": [image.camera for image in response.images],
        "jpeg_bytes": [len(image.data) for image in response.images],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:9092")
    parser.add_argument("--task", default="CloseFridge")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(
            asyncio.run(run(args.target, args.task, args.seed)),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
