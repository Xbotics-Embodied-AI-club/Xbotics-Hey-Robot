from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.simulation.xlerobot_sim_driver import XLeRobotSimDriver
from hey_robot.vla.so101_schema import (
    SO101_STATE_SCHEMA,
    action_chunk_first_vector,
    action_vector_to_targets,
    state_from_sim_driver,
)


async def evaluate(args: argparse.Namespace) -> None:
    config = DeploymentConfig.from_yaml(args.config)
    manager = RobotManager(config)
    robot_id = args.robot_id or next(iter(config.robots))
    driver = manager.require(robot_id)
    if not isinstance(driver, XLeRobotSimDriver):
        raise SystemExit(
            f"evaluation currently requires xlerobot_sim driver, got {type(driver).__name__}"
        )

    out = Path(args.out)
    await asyncio.to_thread(out.mkdir, parents=True, exist_ok=True)
    camera_names = [
        item.strip() for item in str(args.cameras).split(",") if item.strip()
    ]
    image_size = _parse_image_size(args.image_size)
    traces: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []

    await driver.start()
    try:
        for episode in range(int(args.episodes)):
            await driver.reset()
            episode_trace: list[dict[str, Any]] = []
            success = False
            failure_mode = "timeout"
            started = time.perf_counter()
            for step in range(int(args.max_steps)):
                observation = _build_endpoint_observation(
                    driver,
                    camera_names=camera_names,
                    image_size=image_size,
                    arm=str(args.arm),
                )
                request_payload = {
                    "policy_session_id": f"eval-{episode}",
                    "skill_name": "manipulate",
                    "atomic_command": str(args.task),
                    "task": str(args.task),
                    "observation": observation,
                    "frame_id": observation["frame_id"],
                    "metadata": {
                        "robot_id": robot_id,
                        "embodiment": "xlerobot",
                        "mode": "evaluation",
                    },
                }
                infer_started = time.perf_counter()
                response = _post_json(
                    str(args.policy_endpoint),
                    request_payload,
                    timeout=float(args.timeout_sec),
                )
                infer_ms = (time.perf_counter() - infer_started) * 1000.0
                action_chunk = response.get("action_chunk") or response.get(
                    "policy_result"
                )
                if not isinstance(action_chunk, dict):
                    failure_mode = "invalid_policy_response"
                    break
                action = action_chunk_first_vector(action_chunk)
                if action is None:
                    failure_mode = "invalid_action_chunk"
                    break
                driver.write_arm_targets(
                    str(args.arm), action_vector_to_targets(action, driver)
                )
                driver.step_control(1.0 / float(args.fps))
                step_record = {
                    "episode": episode,
                    "step": step,
                    "state": observation["state"],
                    "action": action,
                    "inference_ms": infer_ms,
                    "done": bool(action_chunk.get("done", False)),
                }
                episode_trace.append(step_record)
                if _success(
                    driver,
                    mode=str(args.success_mode),
                    object_body=args.object_body,
                    target_body=args.target_body,
                    min_lift_m=float(args.min_lift_m),
                    max_distance_m=float(args.max_distance_m),
                ):
                    success = True
                    failure_mode = ""
                    break
                if bool(action_chunk.get("done", False)):
                    failure_mode = "policy_done_without_success"
                    break
            elapsed = time.perf_counter() - started
            result = {
                "episode": episode,
                "success": success,
                "failure_mode": failure_mode,
                "steps": len(episode_trace),
                "elapsed_sec": elapsed,
            }
            episode_results.append(result)
            traces.extend(episode_trace)
            print(
                f"episode {episode + 1}/{args.episodes}: "
                f"success={success} steps={len(episode_trace)} failure={failure_mode or '-'}"
            )
    finally:
        await driver.close()

    summary = _summary(episode_results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "episodes.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in episode_results),
        encoding="utf-8",
    )
    (out / "trace.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in traces),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _build_endpoint_observation(
    driver: XLeRobotSimDriver,
    *,
    camera_names: list[str],
    image_size: tuple[int, int],
    arm: str,
) -> dict[str, Any]:
    frames = driver.render_camera_frames(camera_names)
    images = []
    for camera in camera_names:
        frame = frames.get(camera)
        if frame is None:
            continue
        images.append(
            {
                "camera": camera,
                "format": "jpeg",
                "data": _image_to_b64(frame, image_size),
            }
        )
    return {
        "frame_id": driver.frame_id,
        "images": images,
        "state": state_from_sim_driver(driver, arm=arm),
        "state_schema": SO101_STATE_SCHEMA,
        "active_arm": arm,
    }


def _image_to_b64(image: np.ndarray, image_size: tuple[int, int]) -> str:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if image_size[0] > 0 and image_size[1] > 0:
        pil = pil.resize(image_size, Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_json(
    endpoint: str, payload: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("policy endpoint must use http or https")
    req = urllib_request.Request(  # noqa: S310
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _success(
    driver: XLeRobotSimDriver,
    *,
    mode: str,
    object_body: str | None = None,
    target_body: str | None = None,
    min_lift_m: float = 0.03,
    max_distance_m: float = 0.08,
) -> bool:
    if mode == "none":
        return False
    if mode == "gripper_closed":
        status = dict(driver.last_arm_status or {})
        return float(status.get("gripper_opening_pct") or 100.0) < 25.0
    if mode == "object_lifted":
        position = _body_position_base(driver, object_body)
        return bool(position[2] >= min_lift_m)
    if mode == "object_near_target":
        object_position = np.asarray(
            _body_position_base(driver, object_body), dtype=float
        )
        target_position = np.asarray(
            _body_position_base(driver, target_body), dtype=float
        )
        return bool(np.linalg.norm(object_position - target_position) <= max_distance_m)
    raise ValueError(f"unsupported success mode: {mode}")


def _body_position_base(
    driver: XLeRobotSimDriver,
    body_name: str | None,
) -> tuple[float, float, float]:
    if not body_name:
        raise ValueError("success mode requires --object-body or --target-body")
    position_fn = getattr(driver, "_body_position_base", None)
    if not callable(position_fn):
        raise ValueError("driver does not expose body position lookup")
    value = position_fn(str(body_name))
    if not isinstance(value, tuple | list) or len(value) != 3:
        raise ValueError(f"invalid body position for {body_name!r}: {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for item in results if item.get("success"))
    failure_modes: dict[str, int] = {}
    for item in results:
        mode = str(item.get("failure_mode") or "success")
        failure_modes[mode] = failure_modes.get(mode, 0) + 1
    return {
        "episodes": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "mean_steps": (
            sum(float(item.get("steps") or 0.0) for item in results) / total
            if total
            else 0.0
        ),
        "failure_modes": failure_modes,
    }


def _parse_image_size(value: str) -> tuple[int, int]:
    width, height = str(value).lower().split("x", 1)
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SO101 VLA endpoint in XLeRobot home sim."
    )
    parser.add_argument("--config", default="configs/xlerobot.sim.vla_vln.yaml")
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--policy-endpoint", default="http://127.0.0.1:18080/predict")
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--cameras", default="front,right_wrist")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", default="256x256")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--success-mode",
        choices=("none", "gripper_closed", "object_lifted", "object_near_target"),
        default="gripper_closed",
    )
    parser.add_argument("--object-body", default=None)
    parser.add_argument("--target-body", default=None)
    parser.add_argument("--min-lift-m", type=float, default=0.03)
    parser.add_argument("--max-distance-m", type=float, default=0.08)
    parser.add_argument("--out", default="runtime/eval/home_so101_policy")
    return parser.parse_args()


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
