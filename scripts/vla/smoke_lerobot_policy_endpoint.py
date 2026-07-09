from __future__ import annotations

import argparse
import base64
import io
import json
import time
from typing import Any
from urllib import request as urllib_request
from urllib.parse import urljoin, urlparse

from PIL import Image


def _image_data(size: tuple[int, int]) -> str:
    image = Image.new("RGB", size, color=(24, 48, 72))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _post_json(
    endpoint: str, payload: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use http or https")
    req = urllib_request.Request(  # noqa: S310
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _get_json(endpoint: str, *, timeout: float) -> dict[str, Any]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use http or https")
    with urllib_request.urlopen(endpoint, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _predict_payload(args: argparse.Namespace) -> dict[str, Any]:
    image_size = _parse_image_size(str(args.image_size))
    state = [float(item) for item in str(args.state).split(",")]
    return {
        "policy_session_id": "smoke-test",
        "skill_name": "manipulate",
        "atomic_command": str(args.task),
        "task": str(args.task),
        "frame_id": 0,
        "observation": {
            "frame_id": 0,
            "images": [
                {"camera": "front", "format": "jpeg", "data": _image_data(image_size)},
                {
                    "camera": "right_wrist",
                    "format": "jpeg",
                    "data": _image_data(image_size),
                },
            ],
            "state": state,
            "state_schema": "so101_single_arm_rad_gripper01",
            "active_arm": str(args.arm),
        },
        "metadata": {"mode": "smoke_test"},
    }


def _validate_action_chunk(response: dict[str, Any]) -> dict[str, Any]:
    chunk = response.get("action_chunk") or response.get("policy_result")
    if not isinstance(chunk, dict):
        raise ValueError("response is missing action_chunk")
    if chunk.get("kind") != "action_chunk":
        raise ValueError(f"unexpected action chunk kind: {chunk.get('kind')!r}")
    if chunk.get("action_space") != "xlerobot_single_arm_joint":
        raise ValueError(f"unexpected action_space: {chunk.get('action_space')!r}")
    actions = chunk.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action_chunk.actions must be a non-empty list")
    first = actions[0]
    if not isinstance(first, dict):
        raise ValueError("first action must be an object")
    joints = first.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("first action is missing joints")
    required_joints = {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    }
    missing = sorted(required_joints - set(joints))
    if missing:
        raise ValueError(f"first action is missing joints: {missing}")
    gripper = float(first.get("gripper", -1.0))
    if not 0.0 <= gripper <= 1.0:
        raise ValueError(f"gripper must be in [0, 1], got {gripper}")
    return chunk


def _parse_image_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def _health_url(predict_url: str) -> str:
    if predict_url.endswith("/predict"):
        return predict_url[: -len("/predict")] + "/health"
    return urljoin(predict_url.rstrip("/") + "/", "health")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test a Hey Robot LeRobot policy endpoint."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080/predict")
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--image-size", default="256x256")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--state",
        default="0.0,0.8,0.7,-0.6,0.0,1.0",
        help="Comma-separated SO101 state vector.",
    )
    parser.add_argument("--skip-health", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_health:
        health = _get_json(
            _health_url(str(args.endpoint)), timeout=float(args.timeout_sec)
        )
        print(json.dumps({"health": health}, indent=2))

    started = time.perf_counter()
    response = _post_json(
        str(args.endpoint),
        _predict_payload(args),
        timeout=float(args.timeout_sec),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    chunk = _validate_action_chunk(response)
    print(
        json.dumps(
            {
                "ok": True,
                "latency_ms": latency_ms,
                "horizon": len(chunk.get("actions") or []),
                "done": bool(chunk.get("done", False)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
