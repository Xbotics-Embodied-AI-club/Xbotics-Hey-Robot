from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from hey_robot.vla.so101_schema import SO101_JOINT_NAMES, action_vector_to_chunk

LOGGER = logging.getLogger("lerobot_policy_server")

DEFAULT_CAMERA_FEATURES = {
    "front": "observation.images.front",
    "right_wrist": "observation.images.handeye",
}


class PredictRequest(BaseModel):
    policy_session_id: str | None = None
    skill_name: str | None = None
    atomic_command: str | None = None
    task: str | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    frame_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeRobotPolicyServer:
    def __init__(
        self,
        *,
        policy_type: str,
        checkpoint: str,
        device: str,
        image_size: tuple[int, int],
        camera_features: dict[str, str],
        state_key: str,
        action_units: str,
        action_scale: float,
        gripper_index: int,
        dataset_repo_id: str | None = None,
        dataset_root: str | None = None,
    ) -> None:
        self.policy_type = policy_type
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.image_size = image_size
        self.camera_features = camera_features
        self.state_key = state_key
        self.action_units = action_units
        self.action_scale = float(action_scale)
        self.gripper_index = int(gripper_index)
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.policy: Any | None = None

    def load(self) -> None:
        if self.policy is not None:
            return
        policy_cls = _load_policy_class(self.policy_type)
        kwargs: dict[str, Any] = {}
        dataset_stats = self._load_dataset_stats()
        if dataset_stats is not None:
            kwargs["dataset_stats"] = dataset_stats

        LOGGER.info(
            "Loading LeRobot policy type=%s checkpoint=%s",
            self.policy_type,
            self.checkpoint,
        )
        self.policy = policy_cls.from_pretrained(self.checkpoint, **kwargs)
        self.policy.to(self.device)
        self.policy.eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _load_dataset_stats(self) -> Any | None:
        if not self.dataset_repo_id:
            return None
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(
            self.dataset_repo_id,
            root=self.dataset_root or None,
        )
        return meta.stats

    @torch.inference_mode()
    def predict(self, request: PredictRequest) -> dict[str, Any]:
        self.load()
        assert self.policy is not None
        batch = self._build_batch(request)
        if hasattr(self.policy, "predict_action_chunk"):
            raw_action = self.policy.predict_action_chunk(batch)
        else:
            raw_action = self.policy.select_action(batch)
        action = _first_action_vector(raw_action)
        action_chunk = self._to_action_chunk(action)
        return {
            "action_chunk": action_chunk,
            "summary": f"{self.policy_type} action chunk produced",
            "metrics": {
                "vla": {
                    "backend": self.policy_type,
                    "checkpoint": self.checkpoint,
                    "frame_id": request.frame_id or request.observation.get("frame_id"),
                }
            },
        }

    def _build_batch(self, request: PredictRequest) -> dict[str, Any]:
        observation = dict(request.observation)
        images = observation.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("request.observation.images is required")

        batch: dict[str, Any] = {}
        for entry in images:
            if not isinstance(entry, dict):
                continue
            camera = str(entry.get("camera") or "")
            feature_key = self.camera_features.get(camera)
            if feature_key:
                batch[feature_key] = self._image_tensor(entry)

        if not any("image" in key for key in batch):
            raise ValueError(
                f"no configured camera found; expected {sorted(self.camera_features)}"
            )

        state = observation.get("state")
        if state is None:
            raise ValueError("request.observation.state is required")
        batch[self.state_key] = torch.tensor(
            np.asarray(state, dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        task = request.task or request.atomic_command or "manipulate"
        batch["task"] = [str(task)]
        return batch

    def _image_tensor(self, entry: dict[str, Any]) -> torch.Tensor:
        data = entry.get("data")
        if not data:
            raise ValueError("image entry is missing base64 data")
        image = Image.open(io.BytesIO(base64.b64decode(str(data)))).convert("RGB")
        if self.image_size != (0, 0):
            image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).to(self.device).unsqueeze(0)

    def _to_action_chunk(self, action: Sequence[float]) -> dict[str, Any]:
        values = [float(value) * self.action_scale for value in action]
        if self.action_units == "deg":
            values = [
                value * np.pi / 180.0 if index < len(SO101_JOINT_NAMES) else value
                for index, value in enumerate(values)
            ]
        gripper = (
            values[self.gripper_index] if self.gripper_index < len(values) else 0.5
        )
        action_values = [
            *values[: len(SO101_JOINT_NAMES)],
            max(0.0, min(1.0, float(gripper))),
        ]
        return action_vector_to_chunk(action_values)


def _load_policy_class(policy_type: str) -> Any:
    from lerobot.policies.factory import get_policy_class

    return get_policy_class(policy_type)


def _first_action_vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    while array.ndim > 1:
        array = array[0]
    return [float(item) for item in array.reshape(-1)]


def _parse_camera_features(raw: str | None) -> dict[str, str]:
    if raw is None:
        return dict(DEFAULT_CAMERA_FEATURES)
    value = json.loads(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError("--camera-features must be a non-empty JSON object")
    return {str(camera): str(feature) for camera, feature in value.items()}


def create_app(server: LeRobotPolicyServer) -> FastAPI:
    app = FastAPI(title="Hey Robot LeRobot Policy Server")

    @app.on_event("startup")
    async def startup() -> None:
        server.load()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "online": True,
            "loaded": server.policy is not None,
            "policy_type": server.policy_type,
            "checkpoint": server.checkpoint,
            "device": str(server.device),
        }

    @app.post("/predict")
    async def predict(request: PredictRequest) -> dict[str, Any]:
        try:
            return server.predict(request)
        except Exception as exc:
            LOGGER.exception("LeRobot policy prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a trained LeRobot policy as a Hey Robot action_chunk endpoint."
    )
    parser.add_argument(
        "--policy-type",
        required=True,
        help="LeRobot registered policy type, for example act, smolvla, or pi0.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", default="256x256")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--camera-features", default=None)
    parser.add_argument("--action-units", choices=("rad", "deg"), default="rad")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--gripper-index", type=int, default=5)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def _parse_image_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = LeRobotPolicyServer(
        policy_type=str(args.policy_type),
        checkpoint=str(Path(args.checkpoint).expanduser()),
        device=str(args.device),
        image_size=_parse_image_size(str(args.image_size)),
        camera_features=_parse_camera_features(args.camera_features),
        state_key=str(args.state_key),
        action_units=str(args.action_units),
        action_scale=float(args.action_scale),
        gripper_index=int(args.gripper_index),
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
    )
    uvicorn.run(
        create_app(server),
        host=str(args.host),
        port=int(args.port),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
