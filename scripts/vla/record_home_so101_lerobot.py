from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hey_robot.config import DeploymentConfig
from hey_robot.robot_runtime.manager import RobotManager
from hey_robot.robot_runtime.simulation.xlerobot_sim_driver import XLeRobotSimDriver
from hey_robot.vla.so101_schema import (
    SO101_STATE_SCHEMA,
    SO101_VECTOR_NAMES,
    action_vector_to_targets,
    state_from_sim_driver,
)


def _import_lerobot_dataset() -> tuple[Any, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is required for dataset recording. Run this script in the "
            "LeRobot training environment, or install the Hey Robot vla group."
        ) from exc
    return LeRobotDataset, None


class WaypointExpert:
    def __init__(self, waypoints: list[list[float]], *, hold_steps: int) -> None:
        self.waypoints = [np.asarray(item, dtype=np.float32) for item in waypoints]
        self.hold_steps = max(1, int(hold_steps))
        self.index = 0
        self.step_in_waypoint = 0

    def reset(self) -> None:
        self.index = 0
        self.step_in_waypoint = 0

    def act(self, _state: list[float]) -> np.ndarray:
        if not self.waypoints:
            raise RuntimeError("expert has no waypoints")
        action = self.waypoints[min(self.index, len(self.waypoints) - 1)]
        self.step_in_waypoint += 1
        if self.step_in_waypoint >= self.hold_steps:
            self.step_in_waypoint = 0
            self.index = min(self.index + 1, len(self.waypoints) - 1)
        return action.copy()

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints) - 1 and self.step_in_waypoint == 0


def _default_waypoints() -> list[list[float]]:
    return [
        [0.0, 0.8, 0.7, -0.6, 0.0, 1.0],
        [0.0, 0.7, 0.9, -0.5, 0.0, 1.0],
        [0.0, 0.9, 1.05, -0.75, 0.0, 1.0],
        [0.0, 0.9, 1.05, -0.75, 0.0, 0.0],
        [0.0, 0.65, 0.6, -0.45, 0.0, 0.0],
    ]


def _resize_rgb(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if image_size[0] > 0 and image_size[1] > 0:
        pil = pil.resize(image_size, Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _features(image_size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    width, height = image_size
    shape = (height, width, 3)
    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.handeye": {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(SO101_VECTOR_NAMES),),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(SO101_VECTOR_NAMES),),
            "names": ["action"],
        },
    }


async def record(args: argparse.Namespace) -> None:
    lerobot_dataset_cls, _ = _import_lerobot_dataset()
    root = Path(args.root)
    resume_existing = _prepare_dataset_root(
        root, overwrite=bool(args.overwrite), resume=bool(args.resume)
    )

    config = DeploymentConfig.from_yaml(args.config)
    manager = RobotManager(config)
    robot_id = args.robot_id or next(iter(config.robots))
    driver = manager.require(robot_id)
    if not isinstance(driver, XLeRobotSimDriver):
        raise SystemExit(
            f"recording currently requires xlerobot_sim driver, got {type(driver).__name__}"
        )

    image_size = _parse_image_size(args.image_size)
    if resume_existing:
        dataset = lerobot_dataset_cls(args.repo_id, root=str(root))
    else:
        dataset = lerobot_dataset_cls.create(
            repo_id=args.repo_id,
            root=str(root),
            robot_type="xlerobot_home_so101",
            fps=int(args.fps),
            features=_features(image_size),
            image_writer_threads=int(args.image_writer_threads),
            image_writer_processes=int(args.image_writer_processes),
        )

    waypoints = (
        _load_waypoints(args.waypoints) if args.waypoints else _default_waypoints()
    )
    expert = WaypointExpert(waypoints, hold_steps=int(args.hold_steps))
    camera_names = [
        item.strip() for item in str(args.cameras).split(",") if item.strip()
    ]

    await driver.start()
    try:
        for episode in range(int(args.episodes)):
            await driver.reset()
            expert.reset()
            for _ in range(int(args.max_steps)):
                frames = driver.render_camera_frames(camera_names)
                front = frames.get("front")
                handeye = frames.get("right_wrist")
                if handeye is None:
                    handeye = frames.get("handeye")
                if front is None or handeye is None:
                    raise RuntimeError(
                        f"missing required camera frames: {sorted(frames)}"
                    )

                state = state_from_sim_driver(driver, arm=str(args.arm))
                action = expert.act(state).astype(np.float32)
                dataset.add_frame(
                    {
                        "observation.images.front": _resize_rgb(front, image_size),
                        "observation.images.handeye": _resize_rgb(handeye, image_size),
                        "observation.state": np.asarray(state, dtype=np.float32),
                        "action": action,
                    },
                    task=str(args.task),
                )
                driver.write_arm_targets(
                    str(args.arm), action_vector_to_targets(action, driver)
                )
                driver.step_control(1.0 / float(args.fps))
                if expert.done:
                    break
            dataset.save_episode()
            print(f"saved episode {episode + 1}/{args.episodes}")
    finally:
        await driver.close()

    print(
        f"dataset ready: repo_id={args.repo_id} root={root} schema={SO101_STATE_SCHEMA}"
    )


def _load_waypoints(path: str) -> list[list[float]]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("waypoints JSON must be a list")
    return [[float(value) for value in item] for item in data]


def _prepare_dataset_root(root: Path, *, overwrite: bool, resume: bool) -> bool:
    if root.exists() and overwrite:
        shutil.rmtree(root)
        return False
    if root.exists() and not resume:
        raise SystemExit(
            f"dataset root already exists: {root} (use --overwrite or --resume)"
        )
    return root.exists() and resume


def _parse_image_size(value: str) -> tuple[int, int]:
    width, height = str(value).lower().split("x", 1)
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record XLeRobot home SO101 episodes as a LeRobotDataset."
    )
    parser.add_argument("--config", default="configs/xlerobot.sim.vla_vln.yaml")
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--repo-id", default="xlerobot_home_so101_single_arm")
    parser.add_argument("--root", default="data/lerobot/xlerobot_home_so101_single_arm")
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--cameras", default="front,right_wrist")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--hold-steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", default="256x256")
    parser.add_argument("--waypoints", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    asyncio.run(record(parse_args()))


if __name__ == "__main__":
    main()
