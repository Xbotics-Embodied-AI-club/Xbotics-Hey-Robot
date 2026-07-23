from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.habitat3.worker.preflight import required_assets
from evaluation.habitat3.worker.profiles import HabitatProfile, get_profile


@dataclass(frozen=True)
class RuntimeAsset:
    kind: str
    data: bytes
    role: str | None = None
    name: str | None = None
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeObservation:
    episode_id: str
    frame_id: int
    assets: list[RuntimeAsset] = field(default_factory=list)
    proprioception: list[float] = field(default_factory=list)
    task: str | None = None
    done: bool = False
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSkillResult:
    observation: RuntimeObservation
    steps: int
    success: bool
    failure_mode: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False


PROFILE = "habitat3_social_spot_human_oracle"
CONTROLLED_AGENT = "agent_0"
NPC_AGENT = "agent_1"


class HabitatEnvironment:
    """One-thread-owned Habitat Env adapter.

    All methods are intentionally synchronous. ``runtime_server.py`` executes
    them through a single owner worker, preserving Habitat-Sim/OpenGL affinity.
    """

    def __init__(self, *, data_root: str, profile: str = PROFILE) -> None:
        self.data_root = Path(data_root)
        self.profile_definition: HabitatProfile = get_profile(profile)
        self.profile = self.profile_definition.name
        self.env: Any | None = None
        self.episode_id: str | None = None
        self.frame_id = 0
        self._cancelled_operations: set[str] = set()

    def load(
        self,
        *,
        task: str,
        split: str,
        seed: int,
        controlled_agent: str,
        requested_dataset_episode_id: str | None = None,
    ) -> RuntimeObservation:
        if task != self.profile_definition.task:
            raise ValueError(f"task is not allowlisted: {task}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split is not allowlisted: {split}")
        if controlled_agent != CONTROLLED_AGENT:
            raise ValueError(f"controlled agent is not allowlisted: {controlled_agent}")
        self._require_assets()
        try:
            import habitat
            from habitat.config.default import get_config
        except ImportError as exc:
            raise RuntimeError("habitat-lab and habitat-sim are not installed") from exc

        config = get_config(config_paths=self.profile_definition.base_config)
        with habitat.config.read_write(config):
            config.habitat.dataset.split = split
            config.habitat.simulator.seed = int(seed)
            config.habitat.dataset.scenes_dir = str(
                self.data_root / "scene_datasets" / "hssd-hab"
            )
            config.habitat.dataset.data_path = str(
                self.data_root
                / "datasets"
                / "hssd"
                / "rearrange"
                / "{split}"
                / self.profile_definition.dataset_filename
            )
            self._configure_asset_paths(config)
        self.env = habitat.Env(config)
        if requested_dataset_episode_id:
            self._select_dataset_episode(requested_dataset_episode_id)
        observations = self.env.reset()
        self.episode_id = f"habitat_{uuid.uuid4().hex}"
        self.frame_id = 1
        return self._observation(observations)

    def observe(self) -> RuntimeObservation:
        self._require_env()
        # Core Env does not expose a separate public observe; extracting the
        # latest simulator observations does not advance physics.
        observations = self.env.sim.get_sensor_observations()
        observations.update(
            self.env.task.sensor_suite.get_observations(
                observations=observations,
                episode=self.env.current_episode,
                task=self.env.task,
                should_time=True,
            )
        )
        return self._observation(observations)

    def step(self, action: dict[str, Any], *, expected_frame_id: int):
        self._check_frame(expected_frame_id)
        self._require_env()
        observations = self.env.step(action)
        self.frame_id += 1
        return self._observation(observations)

    def reset(self, *, expected_frame_id: int) -> RuntimeObservation:
        self._check_frame(expected_frame_id)
        self._require_env()
        observations = self.env.reset()
        self.frame_id += 1
        return self._observation(observations)

    def close(self, *, expected_frame_id: int) -> None:
        self._check_frame(expected_frame_id)
        if self.env is not None:
            self.env.close()
        self.env = None

    def cancel(self, operation_id: str) -> bool:
        if not operation_id:
            return False
        self._cancelled_operations.add(operation_id)
        return True

    def execute_skill(
        self,
        *,
        operation_id: str,
        skill_name: str,
        arguments: dict[str, Any],
        expected_frame_id: int,
        max_steps: int,
    ) -> RuntimeSkillResult:
        self._check_frame(expected_frame_id)
        max_steps = max(1, min(int(max_steps), 1500))
        if skill_name not in self.profile_definition.skills:
            return self._failure(
                f"skill is not enabled by profile {self.profile}",
                "executor_unavailable",
            )
        if skill_name == "habitat_stop":
            return self._run_steps(operation_id, 1, self._zero_action, "stopped")
        if skill_name == "habitat_wait":
            return self._run_steps(
                operation_id,
                min(max_steps, int(arguments.get("steps", max_steps))),
                self._zero_action,
                "waited",
            )
        if skill_name == "habitat_follow_human":
            distance = float(arguments.get("distance_m", 2.0))
            return self._run_steps(
                operation_id,
                max_steps,
                lambda: self._follow_action(distance),
                "followed human",
            )
        if skill_name == "habitat_navigate_to":
            position = arguments.get("position")
            if isinstance(position, list) and len(position) == 3:
                target = np.asarray(position, dtype=np.float32)
            else:
                target = self._resolve_navigation_entity(
                    str(arguments.get("entity_id") or "")
                )
            if target is None:
                return self._failure(
                    "entity resolution is unavailable", "entity_unavailable"
                )
            return self._run_steps(
                operation_id,
                max_steps,
                lambda: self._navigate_action(target),
                "navigated",
            )
        if skill_name in {"habitat_symbolic_pick", "habitat_symbolic_place"}:
            return self._symbolic_action(skill_name, arguments)
        return self._failure(f"skill is not allowlisted: {skill_name}", "unknown_skill")

    def _run_steps(
        self, operation_id: str, max_steps: int, action_factory, success_text: str
    ):
        trace: list[dict[str, Any]] = []
        observation: RuntimeObservation | None = None
        for step_index in range(max_steps):
            if operation_id in self._cancelled_operations:
                self._cancelled_operations.discard(operation_id)
                return RuntimeSkillResult(
                    observation=observation or self.observe(),
                    steps=step_index,
                    success=False,
                    failure_mode="cancelled",
                    trace=trace,
                    cancelled=True,
                )
            observation = self.step(action_factory(), expected_frame_id=self.frame_id)
            trace.append({"step": step_index + 1, "frame_id": observation.frame_id})
            if observation.done:
                return RuntimeSkillResult(
                    observation=observation,
                    steps=step_index + 1,
                    success=observation.success,
                    failure_mode=None if observation.success else "episode_done",
                    metrics=dict(observation.metrics),
                    trace=trace,
                )
            if success_text == "navigated" and self._at_navigation_goal():
                return RuntimeSkillResult(
                    observation=observation,
                    steps=step_index + 1,
                    success=True,
                    metrics={**observation.metrics, "privileged": True},
                    trace=trace,
                )
        assert observation is not None
        return RuntimeSkillResult(
            observation=observation,
            steps=max_steps,
            success=success_text in {"waited", "stopped"},
            failure_mode=None if success_text in {"waited", "stopped"} else "max_steps",
            metrics={**observation.metrics, "privileged": success_text != "waited"},
            trace=trace,
        )

    def _symbolic_action(
        self, skill_name: str, arguments: dict[str, Any]
    ) -> RuntimeSkillResult:
        self._require_env()
        object_id = str(arguments.get("object_id") or "")
        receptacle_id = str(arguments.get("receptacle_id") or "")
        if not object_id or (skill_name.endswith("place") and not receptacle_id):
            return self._failure("PDDL entity is required", "invalid_arguments")
        action_name = "pick" if skill_name.endswith("pick") else "place"
        pddl_action = self.env.task.actions.get("agent_0_pddl_apply_action")
        if pddl_action is None:
            return self._failure(
                "profile does not expose agent_0 PDDL action", "executor_unavailable"
            )
        try:
            entities = list(pddl_action.entities)
            ordered = list(pddl_action._action_ordering)
            vector: list[float] = []
            for item in ordered:
                if item.name != action_name:
                    vector.extend([0.0] * item.n_args)
                    continue
                wanted = [object_id] + (
                    [receptacle_id] if action_name == "place" else []
                )
                indexes = [
                    next(
                        i + 1
                        for i, entity in enumerate(entities)
                        if str(entity) == value
                    )
                    for value in wanted
                ]
                vector.extend(float(index) for index in indexes)
                vector.extend([0.0] * (item.n_args - len(indexes)))
            observation = self.step(
                {
                    "action": "agent_0_pddl_apply_action",
                    "action_args": {"agent_0_pddl_action": vector},
                },
                expected_frame_id=self.frame_id,
            )
        except (StopIteration, ValueError, AttributeError) as exc:
            return self._failure(str(exc), "entity_unavailable")
        return RuntimeSkillResult(
            observation=observation,
            steps=1,
            success=True,
            metrics={
                **observation.metrics,
                "privileged": True,
                "pddl_action": action_name,
            },
            trace=[
                {
                    "step": 1,
                    "pddl_action": action_name,
                    "frame_id": observation.frame_id,
                }
            ],
        )

    def _zero_action(self) -> dict[str, Any]:
        npc_action = (
            "agent_1_oracle_nav_randcoord_action"
            if self.profile_definition.name == "habitat3_social_spot_human_oracle"
            else "agent_1_base_velocity"
        )
        npc_args: dict[str, Any]
        if npc_action == "agent_1_oracle_nav_randcoord_action":
            npc_args = {
                "agent_1_oracle_nav_randcoord_action": np.asarray(
                    [1.0], dtype=np.float32
                )
            }
        else:
            npc_args = {"agent_1_base_vel": np.asarray([0.0, 0.0], dtype=np.float32)}
        return {
            "action": ("agent_0_base_velocity", npc_action),
            "action_args": {
                "agent_0_base_vel": np.asarray([0.0, 0.0], dtype=np.float32),
                **npc_args,
            },
        }

    def _follow_action(self, distance_m: float) -> dict[str, Any]:
        self._require_env()
        target = np.asarray(
            self.env.sim.get_agent_data(1).articulated_agent.base_pos, dtype=np.float32
        )
        action = self._navigate_action(target, desired_distance=distance_m)
        action["action_args"]["agent_1_oracle_nav_randcoord_action"] = np.asarray(
            [1.0], dtype=np.float32
        )
        return action

    def _navigate_action(
        self, target: np.ndarray, desired_distance: float = 0.5
    ) -> dict[str, Any]:
        self._require_env()
        robot = self.env.sim.get_agent_data(0).articulated_agent
        position = np.asarray(robot.base_pos, dtype=np.float32)
        path = self.env.sim.pathfinder.find_path
        import habitat_sim

        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = position
        shortest.requested_end = target
        found = path(shortest)
        waypoint = np.asarray(
            shortest.points[1] if found and len(shortest.points) > 1 else target
        )
        relative = waypoint - position
        distance = float(np.linalg.norm(relative[[0, 2]]))
        forward = np.asarray(
            robot.base_transformation.transform_vector(np.array([1.0, 0.0, 0.0]))
        )[[0, 2]]
        direction = relative[[0, 2]]
        cross = float(forward[0] * direction[1] - forward[1] * direction[0])
        dot = float(np.dot(forward, direction))
        angle = math.atan2(cross, dot)
        linear = 0.0 if distance <= desired_distance or abs(angle) > 0.35 else 1.0
        angular = float(np.clip(angle, -1.0, 1.0))
        action = self._zero_action()
        action["action_args"]["agent_0_base_vel"] = np.asarray(
            [linear, angular], dtype=np.float32
        )
        return action

    def _at_navigation_goal(self) -> bool:
        metrics = self.env.get_metrics()
        return bool(
            metrics.get("nav_to_pos_succ", False) or metrics.get("pddl_success", False)
        )

    def _resolve_navigation_entity(self, entity_id: str) -> np.ndarray | None:
        """Resolve only allowlisted dynamic entities for the SocialNav profile."""
        if entity_id in {"agent_1", "human", "human_0"}:
            return np.asarray(
                self.env.sim.get_agent_data(1).articulated_agent.base_pos,
                dtype=np.float32,
            )
        return None

    def _observation(self, observations: dict[str, Any]) -> RuntimeObservation:
        self._require_env()
        assets: list[RuntimeAsset] = []
        proprioception: list[float] = []
        for key, value in observations.items():
            array = np.asarray(value)
            if "rgb" in key and array.ndim >= 3:
                encoded = _encode_rgb(array)
                assets.append(
                    RuntimeAsset(
                        "image",
                        encoded,
                        "camera",
                        key,
                        "image/jpeg",
                        array.shape[1],
                        array.shape[0],
                    )
                )
            elif "depth" in key and array.ndim >= 2:
                encoded = _encode_depth(array)
                assets.append(
                    RuntimeAsset(
                        "depth",
                        encoded,
                        "depth",
                        key,
                        "image/png",
                        array.shape[1],
                        array.shape[0],
                        {"artifact_type": "depth", "encoding": "uint16-mm"},
                    )
                )
            elif any(
                token in key
                for token in ("joint", "localization", "ee_pos", "is_holding")
            ):
                proprioception.extend(float(item) for item in array.reshape(-1)[:64])
        metrics = _json_safe(self.env.get_metrics())
        entities = self._pddl_entities()
        done = bool(self.env.episode_over)
        success = bool(metrics.get("nav_seek_success") or metrics.get("pddl_success"))
        return RuntimeObservation(
            episode_id=self.episode_id or "",
            frame_id=self.frame_id,
            assets=assets,
            proprioception=proprioception,
            task=str(getattr(self.env.current_episode, "episode_id", "")),
            done=done,
            success=success,
            metrics=metrics,
            entities=entities,
            metadata={
                "profile": self.profile,
                "controlled_agent": CONTROLLED_AGENT,
                "privileged": True,
            },
        )

    def _pddl_entities(self) -> list[dict[str, Any]]:
        try:
            entities = self.env.task.pddl_problem.get_ordered_entities_list()
        except AttributeError:
            return []
        return [
            {
                "entity_id": str(entity),
                "entity_type": str(getattr(entity, "expr_type", "pddl")),
                "attributes": {"privileged": True},
            }
            for entity in entities
        ]

    def _select_dataset_episode(self, requested: str) -> None:
        dataset = self.env.dataset
        matches = [item for item in dataset.episodes if item.episode_id == requested]
        if not matches:
            self.env.close()
            raise ValueError(f"requested dataset episode does not exist: {requested}")
        dataset.episodes = matches

    def _configure_asset_paths(self, config: Any) -> None:
        config.habitat.simulator.agents.agent_0.articulated_agent_urdf = str(
            self.data_root / "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        )
        config.habitat.simulator.agents.agent_1.articulated_agent_urdf = str(
            self.data_root / "humanoids/humanoid_data/female_2/female_2.urdf"
        )
        config.habitat.simulator.agents.agent_1.motion_data_path = str(
            self.data_root
            / "humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"
        )

    def _require_assets(self) -> None:
        missing = [
            str(check.path)
            for check in required_assets(self.data_root, self.profile_definition)
            if not check.path.exists()
        ]
        if missing:
            raise RuntimeError("Habitat assets unavailable: " + ", ".join(missing))

    def _check_frame(self, expected_frame_id: int) -> None:
        if int(expected_frame_id) != self.frame_id:
            raise ValueError(
                f"stale frame: expected={expected_frame_id}, current={self.frame_id}"
            )

    def _require_env(self) -> None:
        if self.env is None or self.episode_id is None:
            raise RuntimeError("Habitat episode is not active")

    def _failure(self, message: str, mode: str) -> RuntimeSkillResult:
        return RuntimeSkillResult(self.observe(), 0, False, mode, {"error": message})


def _encode_rgb(value: np.ndarray) -> bytes:
    array = np.asarray(value)[..., :3].astype(np.uint8)
    out = BytesIO()
    Image.fromarray(array).save(out, format="JPEG", quality=85)
    return out.getvalue()


def _encode_depth(value: np.ndarray) -> bytes:
    depth_mm = np.clip(np.asarray(value).squeeze() * 1000.0, 0, 65535).astype(np.uint16)
    out = BytesIO()
    Image.fromarray(depth_mm).save(out, format="PNG")
    return out.getvalue()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
