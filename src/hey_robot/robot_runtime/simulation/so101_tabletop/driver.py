from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from hey_robot.contracts import SkillContractRuntime
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import (
    Envelope,
    RobotAction,
    RobotSkillAction,
    RobotSkillResult,
    RobotStatus,
)
from hey_robot.robot_runtime.base import (
    RobotCapabilities,
    RobotDriverContext,
    RobotHealth,
)
from hey_robot.robot_runtime.observations import DriverObservation, ObservationAsset
from hey_robot.robot_runtime.simulation.so101_tabletop.arm import (
    ARM_JOINT_NAMES,
    So101ArmKernel,
)
from hey_robot.robot_runtime.simulation.so101_tabletop.gripper import (
    WeldGripperKernel,
)
from hey_robot.robot_runtime.simulation.so101_tabletop.oracle import TabletopOracle
from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    DEFAULT_SCENE,
    So101TabletopSession,
)

logger = HeyRobotLogger(name="so101_tabletop_sim")

CAMERAS = ("overhead", "front", "side")
HOME_JOINTS = [-0.014, -1.238, 0.562, 0.858, 0.311]


class So101TabletopSimDriver:
    """MuJoCo driver for the reference SO101 tabletop environment."""

    def __init__(self, context: RobotDriverContext) -> None:
        self.context = context
        self.robot_id = context.robot_id
        self.settings = dict(context.spec.settings or {})
        scene_value = self.settings.get("mjcf_path") or DEFAULT_SCENE
        scene_path = Path(scene_value)
        if not scene_path.is_absolute():
            scene_path = Path.cwd() / scene_path
        viewer = self.settings.get("viewer", {}) or {}
        self.session = So101TabletopSession(
            scene_path,
            render_width=int(self.settings.get("render_width", 640)),
            render_height=int(self.settings.get("render_height", 480)),
            viewer_enabled=bool(viewer.get("enabled", False)),
        )
        self.arm = So101ArmKernel(self.session)
        self.gripper = WeldGripperKernel(self.session, self.arm)
        self.oracle = TabletopOracle(self.session, self.arm)
        self.contracts = SkillContractRuntime(context.skill_catalog)
        self.state = "created"
        self.frame_id = 0
        self.last_error: str | None = None
        self.last_skill_result: RobotSkillResult | None = None
        self._emergency_stop = False
        self._last_cameras: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        # Renderer creation must stay on the event-loop thread that owns the
        # OpenGL context (not inside asyncio.to_thread).
        self.session.connect()
        self.arm.bind()
        self.gripper.bind()
        self.state = "idle"
        self.last_error = None

    async def close(self) -> None:
        self.session.close()
        self.state = "closed"

    async def capabilities(self) -> RobotCapabilities:
        return RobotCapabilities(
            robot_id=self.robot_id,
            driver_type="so101_tabletop_sim",
            action_dimensions=6,
            control_hz=1.0 / float(self.session.model.opt.timestep)
            if self.session.connected
            else 500.0,
            cameras=list(CAMERAS),
            observation_modalities=["image", "arm_state", "simulation_oracle"],
            supports_reset=True,
            supports_interrupt=True,
            metadata={
                "body": "so101",
                "robot_family": "so101",
                "environment": "sim",
                "driver_kind": "mujoco",
                "embodiment_profile": (
                    self.context.embodiment.name
                    if self.context.embodiment is not None
                    else None
                ),
                "default_camera": "overhead",
                "perception_source": "mujoco_ground_truth",
                "grasp_model": "conditional_weld",
            },
        )

    async def health(self) -> RobotHealth:
        finite = self._state_is_finite()
        return RobotHealth(
            robot_id=self.robot_id,
            online=self.state != "closed",
            state=self.state if finite else "degraded",
            frame_id=self.frame_id,
            error=self.last_error if finite else "simulation state is not finite",
            metrics={
                "driver": "so101_tabletop_sim",
                "model_dimensions": self._model_dimensions(),
                "readiness": self.readiness(),
            },
        )

    async def observe(self) -> DriverObservation:
        self.session.require_connected()
        self.frame_id += 1
        frames = {name: self.session.render(name, bgr=True) for name in CAMERAS}
        self._last_cameras = {
            name: {
                "ok": frame is not None,
                "frame_id": self.frame_id,
                "image_shape": list(frame.shape),
                "owner": "simulation",
            }
            for name, frame in frames.items()
        }
        assets = [
            ObservationAsset(
                kind="image",
                role="camera",
                name=name,
                data=frame,
                metadata={
                    "driver": "so101_tabletop_sim",
                    "camera_role": name,
                    "color_order": "bgr",
                },
            )
            for name, frame in frames.items()
        ]
        return DriverObservation(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            assets=assets,
            proprioception=self._proprioception(),
            metadata={
                "driver": "so101_tabletop_sim",
                "body": "so101",
                "state": self.state,
                "cameras": self._last_cameras,
                "arm_status": self._arm_status(),
                "oracle": {
                    "source": self.oracle.source,
                    "simulation_only": True,
                    "objects": self.arm.get_object_positions(),
                    "caption": self.oracle.caption(),
                },
                "gripper": {
                    "held_object": self.gripper.held_object,
                    "welds": self.gripper.weld_states(),
                },
                "readiness": self.readiness(),
                "last_skill_result": (
                    self.last_skill_result.to_dict()
                    if self.last_skill_result is not None
                    else None
                ),
            },
        )

    async def status(self) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            success=None,
            error=self.last_error,
            metrics=self._metrics(),
        )

    async def reset(self) -> RobotStatus:
        self.session.reset()
        self.gripper.bind()
        self._emergency_stop = False
        self.frame_id = 0
        self.state = "idle"
        self.last_error = None
        self.last_skill_result = RobotSkillResult(True, "simulation reset")
        return await self.status()

    async def apply_action(self, action: RobotAction) -> RobotStatus:
        try:
            skill = RobotSkillAction.from_robot_action(action)
        except ValueError as exc:
            return self._failure(action, "invalid_action", str(exc))

        _, decision = self.contracts.validate_action(
            skill,
            robot_type="so101",
            status=await self.status(),
            readiness=self.readiness(),
        )
        if not decision.allowed:
            return self._failure(
                action,
                decision.failure_mode or "contract_rejected",
                decision.reason,
            )

        self.state = "executing"
        try:
            result = await self._execute_primitive(skill)
        except (RuntimeError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
            return self._failure(action, "primitive_failed", str(exc))

        self.last_skill_result = result
        if not result.success:
            self.last_error = result.message
            self.state = "failed"
            return self._status_for_action(action, success=False)
        self.last_error = None
        self.state = "skill_completed"
        status = self._status_for_action(action, success=True)
        self.state = "idle"
        return status

    async def _execute_primitive(self, skill: RobotSkillAction) -> RobotSkillResult:
        name = skill.name
        arguments = dict(skill.arguments)
        if name == "arm_get_state":
            return RobotSkillResult(
                True,
                "arm state read",
                {"joint_positions": self.arm.get_joint_positions()},
            )
        if name == "arm_solve_position_ik":
            target = _xyz(arguments.get("target_xyz"))
            seed_value = arguments.get("current_joints")
            seed = (
                [float(value) for value in seed_value]
                if isinstance(seed_value, (list, tuple))
                else None
            )
            solution = self.arm.ik(target, seed)
            if solution is None:
                return RobotSkillResult(
                    True,
                    f"IK unreachable for target {list(target)}",
                    {
                        "operation_success": False,
                        "failure_mode": "ik_unreachable",
                        "target_xyz": list(target),
                        "joint_positions": None,
                    },
                )
            return RobotSkillResult(
                True,
                "IK solved",
                {"operation_success": True, "joint_positions": solution},
            )
        if name == "move_arm_joints":
            joint_positions = self._resolve_joint_targets(arguments)
            duration = float(arguments.get("duration", 3.0))
            self.arm.move_joints(joint_positions, duration)
            return RobotSkillResult(
                True,
                "arm joints moved",
                {"joint_positions": self.arm.get_joint_positions()},
            )
        if name == "set_gripper":
            command = str(arguments.get("action") or "").lower()
            if command == "open":
                self.gripper.open()
            elif command == "close":
                self.gripper.close()
            else:
                opening = float(arguments.get("opening_pct", 0.0))
                (self.gripper.open if opening >= 50.0 else self.gripper.close)()
            return RobotSkillResult(
                True,
                f"gripper {command or 'set'}",
                {
                    "held_object": self.gripper.held_object,
                    "welds": self.gripper.weld_states(),
                },
            )
        if name == "sim_locate_object":
            query = str(arguments.get("query") or "")
            located = self.oracle.locate(
                query,
                sample_count=int(arguments.get("sample_count", 1)),
                sample_interval=float(arguments.get("sample_interval", 0.0)),
            )
            if located is None:
                return RobotSkillResult(
                    True,
                    f"object not found: {query}",
                    {
                        "operation_success": False,
                        "failure_mode": "object_not_found",
                        "query": query,
                    },
                )
            object_name, samples = located
            return RobotSkillResult(
                True,
                f"located {object_name}",
                {
                    "object_name": object_name,
                    "samples": samples,
                    "source": self.oracle.source,
                    "operation_success": True,
                },
            )
        if name == "sim_get_object_state":
            object_name = str(arguments.get("object_name") or "")
            if not object_name:
                object_name = self.gripper.held_object or ""
            object_positions = self.arm.get_object_positions()
            if object_name and object_name not in object_positions:
                return RobotSkillResult(
                    True,
                    f"unknown object: {object_name}",
                    {
                        "operation_success": False,
                        "failure_mode": "object_not_found",
                    },
                )
            return RobotSkillResult(
                True,
                "object state read",
                {
                    "object_name": object_name or None,
                    "position": object_positions.get(object_name),
                    "objects": object_positions if not object_name else None,
                    "held_object": self.gripper.held_object,
                    "welds": self.gripper.weld_states(),
                    "operation_success": True,
                },
            )
        if name in {"reset_posture", "set_arm_pose"}:
            pose_name = str(arguments.get("pose_name") or "home")
            if pose_name != "home":
                raise ValueError(f"unknown SO101 tabletop pose: {pose_name}")
            self.arm.move_joints(HOME_JOINTS, 3.0)
            return RobotSkillResult(True, "arm returned home")
        if name == "stop_motion":
            self.arm.stop()
            self._emergency_stop = bool(arguments.get("emergency", False))
            return RobotSkillResult(True, "arm motion stopped")
        if name == "inspect_scene":
            return RobotSkillResult(
                True,
                "scene inspected",
                {
                    "success": True,
                    "oracle": {
                        "source": self.oracle.source,
                        "caption": self.oracle.caption(),
                        "objects": self.arm.get_object_positions(),
                    },
                },
            )
        raise ValueError(f"unsupported SO101 tabletop primitive: {name}")

    def _resolve_joint_targets(self, arguments: dict[str, Any]) -> list[float]:
        raw = arguments.get("joints")
        if isinstance(raw, dict):
            current = dict(
                zip(ARM_JOINT_NAMES, self.arm.get_joint_positions(), strict=True)
            )
            mode = str(arguments.get("mode") or "absolute")
            for name, value in raw.items():
                if name not in current:
                    raise ValueError(f"unknown SO101 joint: {name}")
                if mode == "delta":
                    current[name] += float(value)
                else:
                    current[name] = float(value)
            return [current[name] for name in ARM_JOINT_NAMES]
        if isinstance(raw, (list, tuple)):
            return [float(value) for value in raw]
        raw = arguments.get("positions")
        if isinstance(raw, (list, tuple)):
            return [float(value) for value in raw]
        raise ValueError("move_arm_joints requires joints or positions")

    def readiness(self) -> dict[str, Any]:
        ready = self.session.connected and self._state_is_finite()
        result: dict[str, Any] = {
            "robot": ready,
            "arm": {"ok": ready},
            "gripper": {"ok": ready},
            "camera": {"ok": ready},
            "emergency_stop": self._emergency_stop,
        }
        for camera in CAMERAS:
            result[f"{camera}_camera"] = {"ok": ready, "owner": "simulation"}
        return result

    def _state_is_finite(self) -> bool:
        if not self.session.connected:
            return False
        return bool(
            np.all(np.isfinite(self.session.data.qpos))
            and np.all(np.isfinite(self.session.data.qvel))
            and np.all(np.isfinite(self.session.data.ctrl))
        )

    def _model_dimensions(self) -> dict[str, int]:
        if not self.session.connected:
            return {}
        model = self.session.model
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "neq": int(model.neq),
            "nbody": int(model.nbody),
            "ncam": int(model.ncam),
        }

    def _arm_status(self) -> dict[str, Any]:
        joints = self.arm.get_joint_positions()
        return {
            "success": self._state_is_finite(),
            "enabled": self.session.connected,
            "joint_count": 5,
            "joint_states": dict(zip(ARM_JOINT_NAMES, joints, strict=True)),
            "gripper_opening_pct": self.gripper.get_position() * 100.0,
        }

    def _proprioception(self) -> list[float]:
        return [
            *self.arm.get_joint_positions(),
            self.gripper.get_position(),
        ]

    def _metrics(self) -> dict[str, Any]:
        return {
            "driver": "so101_tabletop_sim",
            "runtime": "mujoco_simulation",
            "model_dimensions": self._model_dimensions(),
            "arm_status": self._arm_status() if self.session.connected else {},
            "gripper": {
                "held_object": self.gripper.held_object
                if self.session.connected
                else None,
                "welds": self.gripper.weld_states() if self.session.connected else {},
            },
            "cameras": self._last_cameras,
            "last_skill_result": (
                self.last_skill_result.to_dict()
                if self.last_skill_result is not None
                else None
            ),
            "readiness": self.readiness(),
        }

    def _failure(
        self, action: RobotAction, failure_mode: str, message: str
    ) -> RobotStatus:
        self.last_skill_result = RobotSkillResult(
            False, message, {"failure_mode": failure_mode}
        )
        self.last_error = message
        self.state = "failed"
        return self._status_for_action(action, success=False)

    def _status_for_action(self, action: RobotAction, *, success: bool) -> RobotStatus:
        return RobotStatus(
            envelope=self._envelope(),
            frame_id=self.frame_id,
            state=self.state,
            skill_id=action.skill_id,
            success=success,
            error=None if success else self.last_error,
            metrics=self._metrics(),
        )

    def _envelope(self) -> Envelope:
        return Envelope(
            robot_id=self.robot_id,
            deployment_id=self.context.deployment_id,
            trace_id=f"so101_tabletop_{self.robot_id}_{int(time.time() * 1000)}",
        )


def _xyz(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("target_xyz must contain three values")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise ValueError("target_xyz must be finite")
    return result
