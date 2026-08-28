"""RoboCasa365 analytic motion primitives (world-frame transport + gripper).

These are the deterministic "division-of-labour" skills that complement the
RLDX VLA: the agent uses them to stage, transport, and re-position objects in
world coordinates, reserving the VLA (``manipulate``) for contact-rich work.

State layout (matches lerobot ``pixels_agent_pos`` ``agent_pos``, 16 dims):
    [0:3]   end_effector_position_relative   (base frame)
    [3:7]   end_effector_rotation_relative   (base frame, xyzw)
    [7:10]  base_position                    (world)
    [10:14] base_rotation                    (world, xyzw)
    [14:16] gripper_qpos                     (2 dims)

Action layout (robocasa 12D, normalized to [-1, 1]):
    [0:3]  ee_pos delta (OSC)   [3:6] ee_rot delta
    [6]    gripper (+1 close / -1 open)   [7:10] base motion (fwd/lat/turn)
    [10]   torso (unused)      [11]   control mode (-1 arm / +1 base)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from hey_robot.skills.builtins.common import execute_robot_action
from hey_robot.skills.builtins.robocasa_session import (
    mark_repositioned,
    mark_vla_desync,
    session_key,
)
from hey_robot.skills.context import SkillContext
from hey_robot.skills.models import Skill, SkillResult
from hey_robot.skills.registry import SkillRegistry

_OSC_POS_SCALE = 0.05  # action 1.0 -> 0.05 m target delta
_OSC_ROT_SCALE = 0.5  # action 1.0 -> 0.5 rad

# RPent owns one primitive object per episode, so its Jacobian and base-heading
# calibration survive separate tool calls.  A Hey Robot run id identifies one
# tool call; task id is the corresponding episode-scoped identity.
_state_cache: dict[tuple[str, str], _PrimitiveState] = {}


@dataclass
class _PrimitiveState:
    pos_jac: np.ndarray | None = None  # 3x3: world_dpos = J @ action_xyz
    fwd_offset: float | None = None  # world_forward = base_yaw + offset


@dataclass
class _Obs:
    frame_id: int
    proprioception: np.ndarray  # 16 dims
    eef_world: np.ndarray  # 3 dims
    base_pos: np.ndarray  # 3 dims
    base_quat: np.ndarray  # 4 dims (xyzw)
    gripper_qpos: np.ndarray  # 2 dims


def _quat_rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by unit quaternion ``q`` (xyzw)."""
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return cast(np.ndarray, v + w * t + np.cross(qv, t))


def _quat_yaw(q_xyzw: np.ndarray) -> float:
    x, y, z, w = (float(c) for c in q_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _parse_obs(obs: Any) -> _Obs:
    p = np.asarray(obs.proprioception, dtype=np.float64)
    base_pos = p[7:10]
    base_quat = p[10:14]
    eef_rel = p[0:3]
    eef_world = base_pos + _quat_rotate(base_quat, eef_rel)
    return _Obs(
        frame_id=int(obs.frame_id),
        proprioception=p,
        eef_world=eef_world,
        base_pos=base_pos,
        base_quat=base_quat,
        gripper_qpos=p[14:16],
    )


async def _observe(ctx: SkillContext) -> _Obs:
    obs = await ctx.observe(timeout_sec=10.0)
    return _parse_obs(obs)


async def _step(ctx: SkillContext, state: _Obs, action: list[float]) -> _Obs:
    """Send one native 12D action and return the post-action observation."""
    robot = ctx.robot
    if robot is None:
        raise RuntimeError("robot client is unavailable")
    # This mirrors RPent's ``_vla_desync = True`` on every non-VLA primitive:
    # the next policy call must reseed rather than stitch pre/post-manual frames.
    mark_vla_desync(ctx.robot_id, ctx.task_id)
    result = await robot.execute(
        ctx.robot_id,
        "embodiment_native_action",
        {"values": [float(v) for v in action]},
        run_id=ctx.run_id,
        expected_frame_id=state.frame_id,
    )
    if not result.success:
        raise RuntimeError(f"native action failed: {result.error or result.summary}")
    if result.observation is not None:
        return _parse_obs(result.observation)
    return await _observe(ctx)


def _state(ctx: SkillContext) -> _PrimitiveState:
    return _state_cache.setdefault(
        session_key(ctx.robot_id, ctx.task_id), _PrimitiveState()
    )


def _zero(base_mode: float = -1.0) -> np.ndarray:
    a = np.zeros(12)
    a[11] = base_mode
    return a


def _hold_gripper(g: Any) -> float:
    return float(np.clip(g, -1.0, 1.0))


async def _step_arm(
    ctx: SkillContext,
    state: _Obs,
    dpos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper: float = -1.0,
    n: int = 1,
) -> _Obs:
    a = _zero(base_mode=-1.0)
    a[0:3] = np.clip(np.asarray(dpos) / _OSC_POS_SCALE, -1.0, 1.0)
    a[3:6] = np.clip(np.asarray(drot) / _OSC_ROT_SCALE, -1.0, 1.0)
    a[6] = _hold_gripper(gripper)
    for _ in range(n):
        ctx.raise_if_cancelled()
        state = await _step(ctx, state, a.tolist())
    return state


async def _calibrate_pos_jacobian(ctx: SkillContext, state: _Obs) -> np.ndarray:
    """Probe 3 unit arm-xyz actions; measure world dpos -> 3x3 jacobian."""
    cols = []
    for axis in range(3):
        p0 = state.eef_world.copy()
        a = _zero()
        a[axis] = 0.4
        a[6] = _hold_gripper(1.0)
        for _ in range(3):
            ctx.raise_if_cancelled()
            state = await _step(ctx, state, a.tolist())
        d = (state.eef_world - p0) / (0.4 * 3.0)
        cols.append(d)
    return np.stack(cols, axis=1)  # 3x3


async def move_to(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Closed-loop OSC servo of the eef to a world xyz target."""
    xyz = arguments["xyz"]
    gripper = arguments.get("gripper", "hold")
    step_clip = float(arguments.get("step_clip", 0.02))
    max_steps = int(arguments.get("max_steps", 200))
    tol = float(arguments.get("tol", 0.012))

    target = np.asarray(xyz, dtype=np.float64)
    prim = _state(ctx)
    state = await _observe(ctx)
    target_q = float(state.gripper_qpos[0])
    moved = prim.pos_jac is None
    if prim.pos_jac is None:
        prim.pos_jac = await _calibrate_pos_jacobian(ctx, state)
    jinv = np.linalg.pinv(prim.pos_jac)
    for i in range(max_steps):
        state = await _observe(ctx)
        err = target - state.eef_world
        dist = float(np.linalg.norm(err))
        if dist < tol:
            if moved:
                mark_repositioned(ctx.robot_id, ctx.task_id)
            return _ok(i, dist, state, "reached")
        progress_fraction = min(1.0, step_clip / max(dist, 1e-12))
        cartesian_delta = err * progress_fraction
        a_xyz = np.clip(jinv @ cartesian_delta, -1.0, 1.0)
        g = _resolve_grip(gripper, state, target_q)
        a = _zero()
        a[0:3] = a_xyz
        a[6] = g
        state = await _step(ctx, state, a.tolist())
        moved = True
    cur = state.eef_world
    if moved:
        mark_repositioned(ctx.robot_id, ctx.task_id)
    return _ok(max_steps, float(np.linalg.norm(target - cur)), state, "budget")


async def move_delta(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    dxyz = arguments["dxyz"]
    state = await _observe(ctx)
    target = state.eef_world + np.asarray(dxyz, dtype=np.float64)
    return await move_to(
        ctx,
        {
            "xyz": target.tolist(),
            "gripper": arguments.get("gripper", "hold"),
            "step_clip": arguments.get("step_clip", 0.02),
            "max_steps": arguments.get("max_steps", 80),
        },
    )


def _resolve_grip(gripper: Any, state: _Obs, target_q: float) -> float:
    """'hold' maintains the current finger width; numeric passes through."""
    keep_current_opening = gripper is None or isinstance(gripper, str)
    if keep_current_opening:
        cur = float(state.gripper_qpos[0])
        return float(np.clip(60.0 * (cur - target_q), -1.0, 1.0))
    return _hold_gripper(gripper)


async def set_gripper(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    gripper = float(arguments.get("gripper", 1.0))
    steps = int(arguments.get("steps", 10))
    state = await _observe(ctx)
    a = _zero()
    a[6] = _hold_gripper(gripper)
    for _ in range(steps):
        ctx.raise_if_cancelled()
        state = await _step(ctx, state, a.tolist())
    return _ok(steps, 0.0, state, "held")


async def release(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    return await set_gripper(
        ctx, {"gripper": -1.0, "steps": int(arguments.get("steps", 10))}
    )


async def scripted_grasp(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Open -> hover -> descend -> close -> lift (coarse fallback)."""
    xyz = arguments["xyz"]
    approach_z = float(arguments.get("approach_z", 0.10))
    grasp_z_offset = float(arguments.get("grasp_z_offset", 0.0))
    step_clip = float(arguments.get("step_clip", 0.02))
    grasp_point = np.asarray(xyz, dtype=np.float64)

    await set_gripper(ctx, {"gripper": -1.0, "steps": 4})
    r = await move_to(
        ctx,
        {
            "xyz": (grasp_point + np.array((0.0, 0.0, approach_z))).tolist(),
            "gripper": -1.0,
            "step_clip": step_clip,
        },
    )
    if not r.success:
        return _stage(r, "approach")
    r = await move_to(
        ctx,
        {
            "xyz": (grasp_point + np.array((0.0, 0.0, grasp_z_offset))).tolist(),
            "gripper": -1.0,
            "step_clip": 0.012,
            "tol": 0.01,
        },
    )
    if not r.success:
        return _stage(r, "descent")
    await set_gripper(ctx, {"gripper": 1.0, "steps": 14})
    r = await move_to(
        ctx,
        {
            "xyz": (grasp_point + np.array((0.0, 0.0, approach_z + 0.05))).tolist(),
            "gripper": "hold",
            "step_clip": 0.015,
        },
    )
    if not r.success:
        return _stage(r, "lift")
    return r


async def move_base(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Raw base velocity (robot-local: +fwd, +lateral, +turn yaw)."""
    forward = float(arguments.get("forward", 0.0))
    lateral = float(arguments.get("lateral", 0.0))
    turn = float(arguments.get("turn", 0.0))
    steps = int(arguments.get("steps", 10))
    gripper = arguments.get("gripper", "hold")
    state = await _observe(ctx)
    target_q = float(state.gripper_qpos[0])
    a = _zero(base_mode=1.0)
    requested_velocity = np.asarray((forward, lateral, turn), dtype=np.float64)
    a[7:10] = np.clip(requested_velocity, -1.0, 1.0)
    for _ in range(steps):
        ctx.raise_if_cancelled()
        a[6] = _resolve_grip(gripper, state, target_q)
        state = await _step(ctx, state, a.tolist())
    if steps > 0:
        mark_repositioned(ctx.robot_id, ctx.task_id)
    return _ok(steps, 0.0, state, "driven")


async def _calibrate_forward(ctx: SkillContext, state: _Obs) -> float:
    p0 = state.base_pos.copy()
    y0 = _quat_yaw(state.base_quat)
    a = _zero(base_mode=1.0)
    a[6] = 1.0
    a[7] = 1.0
    for _ in range(6):
        ctx.raise_if_cancelled()
        state = await _step(ctx, state, a.tolist())
    disp = (state.base_pos - p0)[:2]
    if np.linalg.norm(disp) > 0.005:
        return math.atan2(float(disp[1]), float(disp[0])) - y0
    return 0.0


async def navigate_to(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Drive the mobile base toward a world (x, y) target (closed-loop)."""
    xy = arguments["xy"]
    tol = float(arguments.get("tol", 0.20))
    max_steps = int(arguments.get("max_steps", 300))
    gripper = arguments.get("gripper", "hold")
    target = np.array([float(value) for value in xy[:2]], dtype=np.float64)
    prim = _state(ctx)
    state = await _observe(ctx)
    target_q = float(state.gripper_qpos[0])
    moved = prim.fwd_offset is None
    if prim.fwd_offset is None:
        prim.fwd_offset = await _calibrate_forward(ctx, state)
    for i in range(max_steps):
        state = await _observe(ctx)
        bp = state.base_pos[:2]
        to = target - bp
        dist = float(np.linalg.norm(to))
        if dist < tol:
            prim.pos_jac = None  # base moved -> recalibrate arm
            if moved:
                mark_repositioned(ctx.robot_id, ctx.task_id)
            return _ok(i, dist, state, "reached")
        world_dir = math.atan2(float(to[1]), float(to[0]))
        cur_forward = _quat_yaw(state.base_quat) + prim.fwd_offset
        dyaw = (world_dir - cur_forward + math.pi) % (2 * math.pi) - math.pi
        a = _zero(base_mode=1.0)
        a[6] = _resolve_grip(gripper, state, target_q)
        if abs(dyaw) > 0.30:
            a[9] = float(np.sign(dyaw))
        else:
            a[7] = 1.0
            a[9] = float(np.clip(dyaw * 1.5, -0.4, 0.4))
        state = await _step(ctx, state, a.tolist())
        moved = True
    prim.pos_jac = None
    if moved:
        mark_repositioned(ctx.robot_id, ctx.task_id)
    return _ok(
        max_steps, float(np.linalg.norm(target - state.base_pos[:2])), state, "budget"
    )


_CAMERA_ALIAS = {
    "camera1": "robot0_agentview_left",
    "camera2": "robot0_agentview_right",
    "camera3": "robot0_eye_in_hand",
    "agentview": "robot0_agentview_left",
    "wrist": "robot0_eye_in_hand",
}


async def localize(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Back-project pixels with RPent metric depth, with a plane fallback."""
    pixels = arguments["pixels"]
    z = float(arguments.get("z", 0.9))
    camera = str(arguments.get("camera", "camera1"))
    transport_camera = {
        "agentview": "camera1",
        "wrist": "camera3",
        "robot0_agentview_left": "camera1",
        "robot0_agentview_right": "camera2",
        "robot0_eye_in_hand": "camera3",
    }.get(camera, camera)
    if await _has_robot_action(ctx, "localize_pixels"):
        return await execute_robot_action(
            ctx,
            "localize_pixels",
            {"pixels": pixels, "camera": transport_camera},
        )
    cam = _CAMERA_ALIAS.get(camera, camera)

    obs = await ctx.observe(timeout_sec=10.0)
    cameras = dict((obs.raw or {}).get("cameras", {}) or {})
    calib = cameras.get(cam)
    if not calib or "intrinsic" not in calib:
        return SkillResult(
            False,
            f"camera calibration unavailable for {cam}",
            "failed",
            failure_mode="calibration_unavailable",
        )
    intrinsics = np.asarray(calib["intrinsic"], dtype=np.float64)
    camera_to_world = np.asarray(calib["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    image_height = int(calib["height"])
    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]

    results = []
    valid = []
    for pixel in pixels:
        row, col = int(pixel[0]), int(pixel[1])
        # The raw OpenGL render is bottom-up; the agent sees row 0 at the top
        # of the PNG, while the intrinsics use bottom-up v. Flip the row.
        u = float(col)
        v = float(image_height - 1 - row)
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        d_world = rotation @ d_cam
        if abs(float(d_world[2])) < 1e-6:
            results.append({"pixel": pixel, "world_xyz": None, "valid": False})
            continue
        ray_scale = (z - float(origin[2])) / float(d_world[2])
        world = origin + ray_scale * d_world
        world = world.tolist()
        results.append(
            {
                "pixel": pixel,
                "world_xyz": [round(float(x), 4) for x in world],
                "valid": True,
            }
        )
        valid.append(world)
    summary = f"localized {len(valid)}/{len(pixels)} pixels at z={z}"
    return SkillResult(
        True,
        summary,
        "completed",
        data={"results": results, "camera": cam, "z": z},
    )


async def _has_robot_action(ctx: SkillContext, action_name: str) -> bool:
    if ctx.robot is None:
        return False
    try:
        capabilities = await ctx.robot.capabilities(ctx.robot_id)
    except (AttributeError, NotImplementedError):
        return False
    return action_name in {action.name for action in capabilities.actions}


def _ok(steps: int, dist: float, state: _Obs, stage: str) -> SkillResult:
    return SkillResult(
        True,
        f"moved ({stage}): {steps} steps, final_dist={dist:.3f}",
        "completed",
        data={
            "steps": steps,
            "final_dist": round(dist, 4),
            "eef_world": [round(float(v), 4) for v in state.eef_world],
            "base_pos": [round(float(v), 4) for v in state.base_pos],
            "gripper_qpos": [round(float(v), 4) for v in state.gripper_qpos],
        },
    )


def _stage(result: SkillResult, stage: str) -> SkillResult:
    return SkillResult(
        result.success,
        f"{stage} failed: {result.summary}",
        result.status,
        data={**result.data, "stage": stage},
    )


MOVE_TO = Skill(
    name="move_to",
    description=(
        "Servo the arm end-effector to a WORLD-frame [x, y, z] target (meters). "
        "Use for precise transport of the gripper. gripper='hold' keeps the "
        "current grasp width; +1 closes, -1 opens."
    ),
    parameters={
        "type": "object",
        "properties": {
            "xyz": {"type": "array", "items": {"type": "number"}},
            "gripper": {"type": ["number", "string"]},
            "step_clip": {"type": "number"},
            "max_steps": {"type": "integer"},
            "tol": {"type": "number"},
        },
        "required": ["xyz"],
        "additionalProperties": False,
    },
    handler=move_to,
    resources=("arm",),
    timeout_sec=60.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

MOVE_DELTA = Skill(
    name="move_delta",
    description="Displace the end-effector by a relative [dx, dy, dz] in world meters.",
    parameters={
        "type": "object",
        "properties": {
            "dxyz": {"type": "array", "items": {"type": "number"}},
            "gripper": {"type": ["number", "string"]},
            "step_clip": {"type": "number"},
            "max_steps": {"type": "integer"},
        },
        "required": ["dxyz"],
        "additionalProperties": False,
    },
    handler=move_delta,
    resources=("arm",),
    timeout_sec=60.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

SET_GRIPPER_RC = Skill(
    name="grip",
    description="Hold the arm pose and drive the gripper (+1 close, -1 open) for N steps.",
    parameters={
        "type": "object",
        "properties": {
            "gripper": {"type": "number"},
            "steps": {"type": "integer"},
        },
        "additionalProperties": False,
    },
    handler=set_gripper,
    resources=("gripper",),
    timeout_sec=20.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

RELEASE = Skill(
    name="release",
    description="Open the gripper to drop a grasped object.",
    parameters={
        "type": "object",
        "properties": {"steps": {"type": "integer"}},
        "additionalProperties": False,
    },
    handler=release,
    resources=("gripper",),
    timeout_sec=20.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

SCRIPTED_GRASP = Skill(
    name="scripted_grasp",
    description=(
        "Coarse scripted grasp: open -> hover above target -> descend -> close -> lift. "
        "A deterministic fallback; prefer manipulate (VLA) for hard objects."
    ),
    parameters={
        "type": "object",
        "properties": {
            "xyz": {"type": "array", "items": {"type": "number"}},
            "approach_z": {"type": "number"},
            "grasp_z_offset": {"type": "number"},
            "step_clip": {"type": "number"},
        },
        "required": ["xyz"],
        "additionalProperties": False,
    },
    handler=scripted_grasp,
    resources=("arm", "gripper"),
    timeout_sec=120.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

MOVE_BASE_RC = Skill(
    name="drive_base",
    description="Raw base velocity (robot-local): +forward, +lateral, +turn yaw. Values in [-1, 1].",
    parameters={
        "type": "object",
        "properties": {
            "forward": {"type": "number"},
            "lateral": {"type": "number"},
            "turn": {"type": "number"},
            "steps": {"type": "integer"},
            "gripper": {"type": ["number", "string"]},
        },
        "additionalProperties": False,
    },
    handler=move_base,
    resources=("base",),
    timeout_sec=30.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)

NAVIGATE_TO_RC = Skill(
    name="drive_to",
    description="Drive the mobile base toward a WORLD [x, y] target (closed-loop).",
    parameters={
        "type": "object",
        "properties": {
            "xy": {"type": "array", "items": {"type": "number"}},
            "tol": {"type": "number"},
            "max_steps": {"type": "integer"},
            "gripper": {"type": ["number", "string"]},
        },
        "required": ["xy"],
        "additionalProperties": False,
    },
    handler=navigate_to,
    resources=("base",),
    # RPent permits the full 300-step closed-loop navigation budget. Hey's
    # remote path also transports a post-step observation, so 120 steps can
    # exceed one minute even though simulator control itself is fast.
    timeout_sec=240.0,
    supported_robots=("robocasa",),
    required_actions=("embodiment_native_action",),
)


LOCALIZE = Skill(
    name="localize",
    description=(
        "Back-project image pixels (row, col) to WORLD xyz using aligned metric "
        "camera depth, matching RPent's world map. camera: camera1=agentview "
        "left, camera3=wrist. The z plane is used only by backends without depth."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pixels": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "z": {"type": "number"},
            "camera": {"type": "string"},
        },
        "required": ["pixels"],
        "additionalProperties": False,
    },
    handler=localize,
    resources=("camera",),
    timeout_sec=15.0,
    supported_robots=("robocasa",),
    required_actions=(),
)


async def read_progress(ctx: SkillContext, arguments: dict[str, Any]) -> SkillResult:
    """Read the structured task-progress predicates for the active task."""
    del arguments
    obs = await ctx.observe(timeout_sec=10.0)
    progress = dict((obs.raw or {}).get("task_progress", {}) or {})
    return SkillResult(
        True,
        f"task_progress={progress}",
        "completed",
        data={"task_progress": progress},
    )


READ_PROGRESS = Skill(
    name="read_progress",
    description=(
        "Read the structured task-progress predicates for the active RoboCasa "
        "task (for example success_time, washed_time, kettle_on_site, "
        "burner_on, washed_loc). Use this to verify sub-goal completion "
        "precisely instead of asking a visual question via inspect_scene."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=read_progress,
    resources=("camera",),
    timeout_sec=15.0,
    supported_robots=("robocasa",),
    required_actions=(),
)


def register(registry: SkillRegistry) -> None:
    registry.register(MOVE_TO)
    registry.register(MOVE_DELTA)
    registry.register(SET_GRIPPER_RC)
    registry.register(RELEASE)
    registry.register(SCRIPTED_GRASP)
    registry.register(MOVE_BASE_RC)
    registry.register(NAVIGATE_TO_RC)
    registry.register(LOCALIZE)
    registry.register(READ_PROGRESS)


__all__ = ["register"]
