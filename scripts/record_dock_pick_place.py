"""Record a video of the single-arm dock pick-and-place cycle."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
from hey_robot.robot_runtime.simulation.so101_mobile.arm import So101MobileArmKernel
from hey_robot.robot_runtime.simulation.so101_mobile.gripper import WandGripperKernel
from hey_robot.robot_runtime.simulation.so101_mobile.session import So101MobileSession

OUTPUT = Path(__file__).resolve().parent.parent / "dock_pick_place.mp4"
FPS = 30


def render_frame(session: So101MobileSession, renderer, camera: str = "table_view"):
    renderer.update_scene(session.data, camera=camera)
    return renderer.render()


def main():
    session = So101MobileSession(
        render_width=640, render_height=480, viewer_enabled=False
    )
    session.connect()
    arm = So101MobileArmKernel(session)
    arm.bind()
    gripper = WandGripperKernel(session, arm)
    gripper.bind()

    # Set up renderer for recording
    renderer = mujoco.Renderer(session.model, height=480, width=640)

    wand_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_BODY, "wand")
    frames: list[np.ndarray] = []

    def record_steps(total_steps: int, sync_interval: int = 3):
        """Step simulation and record frames every sync_interval steps."""
        nonlocal frames
        for i in range(total_steps):
            session.clamp_base()
            mujoco.mj_step(session.model, session.data)
            if i % sync_interval == 0:
                frames.append(render_frame(session, renderer))

    print("Recording pick-and-place cycle...")

    # --- Initial view (1 second) ---
    for _ in range(FPS):
        for _ in range(3):
            session.clamp_base()
            mujoco.mj_step(session.model, session.data)
        frames.append(render_frame(session, renderer))

    # --- Open gripper ---
    print("  Opening gripper...")
    gripper.open()
    frames.extend(render_frame(session, renderer) for _ in range(FPS // 2))

    # --- Locate wand (simulated) ---
    target = session.data.xpos[wand_id].copy().tolist()
    print(f"  Wand at: {[f'{v:.3f}' for v in target]}")

    # --- IK pre-grasp ---
    pre_grasp = [target[0], target[1], target[2] + 0.08]
    pre_ik = arm.ik(tuple(pre_grasp))
    print(f"  Pre-grasp IK: {[f'{v:.3f}' for v in pre_ik]}")

    # --- IK grasp ---
    grasp_ik = arm.ik(tuple(target), current_joints=pre_ik)
    print(f"  Grasp IK: {[f'{v:.3f}' for v in grasp_ik]}")

    # --- Move to pre-grasp ---
    print("  Moving to pre-grasp...")
    dt = float(session.model.opt.timestep)
    steps = max(1, int(2.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, pre_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 2.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))
    # Snap ctrl
    for a, name in zip(arm._actuator_ids, arm.joint_names, strict=True):
        session.data.ctrl[a] = float(session.data.joint(name).qpos[0])

    # --- Move to grasp ---
    print("  Moving to grasp...")
    steps = max(1, int(1.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, grasp_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 1.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))
    for a, name in zip(arm._actuator_ids, arm.joint_names, strict=True):
        session.data.ctrl[a] = float(session.data.joint(name).qpos[0])

    # --- Close gripper (grasp) ---
    print("  Closing gripper (grasping)...")
    gripper.close()
    # Record a few frames post-grasp
    for _ in range(steps := max(1, int(0.5 / dt))):
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if _ % max(1, steps // 10) == 0:
            frames.append(render_frame(session, renderer))
    print(f"  Held: {gripper.held_object}")

    # --- Lift to pre-grasp ---
    print("  Lifting...")
    steps = max(1, int(1.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, pre_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 1.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))
    for a, name in zip(arm._actuator_ids, arm.joint_names, strict=True):
        session.data.ctrl[a] = float(session.data.joint(name).qpos[0])

    # --- Return home ---
    print("  Returning home...")
    home = [0.0, 0.8, 0.7, -0.6, 0.0]
    steps = max(1, int(3.0 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, home, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 3.0)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))

    # --- Hold at home (1 second) ---
    frames.extend(render_frame(session, renderer) for _ in range(FPS))

    # --- Now place back ---
    print("  Placing wand back...")

    # Dock position
    dock_x, dock_y, dock_z = 0.04, 0.133, 0.72
    approach_xyz = [dock_x, dock_y, dock_z + 0.13]
    insert_xyz = [dock_x, dock_y, dock_z + 0.02]

    approach_ik = arm.ik(tuple(approach_xyz))
    insert_ik = arm.ik(tuple(insert_xyz), current_joints=approach_ik)

    # Move to approach
    steps = max(1, int(2.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, approach_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 2.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))
    for a, name in zip(arm._actuator_ids, arm.joint_names, strict=True):
        session.data.ctrl[a] = float(session.data.joint(name).qpos[0])

    # Move to insert
    steps = max(1, int(1.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, insert_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 1.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))
    for a, name in zip(arm._actuator_ids, arm.joint_names, strict=True):
        session.data.ctrl[a] = float(session.data.joint(name).qpos[0])

    # Release
    print("  Releasing...")
    gripper.open()
    frames.extend(render_frame(session, renderer) for _ in range(FPS // 2))

    # Retract
    steps = max(1, int(1.5 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, approach_ik, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 1.5)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))

    # Home
    steps = max(1, int(3.0 / dt))
    start_ctrl = [float(session.data.ctrl[a]) for a in arm._actuator_ids]
    for i in range(steps):
        alpha = (i + 1) / steps
        for a, s, t in zip(arm._actuator_ids, start_ctrl, home, strict=True):
            session.data.ctrl[a] = s + alpha * (t - s)
        session.clamp_base()
        mujoco.mj_step(session.model, session.data)
        if i % int(steps / (FPS * 3.0)) == 0 or i == steps - 1:
            frames.append(render_frame(session, renderer))

    frames.extend(render_frame(session, renderer) for _ in range(FPS))

    # --- Write video ---
    print(f"  Writing {len(frames)} frames to {OUTPUT}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (640, 480))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.close()
    session.close()
    print(f"Done! Video saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
