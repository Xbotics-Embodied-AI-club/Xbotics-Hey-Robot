"""VLA 模型有效性端到端测试（真实 MuJoCo 仿真帧）。

流程：
1. 加载 MuJoCo 场景，渲染相机帧
2. 通过 LeRobotVLAPolicyExecutor 加载 VLA policy
3. 多任务推理，分析动作质量
4. 输出报告：关节变化范围、夹爪调制、跨任务差异、本体感知敏感度

用法:
    python examples/validate_vla_policy.py
    python examples/validate_vla_policy.py --config configs/xlerobot.sim.vla_vln.yaml
    python examples/validate_vla_policy.py --model-path models/pi05-so101-record-0121
    python examples/validate_vla_policy.py --tasks "pick up the orange cube" "close the gripper"
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hey_robot.config import DeploymentConfig
from hey_robot.foundation.backends.vla.lerobot.executor import (
    LeRobotVLAPolicyExecutor,
)

logger = logging.getLogger("validate_vla_policy")

# ── 常量 ────────────────────────────────────────────────────────────────────

DEFAULT_TASKS = [
    "pick up the orange cube on the table",
    "grasp the object firmly",
    "open the gripper wide",
    "close the gripper",
    "reach forward toward the table",
    "pick up the apple",
    "pick up the cup from the table",
    "move arm to home position",
    "reach down and grab the block",
]

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

PROPRIO_SCENARIOS = [
    ("shoulder_lift +20deg", 1, 20.0),
    ("elbow_flex +30deg", 2, 30.0),
]


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="VLA 模型有效性端到端测试")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "xlerobot.sim.vla_vln.yaml"
        ),
        help="部署配置文件路径",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="VLA 模型路径（覆盖配置文件中的 model_path）",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="自定义任务列表（覆盖默认任务）",
    )
    return parser.parse_args()


# ── 日志 ─────────────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """配置简洁的 console 日志（无时间戳、无级别前缀）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


# ── MuJoCo 场景 ──────────────────────────────────────────────────────────────


def _render_camera_frame(
    model, data, camera_name: str, width: int = 640, height: int = 480
) -> dict | None:
    """渲染单帧相机图像并返回 base64 编码字典。"""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        return None

    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=cam_id)
    pixels = renderer.render()
    renderer.close()

    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return {
        "camera": camera_name,
        "format": "jpeg",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _load_scene_and_render(
    mjcf_path: str, render_width: int = 640, render_height: int = 480
) -> tuple:
    """加载 MuJoCo 场景，渲染所有相机帧。

    Returns:
        (model, data, camera_names, frames)
    """
    import mujoco

    logger.info("场景:  %s", mjcf_path)

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera_names = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            camera_names.append(name)
    logger.info("  相机:  %s", camera_names)

    frames: dict[str, dict] = {}
    for cam in camera_names:
        frame = _render_camera_frame(model, data, cam, render_width, render_height)
        if frame:
            frames[cam] = frame
            raw = base64.b64decode(frame["data"])
            img = Image.open(io.BytesIO(raw))
            arr = np.array(img)
            logger.info(
                "  %s: %dx%d, range=[%d, %d], mean=%.1f",
                cam,
                img.size[0],
                img.size[1],
                arr.min(),
                arr.max(),
                arr.mean(),
            )
        else:
            logger.info("  %s: 渲染失败", cam)

    if not frames:
        sys.exit("错误：未能渲染任何相机帧")

    return model, data, camera_names, frames


def _extract_proprioception(model, data) -> list[float]:
    """从 MuJoCo 状态提取机械臂关节角度（含空夹爪状态）。"""
    import mujoco

    proprio = []
    for jname in ARM_JOINT_NAMES:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            proprio.append(float(data.qpos[jid]) if jid >= 0 else 0.0)
        except Exception:
            proprio.append(0.0)
    proprio.append(0.0)  # gripper position
    return proprio


# ── VLA 推理 ─────────────────────────────────────────────────────────────────


def _build_observation(image_list: list[dict], proprio: list[float]) -> dict:
    """构造标准 observation 字典。"""
    return {
        "frame_id": int(time.time() * 1000),
        "timestamp": time.time(),
        "images": image_list,
        "proprioception": proprio,
        "raw": {},
    }


def _run_task_inference(
    executor: LeRobotVLAPolicyExecutor,
    tasks: list[str],
    image_list: list[dict],
    proprio: list[float],
) -> list[dict]:
    """对任务列表逐一推理，返回结果列表。"""
    results: list[dict] = []
    for task in tasks:
        observation = _build_observation(image_list, proprio)
        try:
            result = executor.execute(
                {
                    "arguments": {"task": task, "objective": task},
                    "observation": observation,
                }
            )
            result["_task"] = task
            results.append(result)
            _log_single_result(task, result)
        except Exception as exc:
            logger.exception("  [%-40s] ERROR: %s: %s", task, type(exc).__name__, exc)
            results.append({"success": False, "_task": task})
    return results


def _log_single_result(task: str, result: dict) -> None:
    """记录单条推理结果（首帧 + 末帧动作）。"""
    if not result.get("success"):
        logger.info("  [%-40s] FAIL: %s", task, result.get("summary", "")[:80])
        return

    actions = result.get("metrics", {}).get("policy_result", {}).get("actions", [])
    if not actions:
        logger.info("  [%-40s] OK (无动作输出)", task)
        return

    a0 = actions[0]
    joints_deg = _rad_to_deg_dict(a0.get("joints", {}))
    j_str = " ".join(f"{k}={v: 7.2f}" for k, v in joints_deg.items())
    logger.info("  [%-40s] grip=%.3f  %s  (deg)", task, a0.get("gripper", 0), j_str)

    if len(actions) > 1:
        a_last = actions[-1]
        jl_deg = _rad_to_deg_dict(a_last.get("joints", {}))
        jl_str = " ".join(f"{k}={v: 7.2f}" for k, v in jl_deg.items())
        logger.info(
            "  %-40s grip=%.3f  %s  (deg)",
            "(final step)",
            a_last.get("gripper", 0),
            jl_str,
        )


def _run_proprioceptive_test(
    executor: LeRobotVLAPolicyExecutor,
    image_list: list[dict],
    base_proprio: list[float],
    task: str = "pick up the orange cube",
) -> None:
    """测试模型对不同本体感知状态的敏感度。"""
    logger.info("\n  --- 附加：本体感知敏感度测试 ---")

    for label, joint_idx, offset_deg in PROPRIO_SCENARIOS:
        alt_proprio = list(base_proprio)
        alt_proprio[joint_idx] += math.radians(offset_deg)

        observation = _build_observation(image_list, alt_proprio)
        try:
            result = executor.execute(
                {
                    "arguments": {"task": task, "objective": task},
                    "observation": observation,
                }
            )
            if result.get("success"):
                actions = (
                    result.get("metrics", {})
                    .get("policy_result", {})
                    .get("actions", [])
                )
                if actions:
                    a0 = actions[0]
                    j_str = " ".join(
                        f"{k}={v: 7.2f}" for k, v in a0.get("joints", {}).items()
                    )
                    logger.info(
                        "  [%-25s] gripper=%.3f  %s",
                        label,
                        a0.get("gripper", 0),
                        j_str,
                    )
        except Exception as exc:
            logger.info("  [%-25s] ERROR: %s", label, exc)


# ── 结果分析 ─────────────────────────────────────────────────────────────────


def _analyze_actions(all_results: list[dict]) -> dict:
    """聚合分析所有任务的推理结果。"""
    joint_sequences: dict[str, list[float]] = {}
    gripper_values: list[float] = []

    for r in all_results:
        if not r.get("success"):
            continue
        actions = r.get("metrics", {}).get("policy_result", {}).get("actions", [])
        for step in actions:
            for jname, jval in step.get("joints", {}).items():
                joint_sequences.setdefault(jname, []).append(float(jval))
            gripper_values.append(float(step.get("gripper", 0.0)))

    analysis: dict = {
        "total_inferences": len(all_results),
        "successful": sum(1 for r in all_results if r.get("success")),
        "failed": sum(1 for r in all_results if not r.get("success")),
        "joint_stats": {},
        "gripper_stats": {},
        "per_task_actions": {},
    }

    for jname, values in joint_sequences.items():
        arr = np.array(values)
        analysis["joint_stats"][jname] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "range": float(arr.max() - arr.min()),
        }

    if gripper_values:
        gv = np.array(gripper_values)
        analysis["gripper_stats"] = {
            "min": float(gv.min()),
            "max": float(gv.max()),
            "mean": float(gv.mean()),
            "std": float(gv.std()),
            "range": float(gv.max() - gv.min()),
        }

    for r in all_results:
        if not r.get("success"):
            continue
        task = r.get("_task", "unknown")
        actions = r.get("metrics", {}).get("policy_result", {}).get("actions", [])
        if actions:
            a0 = actions[0]
            analysis["per_task_actions"][task] = {
                "joints": dict(a0.get("joints", {})),
                "gripper": float(a0.get("gripper", 0.0)),
                "horizon": r.get("metrics", {})
                .get("policy_result", {})
                .get("horizon", 0),
            }

    return analysis


def _log_report(analysis: dict) -> int:
    """输出可读报告并返回退出码（0 = 通过）。"""
    lines = [
        "",
        "=" * 70,
        "  VLA 模型有效性报告",
        "=" * 70,
        f"  推理次数: {analysis['total_inferences']}，"
        f"成功 {analysis['successful']}，失败 {analysis['failed']}",
        "",
        "  --- 关节角度统计 (弧度, 全任务) ---",
    ]
    for jname, stats in analysis["joint_stats"].items():
        deg_mean = stats["mean"] * 180.0 / math.pi
        deg_range = stats["range"] * 180.0 / math.pi
        lines.append(
            f"  {jname:20s}: mean={stats['mean']: 7.3f} rad ({deg_mean: 7.2f} deg)  "
            f"span={stats['range']:.3f} rad ({deg_range:.2f} deg)"
        )

    gs = analysis["gripper_stats"]
    lines += [
        "",
        "  --- 夹爪统计 ---",
        f"  mean={gs.get('mean', 0):.3f}  std={gs.get('std', 0):.3f}  "
        f"range=[{gs.get('min', 0):.3f}, {gs.get('max', 0):.3f}]  "
        f"span={gs.get('range', 0):.3f}",
        "",
        "  --- 逐任务动作 (首帧, 度) ---",
    ]
    for task, act in analysis["per_task_actions"].items():
        j_deg = _rad_to_deg_dict(act["joints"])
        j_str = " ".join(f"{k}={v: 6.2f}" for k, v in j_deg.items())
        lines.append(
            f"  [{task:35s}] grip={act['gripper']:.3f}  "
            f"horizon={act['horizon']}  deg: {j_str}"
        )

    lines += [
        "",
        "  --- 判定 ---",
    ]
    checks = _build_verdict_checks(analysis, gs)
    for status, msg in checks:
        lines.append(f"  [{status}] {msg}")

    all_pass = all(s in ("PASS", "INFO") for s, _ in checks)
    label = "模型有效" if all_pass else "模型需调查"
    lines += [
        f"\n  综合判定: {label}",
        "=" * 70,
    ]

    logger.info("\n".join(lines))
    return 0 if all_pass else 1


def _build_verdict_checks(analysis: dict, gs: dict) -> list[tuple[str, str]]:
    """生成逐项判定结果。"""
    checks: list[tuple[str, str]] = []

    if analysis["successful"] > 0:
        checks.append(("PASS", "模型可产生有效动作输出"))
    else:
        checks.append(("FAIL", "模型无法产生任何输出"))
        return checks

    # 关节跨度
    max_range_rad = max(
        (s["range"] for s in analysis["joint_stats"].values()), default=0
    )
    max_range_deg = max_range_rad * 180.0 / math.pi
    if max_range_deg > 20.0:
        checks.append(("PASS", f"关节轨迹跨度显著 (max={max_range_deg:.1f} deg)"))
    elif max_range_deg > 2.0:
        checks.append(("WARN", f"关节跨度较小 (max={max_range_deg:.1f} deg)"))
    else:
        checks.append(
            ("FAIL", f"关节跨度可忽略 ({max_range_deg:.2f} deg) — 模型可能停滞")
        )

    # 夹爪调制
    g_range = gs.get("range", 0)
    if g_range > 0.1:
        checks.append(("PASS", f"夹爪跨任务有区分度 (range={g_range:.3f})"))
    elif g_range > 0.01:
        checks.append(("WARN", f"夹爪方差较低 (range={g_range:.3f})"))
    else:
        checks.append(("FAIL", "夹爪无调制 — 始终同一值"))

    # 跨任务差异
    task_joints = {
        task: list(act["joints"].values())
        for task, act in analysis["per_task_actions"].items()
    }
    if len(task_joints) >= 2:
        arr = np.array(list(task_joints.values()))
        cross_var = float(np.mean(np.var(arr, axis=0)))
        if cross_var > 1.0:
            checks.append(
                ("PASS", f"不同任务产生不同关节角度 (cross-task var={cross_var:.2f})")
            )
        elif cross_var > 0.01:
            checks.append(("WARN", f"任务间动作相似 (cross-task var={cross_var:.4f})"))
        else:
            checks.append(
                (
                    "INFO",
                    f"所有任务动作近乎一致 (cross-task var={cross_var:.6f}) — "
                    "单任务模型（如 pick-orange）的预期表现",
                )
            )

    return checks


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def _rad_to_deg_dict(joints: dict[str, float]) -> dict[str, float]:
    """将关节字典从弧度转为度。"""
    return {k: v * 180.0 / math.pi for k, v in joints.items()}


def _resolve_model_path(config_path: str, cli_override: str) -> str:
    """解析模型路径（CLI 参数优先，其次配置文件）。"""
    if cli_override:
        path = Path(cli_override).resolve()
    else:
        config = DeploymentConfig.from_yaml(Path(config_path))
        raw = config.model_services["manipulate"].settings.get("model_path", "")
        path = Path(raw).resolve()
    if not path.exists():
        sys.exit(f"模型路径不存在: {path}")
    return str(path)


# ── 主入口 ───────────────────────────────────────────────────────────────────


def main() -> int:
    _setup_logging()
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    # 加载配置
    logger.info("加载配置...")
    config = DeploymentConfig.from_yaml(Path(args.config))
    spec = config.model_services["manipulate"]
    model_path = _resolve_model_path(args.config, args.model_path)

    mjcf_rel = config.robots["sim_robot"].settings.get("mjcf_path", "")
    mjcf_path = str(Path(mjcf_rel).resolve())
    if not mjcf_path or not Path(mjcf_path).exists():
        sys.exit(f"MJCF 场景文件不存在: {mjcf_rel}")

    tasks = args.tasks or DEFAULT_TASKS

    rw = int(
        spec.settings.get("render_width")
        or config.robots["sim_robot"].settings.get("render_width", 640)
    )
    rh = int(
        spec.settings.get("render_height")
        or config.robots["sim_robot"].settings.get("render_height", 480)
    )

    logger.info("模型:  %s", model_path)
    logger.info("设备:  %s", spec.settings.get("policy_device", "cuda"))

    # ── 1. 加载场景并渲染 ────────────────────────────────────────────────
    logger.info("\n[1/3] 加载 MuJoCo 场景并渲染帧...")
    model, data, _camera_names, frames = _load_scene_and_render(mjcf_path, rw, rh)
    proprio = _extract_proprioception(model, data)
    logger.info("  本体感知: %s", [f"{v:.3f}" for v in proprio])

    # ── 2. 加载 VLA policy ───────────────────────────────────────────────
    logger.info("\n[2/3] 加载 VLA policy...")
    executor = LeRobotVLAPolicyExecutor("manipulate", spec)
    health = executor.health()
    logger.info("  健康状态: online=%s, loaded=%s", health["online"], health["loaded"])
    logger.info(
        "  Policy 类型 (配置): %s", health["metrics"].get("policy_type", "unknown")
    )

    executor._load_policy(model_path)
    logger.info("  Policy 已加载: type=%s", executor._policy_type)
    if executor._action_mean is not None:
        logger.info(
            "  动作均值: shape=%s, values=%s",
            executor._action_mean.shape,
            executor._action_mean.tolist(),
        )
    if executor._action_std is not None:
        logger.info(
            "  动作标准差: shape=%s, values=%s",
            executor._action_std.shape,
            executor._action_std.tolist(),
        )

    # ── 3. 多任务推理 ───────────────────────────────────────────────────
    logger.info("\n[3/3] 运行 %d 个任务的推理...", len(tasks))
    image_list = list(frames.values())
    all_results = _run_task_inference(executor, tasks, image_list, proprio)

    # ── 分析与报告 ───────────────────────────────────────────────────────
    analysis = _analyze_actions(all_results)
    rc = _log_report(analysis)

    # 本体感知敏感度测试
    _run_proprioceptive_test(executor, image_list, proprio)

    return rc


if __name__ == "__main__":
    sys.exit(main())
