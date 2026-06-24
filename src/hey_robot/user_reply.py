from __future__ import annotations

import json
import re
from typing import Any

_INTERNAL_TOOL_TEXTS = {
    "scene inspected",
    "inspect_scene completed",
    "skill completed",
    "skill accepted",
}


def present_tool_result_for_user(
    *,
    tool: str,
    args: dict[str, Any],
    result: str,
    success: bool | None,
) -> str | None:
    """Convert internal tool results into user-facing text."""

    if tool == "request_capability":
        return _present_capability_result(args=args, result=result, success=success)
    if tool == "request_perception":
        return _present_perception_result(result)
    payload = _json_object(result)
    if payload is not None:
        summary = _clean_user_text(
            str(payload.get("result") or payload.get("summary") or "")
        )
        return (
            summary if summary and not looks_like_internal_user_reply(summary) else None
        )
    clean = _clean_user_text(result)
    return clean if clean and not looks_like_internal_user_reply(clean) else None


def looks_like_internal_user_reply(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in _INTERNAL_TOOL_TEXTS:
        return True
    if lowered.startswith("execution feedback for skill "):
        return True
    if normalized.startswith(("用户说", "用户表示")):
        return True
    if "回顾一下之前的进展" in normalized:
        return True
    if lowered.startswith("issued "):
        return True
    if "subgoal_success:" in lowered or "task_success:" in lowered:
        return True
    if "skill trace:" in lowered or "task continuation:" in lowered:
        return True
    return lowered.endswith(" completed") and (
        "_" in lowered or lowered.startswith(("inspect", "request", "skill"))
    )


def present_runtime_event_for_user(*, kind: str, payload: dict[str, Any]) -> str | None:
    if kind != "skill.lifecycle":
        return None
    skill = str(payload.get("name") or payload.get("skill") or "")
    phase = str(payload.get("phase") or "")
    step = str(payload.get("step") or "")
    summary = _clean_user_text(
        str(payload.get("summary") or payload.get("error") or "")
    )
    ux_payload = payload.get("ux")
    ux = ux_payload if isinstance(ux_payload, dict) else {}
    ux_phase = str(ux.get("phase") or step or phase)

    if skill == "human_follow":
        return _human_follow_voice_text(ux_phase, summary=summary)
    if phase in {"failed", "interrupted", "feedback_failed"}:
        if summary and not looks_like_internal_user_reply(summary):
            return summary
        return "技能执行遇到问题，我已经进入恢复状态。"
    return None


def _present_capability_result(
    *, args: dict[str, Any], result: str, success: bool | None
) -> str | None:
    capability = str(args.get("capability") or args.get("name") or "").strip()
    payload = _json_object(result)
    if capability == "inspect_scene":
        if payload is not None:
            return _present_inspect_scene_payload(payload, success=success)
        clean = _clean_user_text(result)
        if clean and not looks_like_internal_user_reply(clean):
            return clean
        if success is False:
            return "我暂时没有拿到可用画面，不能可靠描述当前场景。"
        return "我已经看了一下当前画面。"

    clean = _clean_user_text(result)
    if clean and not looks_like_internal_user_reply(clean):
        return clean
    if success is False:
        return "这个动作没有成功完成。"
    if capability:
        return "动作已经完成。"
    return None


def _present_inspect_scene_payload(
    payload: dict[str, Any], *, success: bool | None
) -> str:
    summary = _clean_user_text(str(payload.get("summary") or ""))
    if summary and not looks_like_internal_user_reply(summary):
        return summary
    message = _clean_user_text(str(payload.get("message") or ""))
    if message and not looks_like_internal_user_reply(message):
        return message
    if success is False or payload.get("success") is False:
        return "我暂时没有拿到可用画面，不能可靠描述当前场景。"
    return "我已经看了一下当前画面。"


def _present_perception_result(result: str) -> str | None:
    payload = _json_object(result)
    if payload is None:
        clean = _clean_user_text(result)
        return clean if clean and not looks_like_internal_user_reply(clean) else None
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        summary = _clean_user_text(str(evidence.get("summary") or ""))
        if summary:
            return summary
        status = str(evidence.get("status") or "")
        if status in {"no_observation", "no_image", "stale", "caption_failed"}:
            return "我暂时没有拿到可用的视觉证据，不能可靠描述当前画面。"
    summary = _clean_user_text(
        str(payload.get("result") or payload.get("summary") or "")
    )
    return summary or None


def _human_follow_voice_text(phase: str, *, summary: str) -> str | None:
    mapping = {
        "starting": "我正在准备跟随，并检查相机和底盘状态。",
        "acquiring": "我正在寻找可以跟随的人。",
        "following": "我已经看到目标，正在跟随。",
        "searching": "目标短暂丢失，我正在原地小范围搜索。",
        "lost": "我找不到目标了，已经停止移动。请站到镜头前再继续。",
        "interrupted": "跟随已中断，我已经停止移动。",
        "completed": "跟随已完成，我已经停止移动。",
        "stopped": "跟随已停止。",
    }
    if phase in mapping:
        return mapping[phase]
    return summary if summary and not looks_like_internal_user_reply(summary) else None


def _json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_user_text(text: str) -> str:
    value = str(text or "").strip()
    if value.lower() in {"none", "null"}:
        return ""
    summary = _summary_line(value)
    if summary is not None:
        return _clean_user_text(summary)
    if "; robot_state=" in value:
        value = value.split("; robot_state=", 1)[0].strip()
    if "\r" in value:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in value:
        value = "\n".join(line.strip() for line in value.splitlines() if line.strip())
    lowered = value.lower()
    if value.startswith("任务监督告警："):
        detail = value.split("：", 1)[1].strip()
        return f"任务监督发现异常：{detail}。我会先暂停继续动作，避免扩大问题。"
    if lowered.startswith("consecutivemotionblocked:"):
        return "为了避免连续动作带来风险，我需要先重新观察当前画面，再继续执行。"
    if lowered.startswith("toolunavailable:") or (
        "not available in this execution context" in lowered
    ):
        return "当前运行环境不支持这个工具或能力，所以我没有继续执行动作。"
    if "capability not available" in lowered or "capability unavailable" in lowered:
        return "当前机器人不支持这个能力，所以我没有继续执行动作。"
    if lowered == "arm moved to pregrasp":
        return "机械臂已经切换到预抓取位姿。"
    if lowered.startswith("unknown named pose:"):
        pose_name = value.split(":", 1)[1].strip() or "requested"
        return f"当前没有名为“{_pose_display_name(pose_name)}”的已验证姿态，所以我没有移动机械臂。"
    for marker in ("unknown joint:", "invalid joint:"):
        if lowered.startswith(marker):
            joint_name = value.split(":", 1)[1].strip() or "requested"
            return f"当前没有名为“{joint_name}”的已验证关节，所以我没有移动机械臂。"
    base_move = re.fullmatch(
        r"base moved (?P<direction>[a-z_]+) (?P<distance>[-+]?\d+(?:\.\d+)?)cm",
        lowered,
    )
    if base_move is not None:
        direction = _direction_zh(base_move.group("direction"))
        distance = _format_number(float(base_move.group("distance")))
        return f"已经向{direction}移动约 {distance} 厘米。"
    base_turn = re.fullmatch(
        r"base turned (?P<direction>[a-z_]+) (?P<angle>[-+]?\d+(?:\.\d+)?)deg",
        lowered,
    )
    if base_turn is not None:
        direction = _direction_zh(base_turn.group("direction"))
        angle = _format_number(float(base_turn.group("angle")))
        return f"已经向{direction}转了约 {angle} 度。"
    return value


def _summary_line(text: str) -> str | None:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("- summary:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _direction_zh(direction: str) -> str:
    return {
        "forward": "前",
        "backward": "后",
        "left": "左",
        "right": "右",
    }.get(direction, direction.replace("_", " "))


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _pose_display_name(pose_name: str) -> str:
    return {
        "pre_grasp": "预抓取",
        "home": "home",
        "safe": "安全",
    }.get(pose_name, pose_name)
