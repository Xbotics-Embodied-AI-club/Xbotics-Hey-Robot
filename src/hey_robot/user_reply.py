from __future__ import annotations

import json
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
    return value
