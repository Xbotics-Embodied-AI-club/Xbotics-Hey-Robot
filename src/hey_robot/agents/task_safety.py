from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SkillSafetyContract(Protocol):
    @property
    def safety_level(self) -> str: ...


@dataclass(frozen=True)
class TaskSafetyDecision:
    allowed: bool
    reason: str = ""
    reply: str | None = None
    rule: str = "allow"


UNSUPPORTED_PHYSICAL_TASK_REPLY = "我现在不能安全地执行这个任务。当前系统没有完整的开门能力，只能观察门的位置和门把手，不会移动或操作门。"
VOICE_MOTION_CONFIRMATION_REPLY = "语音指令不能直接触发移动或接触环境的动作。请使用 propose_capability 工具向用户请求确认，确认通过后才能执行；不要跳过确认直接拒绝。"


def evaluate_user_task(
    text: str, *, channel: str | None, settings: dict
) -> TaskSafetyDecision:
    if not _enabled(settings):
        return TaskSafetyDecision(True)

    normalized = _normalize(text)
    if not normalized:
        return TaskSafetyDecision(True)

    if _is_unsupported_door_task(normalized):
        return TaskSafetyDecision(
            False,
            reason="unsupported high-risk door operation",
            reply=UNSUPPORTED_PHYSICAL_TASK_REPLY,
            rule="unsupported_physical_task",
        )

    if (
        _is_voice_channel(channel)
        and _voice_motion_confirmation_required(settings)
        and _looks_like_motion(normalized)
    ):
        return TaskSafetyDecision(
            False,
            reason="voice motion requires explicit confirmation",
            reply=VOICE_MOTION_CONFIRMATION_REPLY,
            rule="voice_motion_confirmation",
        )

    return TaskSafetyDecision(True)


def evaluate_skill_request(
    *,
    capability: str,
    objective: str,
    contract: SkillSafetyContract,
    task: str,
    channel: str | None,
    settings: dict,
    confirmed: bool = False,
) -> TaskSafetyDecision:
    if not _enabled(settings):
        return TaskSafetyDecision(True)

    combined = _normalize(f"{task} {objective} {capability}")
    if _is_unsupported_door_task(combined):
        return TaskSafetyDecision(
            False,
            reason="unsupported high-risk door operation",
            reply=UNSUPPORTED_PHYSICAL_TASK_REPLY,
            rule="unsupported_physical_task",
        )

    if (
        _is_voice_channel(channel)
        and _voice_motion_confirmation_required(settings)
        and not confirmed
        and contract.safety_level in {"motion", "normal"}
        and not _is_observe_or_stop(contract)
    ):
        return TaskSafetyDecision(
            False,
            reason=f"voice channel cannot directly request {contract.safety_level} skill {capability}",
            reply=VOICE_MOTION_CONFIRMATION_REPLY,
            rule="voice_motion_confirmation",
        )

    return TaskSafetyDecision(True)


def _enabled(settings: dict) -> bool:
    safety = settings.get("task_safety")
    if isinstance(safety, dict) and "enabled" in safety:
        return bool(safety.get("enabled"))
    return True


def _voice_motion_confirmation_required(settings: dict) -> bool:
    safety = settings.get("task_safety")
    if isinstance(safety, dict) and "voice_motion_requires_confirmation" in safety:
        return bool(safety.get("voice_motion_requires_confirmation"))
    return True


def _is_voice_channel(channel: str | None) -> bool:
    return str(channel or "").strip().lower() == "voice"


def _normalize(text: str) -> str:
    return "".join(str(text or "").lower().split()).strip("。！？!?，,；;：:")


def _is_unsupported_door_task(text: str) -> bool:
    return any(
        token in text
        for token in ("打开门", "开门", "opendoor", "openadoor", "openthedoor")
    )


def _looks_like_motion(text: str) -> bool:
    markers = (
        "前进",
        "后退",
        "左转",
        "右转",
        "移动",
        "靠近",
        "过去",
        "转向",
        "move",
        "forward",
        "backward",
        "turn",
        "approach",
        "navigate",
    )
    return any(marker in text for marker in markers)


def _is_observe_or_stop(contract: SkillSafetyContract) -> bool:
    return contract.safety_level in {"observe", "stop", "emergency"}
