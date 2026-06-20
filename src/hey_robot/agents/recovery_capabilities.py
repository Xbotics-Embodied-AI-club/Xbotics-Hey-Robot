from __future__ import annotations

from typing import Any

RECOVERY_SAFE_CAPABILITIES = frozenset(
    {"inspect_scene", "stop_motion", "reset_posture"}
)


def is_recovery_safe_capability(skill_name: str, slots: dict[str, Any] | None) -> bool:
    if skill_name in RECOVERY_SAFE_CAPABILITIES:
        return True
    if skill_name != "set_gripper":
        return False
    arguments = dict(slots or {})
    action = str(arguments.get("action") or "").strip().lower()
    if action == "open":
        return True
    opening_pct = arguments.get("opening_pct")
    if opening_pct is None:
        return False
    try:
        return float(opening_pct) >= 80.0
    except (TypeError, ValueError):
        return False
