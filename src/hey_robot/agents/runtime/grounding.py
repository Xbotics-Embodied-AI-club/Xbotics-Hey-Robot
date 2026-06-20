from __future__ import annotations

from hey_robot.skills.registry import load_skill_registry

PERCEPTION_TOOLS = {"request_perception"}
_PERCEPTION_OBSERVATION_ACTIONS = {
    "inspect_scene",
    "look_around",
    "detect_marker",
    "human_follow",
}

_TASK_VISUAL_MARKERS = (
    "what do you see",
    "what can you see",
    "look around",
    "look ahead",
    "inspect the scene",
    "current scene",
    "front camera",
    "camera",
    "visible",
    "see",
    "look",
    "observe",
    "看到",
    "看见",
    "看一下",
    "看一看",
    "观察",
    "场景",
    "画面",
    "镜头",
    "相机",
    "前方",
    "周围",
)

_RESPONSE_VISUAL_ASSERTION_MARKERS = (
    "i see",
    "i can see",
    "in front of me",
    "visible",
    "there is",
    "there are",
    "我看到",
    "我看见",
    "我面前",
    "我前方",
    "前视相机",
    "画面里",
    "镜头里",
    "可以看到",
)

_RESPONSE_UNGROUNDED_MARKERS = (
    "拿不到图像",
    "拿不到像素",
    "无法描述画面",
    "can't access the image",
    "cannot access the image",
    "do not have access to the image",
)


def needs_perception_grounding(task: str, response: str | None = None) -> bool:
    """Return whether the turn needs fresh perception evidence before final text.

    The check intentionally targets semantic evidence requirements, not one
    specific user phrase. It covers current-scene questions and model answers
    that make or refuse visual claims without having used a perception tool.
    """

    task_text = _normalize(task)
    response_text = _normalize(response or "")
    if any(marker in task_text for marker in _TASK_VISUAL_MARKERS):
        return True
    if response_text and any(
        marker in response_text for marker in _RESPONSE_VISUAL_ASSERTION_MARKERS
    ):
        return True
    return bool(
        response_text
        and any(marker in response_text for marker in _RESPONSE_UNGROUNDED_MARKERS)
    )


def is_perception_tool(name: str) -> bool:
    return name in PERCEPTION_TOOLS


def is_perception_evidence_record(name: str, arguments: dict, *, success: bool) -> bool:
    if not success:
        return False
    if is_perception_tool(name):
        return True
    return name == "request_capability" and is_perception_skill_name(
        str(arguments.get("capability") or "")
    )


def is_perception_skill_name(name: str) -> bool:
    if name in _PERCEPTION_OBSERVATION_ACTIONS:
        return True
    try:
        spec = load_skill_registry().catalog(enabled_only=False).get(name)
    except KeyError:
        return False
    return (
        spec.category == "perception"
        and spec.safety_level == "observe"
        and "camera" in spec.required_resources
    )


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())
