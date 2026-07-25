from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from hey_robot.protocol import RobotSkillAction, RobotSkillResult

_TASK_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "marker": ("marker", "pen", "马克笔", "记号笔"),
    "cup": ("cup", "mug", "杯", "杯子", "水杯"),
    "block": ("block", "cube", "积木", "方块"),
    "apple": ("apple", "苹果"),
    "bottle": ("bottle", "瓶子", "水瓶"),
    "mock_object": ("mock_object", "object", "物体", "目标物"),
}


def _target_from_task(task: str) -> str | None:
    text = str(task).lower()
    for candidate, aliases in _TASK_OBJECT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return candidate
    return None


class _WorldMixin:
    # ── attributes from _MockRobotDriverBase.__init__ ──────────────────────
    scan_counts: dict[str, int]
    action_attempts: int
    state: str
    settings: dict[str, Any]
    last_cameras_status: dict[str, Any]
    world: dict[str, Any]
    object_held: str | None
    world_events: list[dict[str, Any]]
    battery_percentage: float
    frame_id: int
    skill_counts: dict[str, int]
    _DEFAULT_WORLD: ClassVar[dict[str, Any]]

    if TYPE_CHECKING:

        def _battery_status(self) -> dict[str, Any]: ...

        def _readiness_resources(self) -> tuple[str, ...]: ...

        def _resource_available(self, resource: str, *, attempt_index: int) -> bool: ...

        @staticmethod
        def _clamp(value: float, low: float, high: float) -> float: ...

    def before_perception_request(
        self, skill_name: str, arguments: dict[str, Any] | None = None
    ) -> None:
        if skill_name not in {
            "inspect_scene",
            "look_around",
            "detect_marker",
        }:
            return
        args = dict(arguments or {})
        target = _target_from_task(str(args.get("question") or args.get("task") or ""))
        if not target:
            return
        self.scan_counts[target] = self.scan_counts.get(target, 0) + 1
        self._reveal_after_scan(target)
        self._record_world_event(
            "perception_scan",
            object=target,
            scans=self.scan_counts[target],
        )

    def readiness(self, *, attempt_index: int | None = None) -> dict[str, Any]:
        resolved_attempt = (
            self.action_attempts if attempt_index is None else attempt_index
        )
        readiness: dict[str, Any] = {
            "robot": self.state != "closed",
            "battery": self._battery_status(),
            "emergency_stop": bool(self.settings.get("emergency_stop", False))
            or bool((self.settings.get("safety") or {}).get("estop", False)),
        }
        for resource in self._readiness_resources():
            readiness[resource] = {
                "ok": self._resource_available(resource, attempt_index=resolved_attempt)
            }
        for camera_name, status in self.last_cameras_status.items():
            readiness.setdefault(
                f"{camera_name}_camera",
                {"ok": bool(status.get("ok")), "owner": "mock"},
            )
        return readiness

    def _world_from_settings(self) -> dict[str, Any]:
        source = self.settings.get("world") or self.settings.get("mock_world") or {}
        world = {
            "robot_near": self._DEFAULT_WORLD["robot_near"],
            "locations": {
                name: dict(value)
                for name, value in dict(self._DEFAULT_WORLD["locations"]).items()
            },
            "objects": {
                name: dict(value)
                for name, value in dict(self._DEFAULT_WORLD["objects"]).items()
            },
        }
        if isinstance(source, dict):
            if isinstance(source.get("locations"), dict):
                for name, value in source["locations"].items():
                    world["locations"][str(name)] = dict(value or {})
            if isinstance(source.get("objects"), dict):
                for name, value in source["objects"].items():
                    base = dict(world["objects"].get(str(name), {}))
                    base.update(dict(value or {}))
                    world["objects"][str(name)] = base
            if source.get("robot_near"):
                world["robot_near"] = str(source["robot_near"])
        for name, obj in world["objects"].items():
            obj.setdefault("label", name)
            obj.setdefault("location", "front_workspace")
            obj.setdefault("visible", True)
            obj.setdefault("graspable", True)
            obj.setdefault("held", False)
            obj.setdefault("color", [210, 80, 70])
        return world

    def _world_snapshot(self) -> dict[str, Any]:
        objects = {
            name: dict(value) for name, value in self.world.get("objects", {}).items()
        }
        visible = [name for name in objects if self._object_visible(name)]
        return {
            "robot_near": self.world.get("robot_near"),
            "held_object": self.object_held,
            "visible_objects": visible,
            "objects": objects,
            "locations": {
                name: dict(value)
                for name, value in self.world.get("locations", {}).items()
            },
            "events": list(self.world_events[-20:]),
            "scan_counts": dict(self.scan_counts),
        }

    def _scene_summary(self) -> dict[str, Any]:
        visible = [
            name for name in self.world.get("objects", {}) if self._object_visible(name)
        ]
        held = self.object_held or "nothing"
        visible_text = ", ".join(visible) or "none"
        return {
            "summary": f"front view near {self.world.get('robot_near')}; visible: {visible_text}; holding: {held}",
            "visible_objects": visible,
            "held_object": self.object_held,
            "robot_near": self.world.get("robot_near"),
            "object_locations": {
                name: self._object_location(name)
                for name in self.world.get("objects", {})
            },
            "last_event": self.world_events[-1] if self.world_events else None,
        }

    def _object(self, name: str) -> dict[str, Any] | None:
        obj = self.world.get("objects", {}).get(name)
        return obj if isinstance(obj, dict) else None

    def _object_location(self, name: str) -> str | None:
        obj = self._object(name)
        return (
            str(obj.get("location"))
            if obj is not None and obj.get("location") is not None
            else None
        )

    def _object_visible(self, name: str) -> bool:
        obj = self._object(name)
        if obj is None:
            return False
        if bool(obj.get("held")):
            return False
        if not bool(obj.get("visible", True)):
            return False
        false_negative_scans = self.settings.get("perception_false_negative_scans", {})
        if isinstance(false_negative_scans, dict):
            required = int(false_negative_scans.get(name, 0) or 0)
            if required > 0 and self.scan_counts.get(name, 0) <= required:
                return False
        return True

    def _nearest_graspable_object(self) -> str | None:
        robot_near = self.world.get("robot_near")
        for name, obj in self.world.get("objects", {}).items():
            if (
                bool(obj.get("graspable", True))
                and self._object_visible(str(name))
                and obj.get("location") == robot_near
            ):
                return str(name)
        for name, obj in self.world.get("objects", {}).items():
            if bool(obj.get("graspable", True)) and self._object_visible(str(name)):
                return str(name)
        return None

    def _release_held_object(self, location: str) -> None:
        if not self.object_held:
            return
        obj = self._object(self.object_held)
        if obj is not None:
            obj["held"] = False
            obj["visible"] = True
            obj["location"] = location
            self._record_world_event(
                "released", object=self.object_held, location=location
            )

    def _reveal_after_scan(self, target: str) -> None:
        obj = self._object(target)
        if obj is None:
            return
        reveal_after = self.settings.get("visible_after_scans", {})
        if not isinstance(reveal_after, dict):
            return
        required = int(reveal_after.get(target, 0) or 0)
        if (
            required > 0
            and self.scan_counts.get(target, 0) >= required
            and not bool(obj.get("visible", True))
        ):
            obj["visible"] = True
            self._record_world_event(
                "revealed", object=target, scans=self.scan_counts.get(target, 0)
            )

    def _record_world_event(self, kind: str, **payload: Any) -> None:
        self.world_events.append(
            {"kind": kind, "frame_id": self.frame_id, "time": time.time(), **payload}
        )

    def _scripted_failure(self, skill: RobotSkillAction) -> RobotSkillResult | None:
        scripts = (
            self.settings.get("scripted_failures")
            or self.settings.get("failure_script")
            or []
        )
        if not isinstance(scripts, list):
            return None
        attempt = self.skill_counts.get(skill.name, 0)
        for item in scripts:
            if not isinstance(item, dict):
                continue
            if str(item.get("skill") or item.get("name") or "") != skill.name:
                continue
            expected_attempt = item.get("attempt")
            if expected_attempt is not None and int(expected_attempt) != attempt:
                continue
            remaining_key = "_remaining"
            if "times" in item:
                item[remaining_key] = int(
                    item.get(remaining_key) or item.get("times") or 0
                )
                if int(item[remaining_key]) <= 0:
                    continue
                item[remaining_key] = int(item[remaining_key]) - 1
            message = str(
                item.get("message") or f"mock scripted failure for {skill.name}"
            )
            failure_mode = str(item.get("failure_mode") or "scripted_failure")
            return RobotSkillResult(
                False,
                message,
                {
                    "skill": skill.to_dict(),
                    "failure_mode": failure_mode,
                    "script": dict(item),
                    "attempt": attempt,
                },
            )
        return None

    def _drain_battery_for_skill(self, skill: RobotSkillAction) -> None:
        if self.settings.get("battery_status") is not None:
            return
        drain = 0.05
        if skill.name.startswith("base_"):
            drain = 0.25
        elif skill.name in {"set_arm_pose", "move_arm_joints", "set_gripper"}:
            drain = 0.18
        self.battery_percentage = self._clamp(
            self.battery_percentage - drain, 0.0, 100.0
        )
