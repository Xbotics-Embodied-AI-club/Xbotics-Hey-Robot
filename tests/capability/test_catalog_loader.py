from __future__ import annotations

from typing import cast

from hey_robot.agents.tools.registry import ToolRegistry
from hey_robot.capability.catalog import CapabilityLoader
from hey_robot.skills import (
    SkillCatalog,
    SkillSpec,
    load_skill_registry,
)


def test_capability_loader_empty_manifest_has_only_runtime_capability_sections() -> (
    None
):
    payload = CapabilityLoader().build().to_dict()

    assert payload == {
        "tools": [],
        "robot_skill_actions": [],
    }


def test_capability_loader_builds_manifest() -> None:
    registry = ToolRegistry()
    registry.register_simple(
        "get_robot_status", lambda: "ok", read_only=True, source="local"
    )

    manifest = CapabilityLoader(
        tools=registry,
        robot_skills=load_skill_registry().catalog(enabled_only=False),
    ).build(robot_type="xlerobot")

    payload = manifest.to_dict()
    assert "prompt_skills" not in payload
    assert payload["tools"][0]["name"] == "get_robot_status"
    assert payload["tools"][0]["source"] == "local"
    names = {item["name"] for item in payload["robot_skill_actions"]}
    assert "move_base" in names
    assert "turn_base" in names
    assert "vla_manipulation" in names
    assert "foundation_locomotion_run" not in names


def test_capability_loader_manifest_exposes_skill_surface_not_backend_implementations() -> (
    None
):
    payload = CapabilityLoader(robot_skills=load_skill_registry()).build().to_dict()

    names = {item["name"] for item in payload["robot_skill_actions"]}

    assert "inspect_scene" in names
    assert "move_base" in names
    assert "set_gripper" in names
    assert "reset_posture" in names
    assert "vla_manipulation" in names
    assert "vln_navigate_run" not in names
    assert "foundation_locomotion_run" not in names


def test_capability_loader_normalizes_tool_defaults_and_filters_robot_skills() -> None:
    class ToolInventory:
        def list_tools(self) -> list[dict[str, object]]:
            return [{"name": "raw_status", "annotations": "bad-provider-shape"}]

    catalog = SkillCatalog(
        (
            SkillSpec(
                name="visible_xlerobot",
                description="Visible only on xlerobot.",
                category="perception",
                input_schema={"type": "object"},
                safety_level="observe",
                required_resources=("camera",),
                preconditions=("robot_online",),
                success_criteria=("fresh frame",),
                failure_modes=("camera_unavailable",),
                recovery_hints=("inspect_scene",),
                timeout_sec=3.0,
                interruptible=True,
                feedback_mode="none",
                refresh_observation=True,
            ),
        )
    )

    payload = (
        CapabilityLoader(
            tools=cast(ToolRegistry, ToolInventory()), robot_skills=catalog
        )
        .build(robot_type="xlerobot")
        .to_dict()
    )

    assert payload["tools"] == [
        {
            "name": "raw_status",
            "source": "local",
            "description": "",
            "input_schema": {},
            "safety_level": "normal",
            "read_only": False,
            "destructive": False,
        }
    ]
    assert payload["robot_skill_actions"] == [
        {
            "name": "visible_xlerobot",
            "description": "Visible only on xlerobot.",
            "input_schema": {"type": "object"},
            "safety_level": "observe",
            "required_resources": ["camera"],
            "preconditions": ["robot_online"],
            "success_criteria": ["fresh frame"],
            "failure_modes": ["camera_unavailable"],
            "recovery_hints": ["inspect_scene"],
            "timeout_sec": 3.0,
            "interruptible": True,
            "feedback_mode": "none",
            "refresh_observation": True,
        }
    ]
