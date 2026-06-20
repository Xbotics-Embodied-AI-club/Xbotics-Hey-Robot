from __future__ import annotations

import pytest

from hey_robot.agents.autonomy import AutonomyManager


def test_autonomy_manager_deduplicates_prioritizes_and_tracks_context() -> None:
    manager = AutonomyManager(max_events=3, default_goal="monitor room")

    duplicate = manager.add_goal("monitor room", source="agent", priority=5)
    second = manager.add_goal("check shelf", source="agent", priority=1)
    third = manager.add_goal("inspect floor", source="operator", priority=3)

    manager.remember("vision", "cup on table", frame_id=7)
    manager.remember("vision", "chair moved", frame_id=8)
    manager.remember("status", "battery normal")
    manager.remember("note", "operator nearby")

    active = manager.active_goal()
    context = manager.prompt_context()

    assert duplicate.goal_id == 1
    assert duplicate.priority == 5
    assert active is not None
    assert active.text == "monitor room"
    assert second.goal_id == 2
    assert third.goal_id == 3
    assert "Active autonomous goal: monitor room" in context
    assert "Other active goals:" in context
    assert "frame=8: chair moved" in context
    assert "operator nearby" in context
    assert "cup on table" not in context


def test_autonomy_manager_complete_abandon_reset_and_error_paths() -> None:
    manager = AutonomyManager()

    assert manager.find_active_goal("") is None
    assert manager.prompt_context() == ""

    with pytest.raises(ValueError, match="goal text must not be empty"):
        manager.add_goal("   ")

    with pytest.raises(ValueError, match="no active goal"):
        manager.complete_goal()

    goal = manager.add_goal("inspect sink", priority=2)
    another = manager.add_goal("close drawer", priority=1)

    completed = manager.complete_goal(goal.goal_id)
    abandoned = manager.abandon_goal(another.goal_id, summary="unsafe to continue")

    assert completed.status == "completed"
    assert completed.summary == "completed"
    assert abandoned.status == "abandoned"
    assert abandoned.summary == "unsafe to continue"

    with pytest.raises(ValueError, match="unknown goal_id"):
        manager.abandon_goal(999)

    manager.remember("note", "")
    assert manager.goals_json().count('"goal_id"') == 2

    manager.reset()
    assert manager.active_goal() is None
    assert "inspect sink" in manager.goals_json()

    manager.reset(keep_goals=False)
    assert manager.goals_json() == '{"goals": []}'

    restarted = manager.add_goal("resume patrol")
    assert restarted.goal_id == 1
