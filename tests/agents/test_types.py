from __future__ import annotations

from hey_robot.agents.types import RobotSnapshot
from hey_robot.protocol import (
    Envelope,
    ImageRef,
    RobotObservation,
    RobotStatus,
    SkillResult,
)


def test_robot_snapshot_summary_includes_status_observation_skill_and_compact_metrics() -> (
    None
):
    envelope = Envelope(episode_id="ep1", robot_id="mock0")
    status = RobotStatus(
        envelope=envelope,
        frame_id=12,
        state="moving",
        task="pick cup",
        success=False,
        error="blocked",
        metrics={
            "custom": "x" * 100,
            "battery": {
                "status": "low",
                "voltage": 10.5,
                "extra": {"cell1": 3.5, "cell2": 3.4},
            },
            "arm": {"joint_count": 6, "position": [1, 2, 3, 4, 5, 6, 7]},
        },
    )
    observation = RobotObservation(
        envelope=envelope, frame_id=13, images=[ImageRef(uri="file://frame.jpg")]
    )
    skill_result = SkillResult(envelope=envelope, skill_id="skill1", status="completed")

    summary = RobotSnapshot(
        robot_id="mock0",
        status=status,
        observation=observation,
        skill_result=skill_result,
    ).summary()

    assert "robot_id=mock0" in summary
    assert "state=moving" in summary
    assert "frame_id=12" in summary
    assert "task=pick cup" in summary
    assert "success=False" in summary
    assert "error=blocked" in summary
    assert "observation_frame=13" in summary
    assert "images=1" in summary
    assert "last_skill=skill1:completed" in summary
    assert "battery={status=low,voltage=10.5,extra={cell1=...,cell2=...}}" in summary
    assert "arm={joint_count=6,position=[7 items]}" in summary
    assert "custom=" + ("x" * 80) + "..." in summary


def test_robot_snapshot_summary_omits_empty_optional_status_fields() -> None:
    envelope = Envelope(episode_id="ep1", robot_id="mock0")
    summary = RobotSnapshot(
        robot_id="mock0",
        status=RobotStatus(envelope=envelope, state="idle", metrics={}),
    ).summary()

    assert summary == "robot_id=mock0 state=idle"
