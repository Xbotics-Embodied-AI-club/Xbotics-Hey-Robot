from __future__ import annotations

from pathlib import Path

from hey_robot.protocol import Envelope, SkillEvent
from hey_robot.skills import SkillStore


def test_skill_store_materializes_lifecycle(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    envelope = Envelope(
        trace_id="tr1", episode_id="s1", agent_id="main", robot_id="mock0"
    )

    store.append(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="issued",
            text="move",
            mode="skill",
        )
    )
    store.append(
        SkillEvent(envelope=envelope, skill_id="skill1", phase="accepted", frame_id=3)
    )
    store.append(
        SkillEvent(
            envelope=envelope,
            skill_id="skill1",
            phase="completed",
            frame_id=9,
            steps_executed=6,
            progress=1.0,
            summary="done",
        )
    )

    recent = store.recent()
    assert recent[0]["skill_id"] == "skill1"
    assert recent[0]["phase"] == "completed"
    assert recent[0]["frame_id_start"] == 3
    assert recent[0]["frame_id_latest"] == 9
    assert len(recent[0]["timeline"]) == 3
