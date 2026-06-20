from __future__ import annotations

import json

from hey_robot.agents.checkpoint import RobotAgentCheckpoint, RobotAgentCheckpointStore
from hey_robot.protocol import Envelope, UserTurn


def test_checkpoint_store_handles_missing_episode_and_invalid_payloads(
    tmp_path,
) -> None:
    store = RobotAgentCheckpointStore(tmp_path)

    assert store.load("missing") is None
    assert (
        store.enqueue_pending_turn(
            UserTurn(envelope=Envelope(trace_id="tr1"), text="hello"),
            reason="follow_up",
        )
        is None
    )
    assert store.pending_turns(None) == []

    broken = store.root / "broken.agent_checkpoint.json"
    broken.write_text("{not-json", encoding="utf-8")
    assert store.list_recent(limit=10) == []


def test_checkpoint_store_pop_pending_turn_handles_bad_payload_and_clear_edges(
    tmp_path,
) -> None:
    store = RobotAgentCheckpointStore(tmp_path)
    checkpoint = RobotAgentCheckpoint(
        episode_id="ep1",
        phase="responded",
        pending_turns=["bad-payload"],  # type: ignore[list-item]
    )
    store.save(checkpoint)

    assert store.pop_pending_turn("ep1") is None

    reset = store.reset_for_external_turn("ep1")
    assert reset is not None
    assert reset.phase == "idle"
    assert reset.skill_id is None
    assert reset.pending_turns == []

    store.clear_if_terminal("ep1")
    assert store.load("ep1") is None
    store.clear("ep1")


def test_checkpoint_store_skips_non_dict_pending_entries(tmp_path) -> None:
    store = RobotAgentCheckpointStore(tmp_path)
    path = store.root / "ep1.agent_checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "episode_id": "ep1",
                "phase": "idle",
                "skill_id": None,
                "updated_at": 1.0,
                "pending_turns": [
                    "bad",
                    {
                        "envelope": {"trace_id": "tr1", "episode_id": "ep1"},
                        "text": "ok",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    turns = store.pending_turns("ep1")
    assert [turn.text for turn in turns] == ["ok"]
