from __future__ import annotations

from hey_robot.interaction import InteractionStateStore
from hey_robot.protocol import Envelope, UserTurn


def test_interaction_state_store_records_turn_and_persists(tmp_path) -> None:
    store = InteractionStateStore(tmp_path)
    turn = UserTurn(
        envelope=Envelope(episode_id="ep1", channel="voice"),
        text="stop following",
    )

    state = store.record_turn(
        turn,
        active_task_id="task1",
        pending_confirmation={"objective": "follow me"},
        robot_busy=True,
    )

    assert state is not None
    assert state.active_task_id == "task1"
    assert state.active_channel == "voice"
    assert state.last_user_intent == "interrupt"
    assert state.preferred_reply_mode == "voice"
    assert state.confirmation_reason == "follow me"

    restored = InteractionStateStore(tmp_path).get("ep1")
    assert restored is not None
    assert restored.linked_channels == ("voice",)
