from pathlib import Path

from hey_robot.skill_os.command_store import SkillCommandStore, canonical_payload_hash


def test_receipt_replays_same_command_and_rejects_payload_conflict(
    tmp_path: Path,
) -> None:
    store = SkillCommandStore(tmp_path / "receipts.sqlite3")
    first = {"envelope": {"timestamp": 1.0}, "skill_id": "skill", "arguments": {"x": 1}}
    replay = {
        "envelope": {"timestamp": 2.0},
        "skill_id": "skill",
        "arguments": {"x": 1},
    }
    changed = {
        "envelope": {"timestamp": 2.0},
        "skill_id": "skill",
        "arguments": {"x": 2},
    }
    assert store.receive("skill", canonical_payload_hash(first)) == "new"
    assert store.receive("skill", canonical_payload_hash(replay)) == "replay"
    assert store.receive("skill", canonical_payload_hash(changed)) == "conflict"
    store.terminal("skill", {"status": "completed"})
    assert store.result("skill") == {"status": "completed"}
