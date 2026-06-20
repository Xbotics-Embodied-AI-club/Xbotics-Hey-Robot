from __future__ import annotations

from pathlib import Path

from hey_robot.events import EventKind, RuntimeEvent
from hey_robot.events.store import RuntimeEventStore


def test_runtime_event_store_roundtrip(tmp_path: Path) -> None:
    store = RuntimeEventStore(tmp_path, max_items=5)
    store.append(RuntimeEvent.make(EventKind.GATEWAY_READY, source="gateway"))
    store.append(RuntimeEvent.make(EventKind.ROBOT_STARTED, source="robot"))

    recent = store.recent()
    assert len(recent) == 2
    assert store.count() == 2


def test_runtime_event_store_trims_in_batches_and_reads_latest_tail(
    tmp_path: Path,
) -> None:
    store = RuntimeEventStore(tmp_path, max_items=5, trim_every=2)
    for index in range(8):
        store.append(
            RuntimeEvent.make(f"event.{index}", source="test", payload={"index": index})
        )

    assert store.count() == 5
    assert [item["payload"]["index"] for item in store.recent(limit=3)] == [5, 6, 7]
