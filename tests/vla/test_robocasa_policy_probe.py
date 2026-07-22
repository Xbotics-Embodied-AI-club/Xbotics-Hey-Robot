from __future__ import annotations

import sys
from types import ModuleType

from hey_robot.foundation.backends.vla.lerobot.robocasa_policy_probe import (
    offline_processor_overrides,
)


def test_offline_processor_overrides_resolve_pi052_snapshots(
    monkeypatch,
) -> None:
    snapshots = {
        "google/paligemma-3b-pt-224": "/cache/paligemma",
        "lerobot/fast-action-tokenizer": "/cache/action-tokenizer",
    }
    calls: list[tuple[str, bool]] = []

    def fake_snapshot_download(repo_id: str, *, local_files_only: bool) -> str:
        calls.append((repo_id, local_files_only))
        return snapshots[repo_id]

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setenv("ROBOCASA_OFFLINE", "1")
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    assert offline_processor_overrides("pi052") == {
        "preprocessor_overrides": {
            "pi052_text_tokenizer": {"tokenizer_name": "/cache/paligemma"},
            "action_tokenizer_processor": {
                "action_tokenizer_name": "/cache/action-tokenizer",
                "paligemma_tokenizer_name": "/cache/paligemma",
            },
        }
    }
    assert calls == [
        ("google/paligemma-3b-pt-224", True),
        ("lerobot/fast-action-tokenizer", True),
    ]


def test_offline_processor_overrides_only_apply_to_offline_pi052(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROBOCASA_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    assert offline_processor_overrides("pi052") == {}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert offline_processor_overrides("act") == {}
