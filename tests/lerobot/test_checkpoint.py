from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from hey_robot.foundation.backends.lerobot import (
    checkpoint as checkpoint_module,
    executor as executor_module,
)
from hey_robot.foundation.backends.lerobot.checkpoint import (
    checkpoint_metadata,
    load_checkpoint_config,
    offline_processor_overrides,
    register_policy_processors,
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
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
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
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    assert offline_processor_overrides("pi052") == {}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert offline_processor_overrides("act") == {}


def test_checkpoint_metadata_does_not_filter_policy_family() -> None:
    metadata = checkpoint_metadata(
        {
            "type": "fastwam",
            "input_features": {
                "observation.state": {"shape": [32]},
            },
            "output_features": {"action": {"shape": [14]}},
        }
    )

    assert metadata == {
        "policy_type": "fastwam",
        "input_features": {"observation.state": [32]},
        "output_features": {"action": [14]},
    }


def test_fastwam_checkpoint_uses_the_standard_lerobot_factory(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "config.json").write_text(
        '{"type": "fastwam", "input_features": {}}', encoding="utf-8"
    )
    selected_types: list[str] = []

    class Config:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, _path):
            return cls()

    class Policy:
        @classmethod
        def from_pretrained(cls, _path, *, config):
            assert config.device == "cpu"
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    torch = ModuleType("torch")
    torch.device = lambda value: value  # type: ignore[attr-defined]
    configs = ModuleType("lerobot.configs")
    configs.PreTrainedConfig = Config  # type: ignore[attr-defined]
    factory = ModuleType("lerobot.policies.factory")

    def get_policy_class(policy_type: str):
        selected_types.append(policy_type)
        return Policy

    factory.get_policy_class = get_policy_class  # type: ignore[attr-defined]
    factory.make_pre_post_processors = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: (lambda value: value, lambda value: value)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setattr(executor_module, "register_policy_processors", lambda _: None)

    runtime = executor_module._load_direct_policy_runtime(
        str(tmp_path), "cpu", {"offline": True}
    )

    assert runtime.policy_type == "fastwam"
    assert selected_types == ["fastwam"]


def test_checkpoint_config_loads_local_and_remote_paths(tmp_path, monkeypatch) -> None:
    local = tmp_path / "local"
    local.mkdir()
    config_path = local / "config.json"
    config_path.write_text('{"type": "act"}', encoding="utf-8")

    assert load_checkpoint_config(str(local)) == ({"type": "act"}, str(config_path))
    assert load_checkpoint_config(str(config_path)) == (
        {"type": "act"},
        str(config_path),
    )

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *_args: str(config_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert load_checkpoint_config("org/policy")[0] == {"type": "act"}


def test_processor_registration_is_optional(monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(checkpoint_module.importlib.util, "find_spec", lambda _: None)
    assert register_policy_processors("act") is None

    monkeypatch.setattr(
        checkpoint_module.importlib.util, "find_spec", lambda _: object()
    )
    monkeypatch.setattr(
        checkpoint_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    assert register_policy_processors("fastwam") == (
        "lerobot.policies.fastwam.processor_fastwam"
    )
    assert imported == ["lerobot.policies.fastwam.processor_fastwam"]

    def missing(_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(checkpoint_module.importlib.util, "find_spec", missing)
    assert register_policy_processors("missing") is None


def test_checkpoint_cli_reports_factory_support(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"type": "fastwam"}', encoding="utf-8")
    factory = ModuleType("lerobot.policies.factory")
    factory.get_policy_class = lambda policy_type: policy_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setattr(sys, "argv", ["checkpoint", "--policy-path", str(config)])

    checkpoint_module.main()

    assert '"policy_type": "fastwam"' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "checkpoint",
            "--policy-path",
            str(config),
            "--expected-type",
            "act",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        checkpoint_module.main()


@pytest.mark.parametrize("raw_config", [{}, {"type": "unknown_policy"}])
def test_checkpoint_cli_rejects_missing_or_unregistered_type(
    raw_config, tmp_path, monkeypatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw_config), encoding="utf-8")
    factory = ModuleType("lerobot.policies.factory")

    def unsupported(_policy_type):
        raise ValueError("not registered")

    factory.get_policy_class = unsupported  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setattr(sys, "argv", ["checkpoint", "--policy-path", str(config)])

    with pytest.raises(SystemExit, match="2"):
        checkpoint_module.main()
