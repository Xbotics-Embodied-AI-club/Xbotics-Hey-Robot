"""LeRobot checkpoint discovery shared by every learned policy family."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_checkpoint_config(policy_path: str) -> tuple[dict[str, Any], str]:
    """Load ``config.json`` from a local checkpoint or Hugging Face repo."""
    path = Path(policy_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(policy_path, "config.json")
    return json.loads(Path(config_path).read_text(encoding="utf-8")), config_path


# Kept private for the executor; the public name describes the actual boundary.
_load_raw_config = load_checkpoint_config


def checkpoint_metadata(config: Any) -> dict[str, Any]:
    """Return family-neutral policy and feature metadata for diagnostics."""
    return {
        "policy_type": _config_value(config, "type"),
        "input_features": {
            key: list(shape)
            for key, shape in _feature_shapes(
                _config_value(config, "input_features")
            ).items()
        },
        "output_features": {
            key: list(shape)
            for key, shape in _feature_shapes(
                _config_value(config, "output_features")
            ).items()
        },
    }


def register_policy_processors(policy_type: str) -> str | None:
    """Import a policy's optional processor module before pipeline loading."""
    module_name = f"lerobot.policies.{policy_type}.processor_{policy_type}"
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    importlib.import_module(module_name)
    return module_name


def offline_processor_overrides(policy_type: str) -> dict[str, Any]:
    """Resolve policy-owned tokenizer repos from the local cache offline."""
    if policy_type != "pi052" or not (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    ):
        return {}
    from huggingface_hub import snapshot_download

    paligemma = snapshot_download("google/paligemma-3b-pt-224", local_files_only=True)
    action_tokenizer = snapshot_download(
        "lerobot/fast-action-tokenizer", local_files_only=True
    )
    return {
        "preprocessor_overrides": {
            "pi052_text_tokenizer": {"tokenizer_name": paligemma},
            "action_tokenizer_processor": {
                "action_tokenizer_name": action_tokenizer,
                "paligemma_tokenizer_name": paligemma,
            },
        }
    }


def _feature_shapes(features: dict[str, Any] | None) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(int(value) for value in _config_value(feature, "shape"))
        for key, feature in (features or {}).items()
    }


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot checkpoint and verify runtime registration"
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument(
        "--expected-type",
        help="Optional diagnostic assertion; normally inferred from the checkpoint.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    raw_config, config_path = load_checkpoint_config(args.policy_path)
    result = {
        "policy_path": args.policy_path,
        "config_path": config_path,
        **checkpoint_metadata(raw_config),
        "valid": True,
        "errors": [],
    }
    policy_type = str(result["policy_type"] or "")
    if not policy_type:
        result["valid"] = False
        result["errors"].append("checkpoint config has no policy type")
    elif args.expected_type and policy_type != args.expected_type:
        result["valid"] = False
        result["errors"].append(
            f"policy type is {policy_type!r}, expected {args.expected_type!r}"
        )

    if policy_type:
        from lerobot.policies.factory import get_policy_class

        try:
            get_policy_class(policy_type)
        except (TypeError, ValueError) as exc:
            result["runtime_supported"] = False
            result["runtime_error"] = str(exc)
        else:
            result["runtime_supported"] = True

    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["valid"] or not result.get("runtime_supported", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
