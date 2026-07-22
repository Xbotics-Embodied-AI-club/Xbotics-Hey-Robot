"""Validate a RoboCasa policy checkpoint before starting a simulation rollout."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

EXPECTED_INPUTS = {
    "observation.images.robot0_agentview_left": (3, 256, 256),
    "observation.images.robot0_agentview_right": (3, 256, 256),
    "observation.images.robot0_eye_in_hand": (3, 256, 256),
}
EXPECTED_OUTPUTS: dict[str, tuple[int, ...]] = {"action": (12,)}
CAMERA_ALIASES = (
    (
        "observation.images.robot0_agentview_left",
        "observation.images.camera1",
    ),
    (
        "observation.images.robot0_agentview_right",
        "observation.images.camera2",
    ),
    (
        "observation.images.robot0_eye_in_hand",
        "observation.images.camera3",
    ),
)
SUPPORTED_STATE_SHAPES = {(6,), (16,)}


def validate_feature_contract(config: Any) -> dict[str, Any]:
    policy_type = _config_value(config, "type")
    inputs = _feature_shapes(_config_value(config, "input_features"))
    outputs = _feature_shapes(_config_value(config, "output_features"))
    errors = [
        *(_camera_shape_errors(inputs)),
        *(_state_shape_errors(inputs)),
        *(_shape_errors("output", outputs, EXPECTED_OUTPUTS)),
    ]
    return {
        "valid": not errors,
        "policy_type": policy_type,
        "input_features": {key: list(shape) for key, shape in inputs.items()},
        "output_features": {key: list(shape) for key, shape in outputs.items()},
        "errors": errors,
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


def _load_raw_config(policy_path: str) -> tuple[dict[str, Any], str]:
    path = Path(policy_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(policy_path, "config.json")
    return json.loads(Path(config_path).read_text(encoding="utf-8")), config_path


def register_policy_processors(policy_type: str) -> str | None:
    """Import a policy's processor module before loading serialized pipelines."""
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
    """Resolve PI052 tokenizer repos to concrete snapshots in offline mode."""
    if policy_type != "pi052" or not (
        os.environ.get("ROBOCASA_OFFLINE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
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


def _shape_errors(
    kind: str,
    actual: dict[str, tuple[int, ...]],
    expected: dict[str, tuple[int, ...]],
) -> list[str]:
    errors = []
    for key, shape in expected.items():
        if key not in actual:
            errors.append(f"missing {kind} feature {key}")
        elif actual[key] != shape:
            errors.append(
                f"{kind} feature {key} has shape {actual[key]}, expected {shape}"
            )
    return errors


def _camera_shape_errors(actual: dict[str, tuple[int, ...]]) -> list[str]:
    errors = []
    for aliases in CAMERA_ALIASES:
        matches = {key: actual[key] for key in aliases if key in actual}
        if not matches:
            errors.append(f"missing input camera feature; expected one of {aliases}")
            continue
        bad = {
            key: shape
            for key, shape in matches.items()
            if shape != EXPECTED_INPUTS[aliases[0]]
        }
        for key, shape in bad.items():
            errors.append(
                f"input feature {key} has shape {shape}, expected {EXPECTED_INPUTS[aliases[0]]}"
            )
    return errors


def _state_shape_errors(actual: dict[str, tuple[int, ...]]) -> list[str]:
    shape = actual.get("observation.state")
    if shape is None:
        return ["missing input feature observation.state"]
    if shape not in SUPPORTED_STATE_SHAPES:
        supported = ", ".join(str(item) for item in sorted(SUPPORTED_STATE_SHAPES))
        return [
            f"input feature observation.state has shape {shape}, expected one of {supported}"
        ]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the RoboCasa policy type and feature contract"
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument(
        "--expected-type",
        help="Optional assertion for diagnostics; normally inferred from the checkpoint.",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Also download and deserialize the full policy and processors.",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _parser().parse_args()

    raw_config, config_path = _load_raw_config(args.policy_path)
    result = {
        "policy_path": args.policy_path,
        "config_path": config_path,
        **validate_feature_contract(raw_config),
    }
    policy_type = result["policy_type"]
    if args.expected_type and policy_type != args.expected_type:
        result["valid"] = False
        result["errors"].append(
            f"policy type is {policy_type!r}, expected {args.expected_type!r}"
        )

    from lerobot.policies.factory import get_policy_class

    try:
        policy_class = get_policy_class(policy_type)
    except (TypeError, ValueError) as exc:
        result["runtime_supported"] = False
        result["runtime_error"] = str(exc)
        policy_class = None
    else:
        result["runtime_supported"] = True

    if args.load_weights and result["valid"] and policy_class is not None:
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(args.policy_path)
        config.device = args.device
        policy = policy_class.from_pretrained(args.policy_path, config=config)
        policy.to(args.device)
        policy.eval()
        result["processor_module"] = register_policy_processors(policy_type)
        try:
            make_pre_post_processors(
                config,
                pretrained_path=args.policy_path,
                **offline_processor_overrides(policy_type),
            )
        except Exception:
            if policy_type != "pi052" or not getattr(
                config, "enable_fast_action_loss", False
            ):
                raise
            config.enable_fast_action_loss = False
            make_pre_post_processors(config)
        result["weights_loaded"] = True
        result["device"] = args.device
    elif args.load_weights and policy_class is None:
        result["valid"] = False
        result["errors"].append(
            f"policy type {policy_type!r} is not registered in this LeRobot runtime"
        )

    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
