"""Validate mounted Habitat assets before starting a costly EGL episode."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.habitat3.worker.profiles import HabitatProfile, get_profile


@dataclass(frozen=True)
class AssetCheck:
    name: str
    path: Path
    required: bool = True


def required_assets(data_root: Path, profile: HabitatProfile) -> tuple[AssetCheck, ...]:
    dataset = (
        data_root / "datasets" / "hssd" / "rearrange" / "val" / profile.dataset_filename
    )
    return (
        AssetCheck(
            "HSSD scene dataset config",
            data_root
            / "scene_datasets"
            / "hssd-hab"
            / "hssd-hab.scene_dataset_config.json",
        ),
        AssetCheck("HSSD stages", data_root / "scene_datasets" / "hssd-hab" / "stages"),
        AssetCheck(
            "HSSD uncluttered scene instances",
            data_root / "scene_datasets" / "hssd-hab" / "scenes-uncluttered",
        ),
        AssetCheck("Habitat rearrange dataset", dataset),
        AssetCheck(
            "Spot URDF",
            data_root / "robots" / "hab_spot_arm" / "urdf" / "hab_spot_arm.urdf",
        ),
        AssetCheck("Humanoid data", data_root / "humanoids" / "humanoid_data"),
    )


def report(data_root: str | Path, profile_name: str) -> dict[str, object]:
    root = Path(data_root)
    profile = get_profile(profile_name)
    checks = required_assets(root, profile)
    missing = [str(check.path) for check in checks if not check.path.exists()]
    return {
        "profile": profile.name,
        "data_root": str(root),
        "ok": not missing,
        "missing": missing,
        "checks": [
            {"name": check.name, "path": str(check.path), "exists": check.path.exists()}
            for check in checks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/opt/habitat/data")
    parser.add_argument("--profile", default="habitat3_social_spot_human_oracle")
    args = parser.parse_args()
    result = report(args.data_root, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
