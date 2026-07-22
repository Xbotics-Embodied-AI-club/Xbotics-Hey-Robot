"""Canonical RoboCasa task and observation contract.

This is runtime code shared by the remote simulator, policy adapter, and
evaluation clients.  Benchmark manifests consume this contract; they do not
own it.
"""

from pathlib import Path
from typing import Any

DEFAULT_SPLIT = "target"
DEFAULT_REGISTRIES = ("lightwheel",)

ALLOWED_TASKS = frozenset(
    {
        "CloseFridge",
        "OpenDrawer",
        "PickPlaceCounterToCabinet",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
        "KettleBoiling",
        "PrepareCoffee",
        "LoadDishwasher",
        "SetUpCuttingStation",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "ArrangeTea",
        "CategorizeCondiments",
        "MakeIceLemonade",
        "RecycleBottlesByType",
        "WashFruitColander",
        "WeighIngredients",
    }
)

CAMERA_RENAME_MAP = {
    "observation.images.robot0_agentview_left": "observation.images.camera1",
    "observation.images.robot0_agentview_right": "observation.images.camera2",
    "observation.images.robot0_eye_in_hand": "observation.images.camera3",
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen RoboCasa task manifest."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load the RoboCasa manifest") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RoboCasa manifest must be a mapping")
    suites = payload.get("suites")
    if not isinstance(suites, dict):
        raise ValueError("RoboCasa manifest must define suites")
    tasks = {
        task
        for values in suites.values()
        if isinstance(values, list)
        for task in values
        if isinstance(task, str)
    }
    if not tasks or not tasks.issubset(ALLOWED_TASKS):
        raise ValueError("manifest contains tasks outside the backend allowlist")
    return {
        "version": str(payload.get("version") or ""),
        "split": str(payload.get("split") or DEFAULT_SPLIT),
        "registries": tuple(
            str(item) for item in payload.get("registries", DEFAULT_REGISTRIES)
        ),
        "suites": {str(name): list(values) for name, values in suites.items()},
        "tasks": sorted(tasks),
    }


from pathlib import Path
