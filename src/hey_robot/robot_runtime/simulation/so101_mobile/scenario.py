from __future__ import annotations

from dataclasses import dataclass

DOCK_OBJECTS: tuple[str, ...] = ("cat_wand",)

WAND_ALIASES: dict[str, tuple[str, ...]] = {
    "cat_wand": ("cat_wand", "wand", "逗猫棒", "棒", "玩具棒", "toy"),
}

DOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "wand_dock": ("wand_dock", "dock", "坞", "底座", "充电坞", "holder"),
}


@dataclass(frozen=True)
class DockScenario:
    id: str
    object_names: tuple[str, ...]
    dock_attach_body: str = "base_link"


SCENARIOS: dict[str, DockScenario] = {
    "dock_default": DockScenario("dock_default", DOCK_OBJECTS),
}
