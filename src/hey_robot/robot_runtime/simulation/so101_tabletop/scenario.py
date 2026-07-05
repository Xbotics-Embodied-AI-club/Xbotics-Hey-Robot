from __future__ import annotations

from dataclasses import dataclass

TABLETOP_OBJECTS: tuple[str, ...] = (
    "banana",
    "mug",
    "bottle",
    "screwdriver",
    "duck",
    "lego",
)

OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "banana": ("banana", "香蕉", "黄色", "黄"),
    "mug": ("mug", "cup", "杯子", "杯", "马克杯", "红色", "红"),
    "bottle": ("bottle", "瓶子", "瓶", "蓝色", "蓝", "水瓶"),
    "screwdriver": ("screwdriver", "螺丝刀", "起子", "绿色", "绿"),
    "duck": ("duck", "鸭子", "小鸭", "橙色", "橙", "玩具"),
    "lego": ("lego", "积木", "乐高", "白色", "白", "方块"),
}

TRAY_REGION: tuple[float, float, float, float] = (0.20, 0.0, 0.50, 0.25)


@dataclass(frozen=True)
class TabletopScenario:
    id: str
    object_names: tuple[str, ...]
    place_region: tuple[float, float, float, float] | None = None


SCENARIOS: dict[str, TabletopScenario] = {
    "tabletop": TabletopScenario("tabletop", TABLETOP_OBJECTS),
    "tabletop_tray": TabletopScenario("tabletop_tray", TABLETOP_OBJECTS, TRAY_REGION),
}
