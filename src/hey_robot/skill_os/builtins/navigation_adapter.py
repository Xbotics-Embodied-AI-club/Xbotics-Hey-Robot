from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrimitiveCommand:
    primitive: str
    arguments: dict[str, Any]
    reason: str


def planner_output_to_primitive(
    planner: dict[str, Any],
    *,
    image_width: int = 640,
    _image_height: int = 480,
    center_band_ratio: float = 0.25,
    min_turn_deg: float = 5.0,
    max_turn_deg: float = 15.0,
    forward_step_cm: float = 15.0,
) -> PrimitiveCommand:
    if bool(planner.get("stop")) or planner.get("mode") == "stop":
        return PrimitiveCommand("stop_motion", {}, "planner requested stop")

    heading = planner.get("heading_deg")
    if isinstance(heading, (int, float)):
        if abs(float(heading)) < min_turn_deg:
            return PrimitiveCommand(
                "move_base",
                {"direction": "forward", "distance_cm": forward_step_cm},
                "heading is near center",
            )
        return PrimitiveCommand(
            "turn_base",
            {
                "direction": "right" if float(heading) > 0 else "left",
                "angle_deg": min(abs(float(heading)), max_turn_deg),
            },
            "planner returned heading",
        )

    pixel = planner.get("pixel_goal")
    if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
        x = float(pixel[0])
        center_x = image_width / 2.0
        half_band = image_width * center_band_ratio / 2.0
        offset = x - center_x
        if abs(offset) <= half_band:
            return PrimitiveCommand(
                "move_base",
                {"direction": "forward", "distance_cm": forward_step_cm},
                "pixel goal is centered",
            )
        turn = min_turn_deg + (max_turn_deg - min_turn_deg) * min(
            abs(offset) / max(center_x, 1.0), 1.0
        )
        return PrimitiveCommand(
            "turn_base",
            {"direction": "right" if offset > 0 else "left", "angle_deg": turn},
            "pixel goal is off center",
        )

    raise ValueError("planner output does not contain stop, heading_deg, or pixel_goal")
