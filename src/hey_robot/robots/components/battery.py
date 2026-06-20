from __future__ import annotations

from dataclasses import dataclass, field

from hey_robot.robots.components.servo_bus import ServoBus


@dataclass(frozen=True)
class ServoBusBatteryConfig:
    enabled: bool = True
    servo_ids: list[int] = field(default_factory=lambda: [1])
    full_voltage: float = 12.6
    low_voltage: float = 10.5
    critical_voltage: float = 9.5
    min_voltage: float = 9.0


@dataclass(frozen=True)
class BatteryState:
    ok: bool
    voltage: float | None
    percentage: float | None
    status: str
    temperature: int | None
    servo_id: int | None
    issue: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "voltage": self.voltage,
            "percentage": self.percentage,
            "status": self.status,
            "temperature": self.temperature,
            "servo_id": self.servo_id,
            "issue": self.issue,
        }


class ServoBusBattery:
    """Battery monitor based on voltage telemetry exposed by a shared servo bus."""

    def __init__(self, bus: ServoBus, config: ServoBusBatteryConfig) -> None:
        self.bus = bus
        self.config = config

    def read(self) -> BatteryState:
        if not self.config.enabled:
            return BatteryState(True, None, None, "disabled", None, None)
        if not self.bus.connected:
            return BatteryState(
                False, None, None, "unknown", None, None, "servo bus is not connected"
            )
        issues: list[str] = []
        for servo_id in self.config.servo_ids:
            state = self.bus.read_state(int(servo_id))
            if state.voltage is None:
                issues.append(f"servo {servo_id}: voltage unavailable")
                continue
            percentage = _percentage(
                state.voltage,
                minimum=self.config.min_voltage,
                maximum=self.config.full_voltage,
            )
            return BatteryState(
                ok=True,
                voltage=state.voltage,
                percentage=percentage,
                status=_status(
                    state.voltage,
                    low=self.config.low_voltage,
                    critical=self.config.critical_voltage,
                ),
                temperature=state.temperature,
                servo_id=state.servo_id,
            )
        return BatteryState(
            False,
            None,
            None,
            "unknown",
            None,
            None,
            "; ".join(issues) or "no servo ids configured",
        )


def _percentage(voltage: float, *, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return round(
        max(0.0, min(1.0, (voltage - minimum) / (maximum - minimum))) * 100.0, 1
    )


def _status(voltage: float, *, low: float, critical: float) -> str:
    if voltage <= critical:
        return "critical"
    if voltage <= low:
        return "low"
    return "normal"
