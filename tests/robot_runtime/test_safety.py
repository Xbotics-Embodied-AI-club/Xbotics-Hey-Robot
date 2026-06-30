from __future__ import annotations

from hey_robot.protocol import Envelope, RobotAction
from hey_robot.robot_runtime.base import RobotCapabilities, RobotHealth
from hey_robot.robot_runtime.safety import RobotSafetySupervisor


def _action(
    values: list[float],
    *,
    skill_name: str | None = None,
) -> RobotAction:
    metadata = {"skill": {"name": skill_name}} if skill_name is not None else {}
    return RobotAction(
        envelope=Envelope(robot_id="xlerobot"),
        values=values,
        metadata=metadata,
    )


def _capabilities(
    *,
    action_dimensions: int | None = 3,
    metadata: dict | None = None,
) -> RobotCapabilities:
    return RobotCapabilities(
        robot_id="xlerobot",
        driver_type="xlerobot",
        action_dimensions=action_dimensions,
        metadata=metadata or {},
    )


def _health(*, online: bool = True, metrics: dict | None = None) -> RobotHealth:
    return RobotHealth(
        robot_id="xlerobot",
        online=online,
        state="ready" if online else "offline",
        metrics=metrics or {},
    )


def test_robot_safety_supervisor_rejects_offline_and_active_safety_flags() -> None:
    supervisor = RobotSafetySupervisor()

    offline = supervisor.evaluate_action(
        _action([0.0, 0.0, 0.0]),
        capabilities=_capabilities(),
        health=_health(online=False),
    )
    stopped = supervisor.evaluate_action(
        _action([0.0, 0.0, 0.0]),
        capabilities=_capabilities(),
        health=_health(metrics={"emergency_stop": True, "protective_stop": True}),
    )

    assert offline.allowed is False
    assert offline.reason == "robot xlerobot is offline"
    assert stopped.allowed is False
    assert stopped.reason == "active safety flags: emergency_stop,protective_stop"


def test_robot_safety_supervisor_allows_stop_motion_during_critical_battery_only() -> (
    None
):
    supervisor = RobotSafetySupervisor()
    health = _health(
        metrics={"battery": {"status": "critical", "voltage": 9.2}},
    )

    normal_motion = supervisor.evaluate_action(
        _action([0.0, 0.0, 0.0], skill_name="move_base"),
        capabilities=_capabilities(),
        health=health,
    )
    stop_motion = supervisor.evaluate_action(
        _action([0.0, 0.0, 0.0], skill_name="stop_motion"),
        capabilities=_capabilities(),
        health=health,
    )
    malformed_skill_metadata = supervisor.evaluate_action(
        RobotAction(
            envelope=Envelope(robot_id="xlerobot"),
            values=[0.0, 0.0, 0.0],
            metadata={"skill": "stop_motion"},
        ),
        capabilities=_capabilities(),
        health=health,
    )

    assert normal_motion.allowed is False
    assert normal_motion.reason == "battery critical voltage=9.2"
    assert normal_motion.metadata == {"battery": {"status": "critical", "voltage": 9.2}}
    assert stop_motion.allowed is True
    assert malformed_skill_metadata.allowed is False


def test_robot_safety_supervisor_validates_dimensions_abs_limit_and_bounds() -> None:
    supervisor = RobotSafetySupervisor()
    health = _health()

    dimension = supervisor.evaluate_action(
        _action([0.0, 0.0]),
        capabilities=_capabilities(action_dimensions=3),
        health=health,
    )
    max_abs = supervisor.evaluate_action(
        _action([0.0, -1.2, 0.0]),
        capabilities=_capabilities(
            metadata={"safety": {"max_abs_action": 1.0}},
        ),
        health=health,
    )
    bounded = supervisor.evaluate_action(
        _action([0.0, 0.8, 1.1]),
        capabilities=_capabilities(
            metadata={"action_bounds": [-1.0, 1.0]},
        ),
        health=health,
    )
    allowed = supervisor.evaluate_action(
        _action([0.2, -0.3, 0.4]),
        capabilities=_capabilities(
            metadata={"safety": {"max_abs_action": 1.0, "action_bounds": [-1.0, 1.0]}},
        ),
        health=health,
    )

    assert dimension.reason == "action dimension mismatch: expected 3, got 2"
    assert max_abs.reason == "action exceeds max_abs_action=1.0"
    assert bounded.reason == "action outside bounds [-1.0, 1.0]"
    assert allowed.allowed is True
