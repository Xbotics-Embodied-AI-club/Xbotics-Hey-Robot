from hey_robot.robot_backends.mock._camera import _CameraMixin
from hey_robot.robot_backends.mock._diagnostics import _DiagnosticsMixin
from hey_robot.robot_backends.mock._world import _WorldMixin
from hey_robot.robot_backends.mock.driver import _MockRobotDriverBase


class MockRobotDriver(
    _MockRobotDriverBase, _WorldMixin, _CameraMixin, _DiagnosticsMixin
):
    """Deterministic in-memory robot backend for tests and local development."""
