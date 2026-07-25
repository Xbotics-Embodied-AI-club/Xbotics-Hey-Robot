from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBOT_PACKAGE_NAMES = {
    "robocasa_backend",
    "robot_api",
    "robot_backends",
    "robot_hardware",
    "robot_media",
    "robot_runtime",
    "robot_transport",
}


def test_mock_driver_does_not_import_native_or_remote_backend_dependencies() -> None:
    code = """
import sys
from hey_robot.config import DeploymentConfig
from hey_robot.robot_runtime.manager import RobotManager

config = DeploymentConfig.from_dict({"robots": {"mock0": {"type": "mock"}}})
RobotManager(config)
for forbidden in ("grpc", "scservo_sdk"):
    assert forbidden not in sys.modules, forbidden
"""
    completed = subprocess.run(  # noqa: S603 - interpreter and code are test constants
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_robot_runtime_package_has_no_compatibility_exports() -> None:
    import hey_robot.robot_runtime as runtime_package

    assert not hasattr(runtime_package, "RobotService")
    assert not hasattr(runtime_package, "SO101Driver")
    assert not hasattr(runtime_package, "LeKiwiDriver")


def test_robot_test_roots_mirror_robot_source_package_roots() -> None:
    source_roots = {
        path.name
        for path in (ROOT / "src" / "hey_robot").iterdir()
        if path.is_dir() and path.name in ROBOT_PACKAGE_NAMES
    }
    test_roots = {
        path.name
        for path in (ROOT / "tests").iterdir()
        if path.is_dir() and path.name in ROBOT_PACKAGE_NAMES
    }

    assert source_roots == ROBOT_PACKAGE_NAMES
    assert test_roots == source_roots
    assert not (ROOT / "tests" / "robot_runtime" / "media").exists()
    assert (ROOT / "tests" / "robot_backends" / "mock").is_dir()
