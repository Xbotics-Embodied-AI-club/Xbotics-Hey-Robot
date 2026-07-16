from pathlib import Path

from hey_robot.robot_runtime.simulation.mujoco_logging import (
    configure_mujoco_warning_logging,
)


class _MuJoCo:
    def __init__(self) -> None:
        self.warning_callback = None

    def set_mju_user_warning(self, callback) -> None:
        self.warning_callback = callback


def test_mujoco_warnings_are_routed_to_deployment_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    module = _MuJoCo()

    path = configure_mujoco_warning_logging("sim-test", mujoco_module=module)
    module.warning_callback("simulation is unstable")

    assert path == Path("logs/sim-test/mujoco.log")
    assert path.read_text(encoding="utf-8").endswith("simulation is unstable\n")
    assert not Path("MUJOCO_LOG.TXT").exists()
