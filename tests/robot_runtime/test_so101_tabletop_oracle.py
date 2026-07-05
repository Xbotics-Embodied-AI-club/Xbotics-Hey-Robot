from __future__ import annotations

import numpy as np
import pytest

from hey_robot.robot_runtime.simulation.so101_tabletop.arm import So101ArmKernel
from hey_robot.robot_runtime.simulation.so101_tabletop.oracle import TabletopOracle
from hey_robot.robot_runtime.simulation.so101_tabletop.session import (
    So101TabletopSession,
)


@pytest.fixture
def oracle_stack() -> tuple[So101TabletopSession, So101ArmKernel, TabletopOracle]:
    session = So101TabletopSession()
    session.connect()
    arm = So101ArmKernel(session)
    arm.bind()
    oracle = TabletopOracle(session, arm)
    yield session, arm, oracle
    session.close()


def test_oracle_detects_names_aliases_and_all(oracle_stack) -> None:
    _session, _arm, oracle = oracle_stack
    assert [item.label for item in oracle.detect("banana")] == ["banana"]
    assert [item.label for item in oracle.detect("杯子")] == ["mug"]
    assert len(oracle.detect("all")) == 6
    assert oracle.detect("laptop") == []


def test_oracle_track_matches_mujoco(oracle_stack) -> None:
    _session, arm, oracle = oracle_stack
    tracked = oracle.track(oracle.detect("banana"))
    assert len(tracked) == 1
    assert tracked[0].pose is not None
    assert tracked[0].pose.as_list() == pytest.approx(
        arm.get_object_positions()["banana"], abs=0.01
    )


def test_oracle_camera_and_depth_contract(oracle_stack) -> None:
    _session, _arm, oracle = oracle_stack
    frame = oracle.color_frame()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    depth = oracle.depth_frame()
    assert depth.shape == (480, 640)
    assert np.count_nonzero(depth) == 0
    assert oracle.source == "mujoco_ground_truth"
    assert oracle.simulation_only is True
