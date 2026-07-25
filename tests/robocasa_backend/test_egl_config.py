from __future__ import annotations

import os
import subprocess

import pytest

from hey_robot.robocasa_backend.egl_config import configure_headless_egl


def test_egl_configuration_respects_explicit_device(monkeypatch) -> None:
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "3")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    selection = configure_headless_egl()

    assert selection == {"vendor": "glvnd-auto", "device": "3"}
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_egl_configuration_selects_first_working_vendor(tmp_path, monkeypatch) -> None:
    vendor = tmp_path / "nvidia.json"
    vendor.touch()
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.setattr("hey_robot.robocasa_backend.egl_config._NVIDIA_VENDOR", vendor)
    monkeypatch.setattr(
        "hey_robot.robocasa_backend.egl_config._MESA_VENDOR",
        tmp_path / "missing.json",
    )

    def probe_succeeds(*args: object, **_kwargs: object):
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(
        "hey_robot.robocasa_backend.egl_config.subprocess.run",
        probe_succeeds,
    )

    selection = configure_headless_egl()

    assert selection["vendor"] == str(vendor)
    assert selection["device"] == "auto"


def test_egl_configuration_fails_when_no_vendor_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.setattr(
        "hey_robot.robocasa_backend.egl_config._NVIDIA_VENDOR",
        tmp_path / "nvidia.json",
    )
    monkeypatch.setattr(
        "hey_robot.robocasa_backend.egl_config._MESA_VENDOR",
        tmp_path / "mesa.json",
    )

    with pytest.raises(RuntimeError, match="no usable NVIDIA or Mesa EGL"):
        configure_headless_egl()
