"""MuJoCo platform bootstrap helpers.

This module owns process-wide graphics backend selection.  Keeping it outside the
robot driver makes the driver's boundary about the robot protocol, not operating
system and EGL discovery.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from hey_robot.logging import HeyRobotLogger

logger = HeyRobotLogger(name="xlerobot_sim")

_EGL_CONTEXT: Any = None
_EGL_AVAILABLE: bool | None = None


def _test_egl_device_display() -> bool:
    """Probe EGL without importing MuJoCo in the runtime process."""
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mujoco.egl; "
                    "ctx=mujoco.egl.GLContext(64,64); "
                    "ctx.make_current(); ctx.free()"
                ),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _is_egl_available() -> bool:
    global _EGL_AVAILABLE
    if _EGL_AVAILABLE is None:
        _EGL_AVAILABLE = _test_egl_device_display()
    return _EGL_AVAILABLE


def configure_mujoco_gl_backend() -> str | None:
    configured = os.environ.get("MUJOCO_GL")
    if configured:
        if configured in {"egl", "osmesa"}:
            os.environ["PYOPENGL_PLATFORM"] = configured
        return configured
    system = platform.system()
    if system == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
        return "wgl"
    if system == "Linux" and not os.environ.get("DISPLAY"):
        if _is_egl_available():
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            return "egl"
        logger.info("EGL platform-device headless unavailable; falling back to osmesa")
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return "osmesa"
    return None


def ensure_render_context(width: int, height: int) -> None:
    """Create the process EGL context when the selected backend requires it."""
    global _EGL_CONTEXT
    if platform.system() != "Linux" or os.environ.get("MUJOCO_GL") != "egl":
        return
    if _EGL_CONTEXT is not None:
        return
    import mujoco.egl

    _EGL_CONTEXT = mujoco.egl.GLContext(width, height)
    _EGL_CONTEXT.make_current()


def resolve_mjcf_path(settings: dict[str, Any]) -> Path:
    raw = settings.get("mjcf_path") or settings.get("mjcf")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
    return Path.cwd() / "assets" / "scenes" / "home_scene.xml"
