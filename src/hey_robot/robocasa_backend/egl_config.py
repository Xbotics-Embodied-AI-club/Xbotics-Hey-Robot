"""Select a working headless EGL device before MuJoCo is imported."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_NVIDIA_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
_MESA_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
_PROBE = (
    "from mujoco.egl import GLContext; c=GLContext(32,32); c.make_current(); c.free()"
)


def configure_headless_egl() -> dict[str, str]:
    """Prefer NVIDIA EGL and fall back to the first working Mesa device."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return _selection()
    candidates: list[tuple[Path, str | None]] = [(_NVIDIA_VENDOR, None)]
    candidates.extend((_MESA_VENDOR, str(index)) for index in range(8))
    for vendor, device in candidates:
        if not vendor.is_file():
            continue
        env = dict(os.environ)
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
        if device is None:
            env.pop("MUJOCO_EGL_DEVICE_ID", None)
        else:
            env["MUJOCO_EGL_DEVICE_ID"] = device
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _PROBE],
            env=env,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
            if device is not None:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = device
            return _selection()
    raise RuntimeError("no usable NVIDIA or Mesa EGL device was found")


def _selection() -> dict[str, str]:
    return {
        "vendor": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "glvnd-auto"),
        "device": os.environ.get("MUJOCO_EGL_DEVICE_ID", "auto"),
    }
