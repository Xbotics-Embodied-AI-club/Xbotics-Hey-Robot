from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SCENE = (
    Path(__file__).resolve().parents[5]
    / "assets"
    / "robots"
    / "so101_tabletop"
    / "scene.xml"
)


def configure_mujoco_gl() -> str | None:
    configured = os.environ.get("MUJOCO_GL")
    if configured:
        return configured
    if platform.system() == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
        return "wgl"
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "egl"
        return "egl"
    return None


class So101TabletopSession:
    """Single owner of the SO101 tabletop MuJoCo state."""

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        *,
        render_width: int = 640,
        render_height: int = 480,
        viewer_enabled: bool = False,
    ) -> None:
        self.scene_path = Path(scene_path).expanduser().resolve()
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.viewer_enabled = bool(viewer_enabled)
        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self.viewer: Any = None

    @property
    def connected(self) -> bool:
        return self.model is not None and self.data is not None

    def connect(self) -> None:
        if self.connected:
            return
        configure_mujoco_gl()
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._settle()
        self.renderer = mujoco.Renderer(
            self.model, height=self.render_height, width=self.render_width
        )
        if self.viewer_enabled:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )

    def require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("SO101 tabletop simulation is not connected")

    def _settle(self, duration: float = 2.0) -> None:
        """Step the simulation to let free bodies settle under gravity."""
        dt = float(self.model.opt.timestep)
        steps = max(1, int(duration / dt))
        import mujoco

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.data = None
        self.model = None

    def reset(self) -> None:
        self.require_connected()
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._settle()

    def step(self, count: int = 1) -> None:
        self.require_connected()
        import mujoco

        for _ in range(max(0, int(count))):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def render(self, camera: str, *, bgr: bool = True) -> np.ndarray:
        self.require_connected()
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        if not bgr:
            return np.ascontiguousarray(rgb)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def id_for(self, object_type: Any, name: str) -> int:
        self.require_connected()
        import mujoco

        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id
