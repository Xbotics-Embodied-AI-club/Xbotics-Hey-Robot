"""Route MuJoCo native warnings into the project's diagnostic log tree."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_callback: Any = None


def configure_mujoco_warning_logging(
    deployment_id: str, *, mujoco_module: Any | None = None
) -> Path:
    """Install MuJoCo's global warning callback for one deployment.

    Without this callback MuJoCo writes ``MUJOCO_LOG.TXT`` to the process
    working directory. Diagnostic text belongs in ``logs/``, never ``runtime/``.
    """
    global _callback
    path = Path("logs") / deployment_id / "mujoco.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hey_robot.mujoco")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    for handler in list(logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    def warning(message: str) -> None:
        logger.warning(str(message))

    _callback = warning
    module: Any = mujoco_module
    if module is None:
        import mujoco

        module = mujoco
    module.set_mju_user_warning(_callback)
    return path
