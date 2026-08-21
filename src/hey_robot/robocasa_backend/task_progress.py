"""Structured task-progress extraction for RoboCasa tasks.

The RoboCasa gym env only surfaces a binary ``is_success`` in its ``info``
dict. The intermediate predicates that actually gate success (``success_time``,
``washed_time``, ``kettle_on_site``, ``burner_on``, ``washed_loc``, ...) live as
attributes of the task env and as locals computed inside ``_check_success``.

This module recovers them without changing environment state:

1. parse ``_check_success`` source for ``self.<attr>`` / dotted paths, then read
   those attributes live;
2. trace one read-only ``_check_success()`` call to capture the locals computed
   in its frame.

The result is a JSON-safe dict the agent can read to verify sub-goal progress
precisely, instead of asking a VLM via ``inspect_scene``.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import sys
from typing import Any

import numpy as np

_SELF_PATH_RE = re.compile(r"self\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")

#: Parsed attribute paths, cached per task class so source is inspected once.
_path_cache: dict[type[Any], list[str]] = {}


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar, or ``None`` when unsupported."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    if isinstance(value, str):
        return value
    return None


def _json_value(value: Any) -> Any:
    """Convert an extracted value to a JSON-safe representation."""
    scalar = _json_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    return None


def _paths_for(task_type: type[Any]) -> list[str]:
    if task_type in _path_cache:
        return _path_cache[task_type]
    paths: list[str] = []
    try:
        src = inspect.getsource(task_type._check_success)
        paths = sorted(set(_SELF_PATH_RE.findall(src)))
    except Exception:
        paths = []
    _path_cache[task_type] = paths
    return paths


def _attribute_progress(env: Any) -> dict[str, Any]:
    """Read the ``self.<attr>`` paths referenced by ``_check_success``."""
    prog: dict[str, Any] = {}
    for path in _paths_for(type(env)):
        obj = env
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if not ok:
            continue
        value = _json_value(obj)
        if value is not None:
            prog[path.replace(".", "_")] = value
    return prog


def _local_progress(env: Any) -> dict[str, Any]:
    """Trace one read-only ``_check_success`` call and capture its locals."""
    code = type(env)._check_success.__code__
    captured: dict[str, Any] = {}

    def _tracer(frame, event, _arg):
        if event == "call" and frame.f_code is code:

            def _local(f, e, _a):
                if e == "return":
                    captured.update(f.f_locals)
                return _local

            return _local
        return None

    old = sys.gettrace()
    try:
        sys.settrace(_tracer)
        with contextlib.suppress(Exception):
            env._check_success()
    finally:
        sys.settrace(old)

    prog: dict[str, Any] = {}
    for key, value in captured.items():
        if key == "self":
            continue
        converted = _json_value(value)
        if converted is not None:
            prog[key] = converted
    return prog


def extract_task_progress(env: Any) -> dict[str, Any]:
    """Return a JSON-safe progress dict for the given task env.

    ``env`` must be the task env exposing ``_check_success``. For the lerobot
    wrapper this is ``wrapper._env.env``.
    """
    prog: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        prog = _attribute_progress(env)
    with contextlib.suppress(Exception):
        prog.update(_local_progress(env))
    return prog
