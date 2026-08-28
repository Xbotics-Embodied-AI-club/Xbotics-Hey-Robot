"""Episode-scoped state shared by RoboCasa analytic and VLA skills.

RPent keeps one ``RoboCasaPrimitives`` instance for an episode.  Hey Robot
creates a fresh ``SkillContext`` (and run id) for every tool call, so the small
piece of cross-call state that affects VLA history must be keyed by the task
instead of by the individual skill run.
"""

from __future__ import annotations

_vla_desync: dict[tuple[str, str], bool] = {}
_reposition_required: dict[tuple[str, str], bool] = {}


def session_key(robot_id: str, task_id: str) -> tuple[str, str]:
    return robot_id, task_id


def mark_vla_desync(robot_id: str, task_id: str) -> None:
    """Record that a non-VLA environment step invalidated VLA frame history."""
    _vla_desync[session_key(robot_id, task_id)] = True


def consume_vla_desync(robot_id: str, task_id: str) -> bool:
    """Return/reset the RPent-style force-reset flag for the next VLA call.

    A previously unseen episode starts desynchronized, matching RPent's
    ``RoboCasaPrimitives.__init__``.
    """
    key = session_key(robot_id, task_id)
    desynchronized = _vla_desync.get(key, True)
    _vla_desync[key] = False
    return desynchronized


def set_reposition_required(robot_id: str, task_id: str, required: bool) -> None:
    """Persist the RPent retry guard reported by the local option runtime."""
    _reposition_required[session_key(robot_id, task_id)] = required


def reposition_required(robot_id: str, task_id: str) -> bool:
    """Return whether another VLA rollout must wait for a physical re-stance."""
    return _reposition_required.get(session_key(robot_id, task_id), False)


def mark_repositioned(robot_id: str, task_id: str) -> None:
    """Clear the retry guard after an arm/base positioning primitive ran."""
    _reposition_required[session_key(robot_id, task_id)] = False


def clear_session_state() -> None:
    """Test helper for deterministic isolation."""
    _vla_desync.clear()
    _reposition_required.clear()
