from __future__ import annotations

import numpy as np

from hey_robot.robocasa_backend import task_progress


class _NestedProgress:
    def __init__(self) -> None:
        self.ready = np.bool_(True)


class _ProgressEnvironment:
    def __init__(self) -> None:
        self.washed_time = np.int64(4)
        self.nested = _NestedProgress()
        self.unsupported = object()

    def _check_success(self) -> bool:
        counter = np.int64(self.washed_time + 1)
        score = np.float64(0.87654)
        flags = (self.nested.ready, False)
        ignored = {"unsupported": self.unsupported}
        return bool(counter and score and flags and ignored)


def test_extract_task_progress_reads_referenced_attributes_and_locals() -> None:
    task_progress._path_cache.clear()

    progress = task_progress.extract_task_progress(_ProgressEnvironment())

    assert progress == {
        "washed_time": 4,
        "nested_ready": True,
        "counter": 5,
        "score": 0.8765,
        "flags": [True, False],
    }


def test_progress_conversion_and_uninspectable_task_are_safe() -> None:
    class _NoCheck:
        pass

    assert task_progress._json_value(
        np.array([np.int64(2), "ok", object()], dtype=object)
    ) == [2, "ok", None]
    assert task_progress._paths_for(_NoCheck) == []
    assert task_progress.extract_task_progress(object()) == {}


def test_progress_ignores_missing_attributes_and_check_success_exceptions() -> None:
    class _Broken:
        def _check_success(self) -> bool:
            marker = "before-error"
            del marker
            raise RuntimeError("simulator unavailable")

    task_progress._path_cache.clear()
    assert task_progress.extract_task_progress(_Broken()) == {}
