"""File-backed run artifacts for skill worker execution."""

from __future__ import annotations

import json
from pathlib import Path

from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills.models import SkillEvent, SkillResult


class FileRunStore:
    """Append skill run events and terminal result artifacts under one directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: SkillEvent) -> None:
        run_dir = self._run_dir(event.run_id)
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_payload(event), ensure_ascii=False))
            handle.write("\n")
        if event.result is not None:
            self.write_result(event.run_id, event.result)

    def write_result(self, run_id: str, result: SkillResult) -> None:
        result_path = self._run_dir(run_id) / "result.json"
        result_path.write_text(
            json.dumps(to_payload(result), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def latest_event(self, run_id: str) -> SkillEvent | None:
        events = self.events(run_id)
        return events[-1] if events else None

    def events(self, run_id: str) -> tuple[SkillEvent, ...]:
        path = self._run_dir(run_id, create=False) / "events.jsonl"
        if not path.exists():
            return ()
        return tuple(
            from_payload(SkillEvent, json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def result(self, run_id: str) -> SkillResult | None:
        path = self._run_dir(run_id, create=False) / "result.json"
        if not path.exists():
            return None
        return from_payload(SkillResult, json.loads(path.read_text(encoding="utf-8")))

    def _run_dir(self, run_id: str, *, create: bool = True) -> Path:
        path = self._root / run_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
