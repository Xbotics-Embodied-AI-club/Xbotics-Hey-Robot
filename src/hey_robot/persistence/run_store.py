"""File-backed run artifacts for skill worker execution."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from hey_robot.protocol import ArtifactRef, ImageRef
from hey_robot.protocol.messages import from_payload, to_payload
from hey_robot.skills.models import SkillCommand, SkillEvent, SkillResult

logger = logging.getLogger(__name__)


class RunArtifactStore(Protocol):
    def pin_image(self, ref: ImageRef, *, namespace: str) -> ImageRef: ...

    def put_json_artifact(
        self,
        payload: Any,
        *,
        artifact_type: str,
        role: str | None = None,
        robot_id: str | None = None,
        frame_id: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef: ...

    def put_npz_artifact(
        self,
        payload: Any,
        *,
        artifact_type: str,
        role: str | None = None,
        robot_id: str | None = None,
        frame_id: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef: ...


class RunStore(Protocol):
    """Durable source of truth for submitted skill runs."""

    def record_submission(self, command: SkillCommand) -> None: ...

    def submission(self, run_id: str) -> SkillCommand | None: ...

    def append_event(self, event: SkillEvent) -> SkillEvent: ...

    def latest_event(self, run_id: str) -> SkillEvent | None: ...

    def result(self, run_id: str) -> SkillResult | None: ...

    def recent(self, limit: int = 50) -> tuple[SkillEvent, ...]: ...


class FileRunStore:
    """Append skill run events and terminal result artifacts under one directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        artifact_store: RunArtifactStore | None = None,
        inline_result_limit_bytes: int = 64 * 1024,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._artifact_store = artifact_store
        self._inline_result_limit_bytes = max(1024, inline_result_limit_bytes)

    def record_submission(self, command: SkillCommand) -> None:
        path = self._run_dir(command.run_id) / "command.json"
        if path.exists():
            existing = self.submission(command.run_id)
            if existing != command:
                raise ValueError(
                    f"run_id {command.run_id!r} was submitted with a different command"
                )
            return
        self._write_json(path, to_payload(command))

    def submission(self, run_id: str) -> SkillCommand | None:
        path = self._run_dir(run_id, create=False) / "command.json"
        if not path.exists():
            return None
        return from_payload(SkillCommand, json.loads(path.read_text(encoding="utf-8")))

    def append_event(self, event: SkillEvent) -> SkillEvent:
        run_dir = self._run_dir(event.run_id)
        path = run_dir / "events.jsonl"
        events = self._read_events(event.run_id, repair_trailing_corruption=True)
        if events and event.sequence <= events[-1].sequence:
            if event == events[-1] or self._matches_externalized_event(
                events[-1], event
            ):
                return events[-1]
            raise ValueError(
                f"run {event.run_id!r} event sequence must increase beyond "
                f"{events[-1].sequence}"
            )
        event = self._externalize_terminal_result(event)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_payload(event), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if event.result is not None:
            self.write_result(event.run_id, event.result)
        return event

    def write_result(self, run_id: str, result: SkillResult) -> None:
        result_path = self._run_dir(run_id) / "result.json"
        self._write_json(result_path, to_payload(result))

    def latest_event(self, run_id: str) -> SkillEvent | None:
        events = self.events(run_id)
        return events[-1] if events else None

    def events(self, run_id: str) -> tuple[SkillEvent, ...]:
        return self._read_events(run_id, repair_trailing_corruption=False)

    def _read_events(
        self, run_id: str, *, repair_trailing_corruption: bool
    ) -> tuple[SkillEvent, ...]:
        path = self._run_dir(run_id, create=False) / "events.jsonl"
        if not path.exists():
            return ()
        records = [
            (line_number, line)
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]
        events: list[SkillEvent] = []
        valid_lines: list[str] = []
        for record_index, (line_number, line) in enumerate(records):
            try:
                event = from_payload(SkillEvent, json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if record_index != len(records) - 1:
                    raise ValueError(
                        f"run {run_id!r} events artifact is corrupt at line {line_number}"
                    ) from exc
                logger.warning(
                    "run %s has a corrupt trailing events.jsonl record; "
                    "preserving %d valid events",
                    run_id,
                    len(events),
                )
                if repair_trailing_corruption:
                    content = "".join(f"{item}\n" for item in valid_lines)
                    self._write_text(path, content)
                break
            events.append(event)
            valid_lines.append(line)
        return tuple(events)

    def result(self, run_id: str) -> SkillResult | None:
        path = self._run_dir(run_id, create=False) / "result.json"
        if not path.exists():
            return None
        return from_payload(SkillResult, json.loads(path.read_text(encoding="utf-8")))

    def recent(self, limit: int = 50) -> tuple[SkillEvent, ...]:
        candidates = sorted(
            (path for path in self._root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        events: list[SkillEvent] = []
        for path in candidates:
            event = self.latest_event(path.name)
            if event is not None:
                events.append(event)
            if len(events) >= max(0, limit):
                break
        return tuple(events)

    def _externalize_terminal_result(self, event: SkillEvent) -> SkillEvent:
        result = event.result
        if result is None or self._artifact_store is None:
            return event
        pinned_observations: tuple[ImageRef, ...] = ()
        if result.observations:
            pinned: list[ImageRef] = []
            for observation in result.observations:
                try:
                    pinned.append(
                        self._artifact_store.pin_image(
                            observation, namespace=event.run_id
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "run %s could not pin observation %s: %s",
                        event.run_id,
                        observation.uri,
                        exc,
                    )
                    pinned.append(observation)
            pinned_observations = tuple(pinned)
            result = replace(result, observations=pinned_observations)
            event = replace(event, result=result)
        payload = dict(result.data)
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode()
        except TypeError:
            encoded = b""
        trace_keys = {"steps", "vla_history", "vln_history", "model_outputs"}
        if len(encoded) <= self._inline_result_limit_bytes and not (
            trace_keys & result.data.keys()
        ):
            return event
        artifact_metadata = {"run_id": event.run_id, "skill": event.name}
        if encoded:
            artifact = self._artifact_store.put_json_artifact(
                payload,
                artifact_type="skill_run_trace",
                role="execution_trace",
                robot_id=event.envelope.robot_id,
                name=event.run_id,
                metadata=artifact_metadata,
            )
        else:
            artifact = self._artifact_store.put_npz_artifact(
                payload,
                artifact_type="skill_run_trace",
                role="execution_trace",
                robot_id=event.envelope.robot_id,
                name=event.run_id,
                metadata=artifact_metadata,
            )
        compact = {
            key: value
            for key, value in result.data.items()
            if key
            in {
                "termination_reason",
                "option_completed",
                "subgoal_succeeded",
                "before_frame_id",
                "after_frame_id",
                "steps_used",
                "command",
            }
        }
        steps = result.data.get("steps")
        if isinstance(steps, list):
            compact["steps_executed"] = len(steps)
        compact["execution_trace"] = artifact.uri
        persisted = replace(
            result,
            data=compact,
            artifacts=(*result.artifacts, artifact),
        )
        return replace(event, result=persisted)

    @staticmethod
    def _matches_externalized_event(stored: SkillEvent, incoming: SkillEvent) -> bool:
        stored_result = stored.result
        incoming_result = incoming.result
        return bool(
            stored.sequence == incoming.sequence
            and stored.name == incoming.name
            and stored.phase == incoming.phase
            and stored_result is not None
            and incoming_result is not None
            and stored_result.success == incoming_result.success
            and stored_result.status == incoming_result.status
            and stored_result.summary == incoming_result.summary
            and stored_result.failure_mode == incoming_result.failure_mode
            and stored_result.error == incoming_result.error
            and (
                any(
                    artifact.role == "execution_trace"
                    for artifact in stored_result.artifacts
                )
                or (
                    len(stored_result.observations) == len(incoming_result.observations)
                    and all(
                        stored_ref.sha256 == incoming_ref.sha256
                        for stored_ref, incoming_ref in zip(
                            stored_result.observations,
                            incoming_result.observations,
                            strict=True,
                        )
                    )
                )
            )
        )

    def _run_dir(self, run_id: str, *, create: bool = True) -> Path:
        path = self._root / run_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        FileRunStore._write_text(
            path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
