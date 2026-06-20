from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from hey_robot.episode.scope import EpisodeScope, scope_to_dict
from hey_robot.protocol.messages import AgentReply, UserTurn, to_payload


@dataclass(frozen=True)
class EpisodeRecord:
    role: str
    content: str
    timestamp: float
    payload: dict


class EpisodeStore(Protocol):
    def ensure(
        self, episode_id: str, scope: EpisodeScope, aliases: list[str]
    ) -> None: ...

    def append_user_turn(self, episode_id: str, turn: UserTurn) -> None: ...

    def append_agent_reply(self, episode_id: str, reply: AgentReply) -> None: ...

    def history(self, episode_id: str, *, limit: int = 50) -> list[EpisodeRecord]: ...


@dataclass
class _Meta:
    key: str
    scope: dict
    aliases: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    count: int = 0


class JsonlEpisodeStore:
    """Append-only episode store with sidecar metadata.

    This mirrors the deployable shape used by gateway systems: a canonical key
    owns durable history and aliases can be resolved without coupling callers to
    storage paths.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.alias_path = self.root / "aliases.json"
        self._aliases = self._load_aliases()

    def ensure(self, episode_id: str, scope: EpisodeScope, aliases: list[str]) -> None:
        key = self.resolve(episode_id)
        meta = self._read_meta(key)
        if meta is None:
            meta = _Meta(key=key, scope=scope_to_dict(scope), aliases=aliases)
        else:
            meta.aliases = sorted(set(meta.aliases) | set(aliases))
            meta.scope = scope_to_dict(scope)
            meta.updated_at = time.time()
        self._write_meta(meta)
        for alias in aliases:
            self._aliases[alias] = key
        self._write_aliases()

    def append_user_turn(self, episode_id: str, turn: UserTurn) -> None:
        self._append(
            episode_id,
            EpisodeRecord("user", turn.text, turn.envelope.timestamp, to_payload(turn)),
        )

    def append_agent_reply(self, episode_id: str, reply: AgentReply) -> None:
        self._append(
            episode_id,
            EpisodeRecord(
                "assistant", reply.text, reply.envelope.timestamp, to_payload(reply)
            ),
        )

    def history(self, episode_id: str, *, limit: int = 50) -> list[EpisodeRecord]:
        path = self._jsonl_path(self.resolve(episode_id))
        if not path.exists():
            return []
        rows: list[EpisodeRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(EpisodeRecord(**data))
        return rows[-limit:]

    def list_episodes(self) -> list[str]:
        episodes = [
            meta_path.stem.removesuffix(".meta")
            for meta_path in self.root.glob("*.meta.json")
        ]
        return sorted(set(episodes))

    def resolve(self, episode_id_or_alias: str) -> str:
        return self._aliases.get(episode_id_or_alias, episode_id_or_alias)

    def _append(self, episode_id: str, record: EpisodeRecord) -> None:
        key = self.resolve(episode_id)
        with self._jsonl_path(key).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        meta = self._read_meta(key)
        if meta is not None:
            meta.count += 1
            meta.updated_at = time.time()
            self._write_meta(meta)

    def _load_aliases(self) -> dict[str, str]:
        if not self.alias_path.exists():
            return {}
        try:
            with self.alias_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _write_aliases(self) -> None:
        tmp = self.alias_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                self._aliases, handle, ensure_ascii=False, indent=2, sort_keys=True
            )
        tmp.replace(self.alias_path)

    def _read_meta(self, key: str) -> _Meta | None:
        path = self._meta_path(key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return _Meta(**json.load(handle))
        except (OSError, TypeError, json.JSONDecodeError):
            return None

    def _write_meta(self, meta: _Meta) -> None:
        path = self._meta_path(meta.key)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                asdict(meta), handle, ensure_ascii=False, indent=2, sort_keys=True
            )
        tmp.replace(path)

    def _jsonl_path(self, key: str) -> Path:
        return self.root / f"{_sanitize(key)}.jsonl"

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{_sanitize(key)}.meta.json"


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
