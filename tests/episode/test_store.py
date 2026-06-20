from __future__ import annotations

from pathlib import Path

from hey_robot.episode.scope import EpisodeScope
from hey_robot.episode.store import JsonlEpisodeStore
from hey_robot.protocol import AgentReply, Envelope, UserTurn


def test_ensure_creates_new_episode(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep1", scope, ["alias1"])
    assert "ep1" in store.list_episodes()
    assert store.resolve("alias1") == "ep1"


def test_ensure_updates_existing_episode_aliases(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep1", scope, ["alias_a"])
    store.ensure("ep1", scope, ["alias_b", "alias_a"])
    assert store.resolve("alias_a") == "ep1"
    assert store.resolve("alias_b") == "ep1"
    assert len(store.list_episodes()) == 1


def test_append_and_read_history(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    env = Envelope(trace_id="tr1", episode_id="ep1")
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep1", scope, [])
    store.append_user_turn("ep1", UserTurn(envelope=env, text="hello"))
    store.append_agent_reply("ep1", AgentReply(envelope=env.child(), text="hi"))

    history = store.history("ep1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"
    assert history[1].content == "hi"


def test_history_skips_corrupted_lines(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    env = Envelope(trace_id="tr1", episode_id="ep1")
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep1", scope, [])
    store.append_user_turn("ep1", UserTurn(envelope=env, text="good"))
    # Write a corrupted line directly
    key = store.resolve("ep1")
    path = tmp_path / f"{_sanitize_for_test(key)}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("not valid json!!!\n")
    store.append_agent_reply("ep1", AgentReply(envelope=env.child(), text="also good"))
    history = store.history("ep1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_list_episodes_after_ensure(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep_a", scope, [])
    store.ensure("ep_b", scope, [])
    episodes = store.list_episodes()
    assert sorted(episodes) == ["ep_a", "ep_b"]


def test_resolve_unknown_returns_input(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    assert store.resolve("nonexistent") == "nonexistent"


def test_history_nonexistent_episode(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    assert store.history("no_such_episode") == []


def test_load_aliases_handles_corrupted_file(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text("not valid json {{{", encoding="utf-8")
    store = JsonlEpisodeStore(tmp_path)
    assert store.resolve("any") == "any"


def test_load_meta_handles_corrupted_file(tmp_path: Path) -> None:
    store = JsonlEpisodeStore(tmp_path)
    key = store.resolve("ep_corrupt")
    meta_path = tmp_path / f"{_sanitize_for_test(key)}.meta.json"
    meta_path.write_text("corrupted {{{", encoding="utf-8")
    scope = EpisodeScope(agent_id="main", dimensions=(), values={})
    store.ensure("ep_corrupt", scope, [])
    assert "ep_corrupt" in store.list_episodes()


def _sanitize_for_test(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
