from __future__ import annotations

from hey_robot.episode.scope import (
    VALID_DIMENSIONS,
    EpisodeScope,
    _dimension_value,
    allocate_episode,
    scope_to_dict,
)
from hey_robot.protocol import Envelope


class TestEpisodeScope:
    def test_key_is_deterministic(self) -> None:
        a = EpisodeScope(
            agent_id="main",
            dimensions=("channel", "robot"),
            values={"channel": "slack", "robot": "r1"},
        )
        b = EpisodeScope(
            agent_id="main",
            dimensions=("channel", "robot"),
            values={"channel": "slack", "robot": "r1"},
        )
        assert a.key() == b.key()

    def test_key_differs_by_agent(self) -> None:
        a = EpisodeScope(agent_id="main", dimensions=(), values={})
        b = EpisodeScope(agent_id="other", dimensions=(), values={})
        assert a.key() != b.key()

    def test_key_differs_by_dimension_values(self) -> None:
        a = EpisodeScope(
            agent_id="main", dimensions=("channel",), values={"channel": "slack"}
        )
        b = EpisodeScope(
            agent_id="main", dimensions=("channel",), values={"channel": "feishu"}
        )
        assert a.key() != b.key()

    def test_key_differs_by_channel_value(self) -> None:
        a = EpisodeScope(
            agent_id="main", dimensions=("channel",), values={"channel": "slack"}
        )
        b = EpisodeScope(
            agent_id="main", dimensions=("channel",), values={"channel": "feishu"}
        )
        assert a.key() != b.key()

    def test_canonical_signature(self) -> None:
        scope = EpisodeScope(
            agent_id="main",
            dimensions=("channel", "robot"),
            values={"channel": "slack", "robot": "r1"},
        )
        sig = scope.canonical_signature()
        assert "v1" in sig
        assert "agent=main" in sig
        assert "channel=slack" in sig
        assert "robot=r1" in sig

    def test_aliases_with_no_dimensions(self) -> None:
        scope = EpisodeScope(agent_id="main", dimensions=(), values={})
        assert scope.aliases() == ["agent:main"]

    def test_aliases_with_dimensions(self) -> None:
        scope = EpisodeScope(
            agent_id="main",
            dimensions=("channel", "robot"),
            values={"channel": "slack", "robot": "r1"},
        )
        assert scope.aliases() == ["agent:main:channel=slack:robot=r1"]

    def test_aliases_with_missing_values(self) -> None:
        scope = EpisodeScope(agent_id="main", dimensions=("channel",), values={})
        assert scope.aliases() == ["agent:main:channel="]

    def test_key_format(self) -> None:
        scope = EpisodeScope(agent_id="main", dimensions=(), values={})
        key = scope.key()
        assert key.startswith("ep_v1_")
        assert len(key) > len("ep_v1_")


class TestAllocateEpisode:
    def test_basic_allocation(self) -> None:
        env = Envelope(
            channel="slack",
            account_id="acct1",
            chat_id="C123",
            sender_id="U456",
            robot_id="r1",
        )
        alloc = allocate_episode(
            env,
            agent_id="main",
            dimensions=["channel", "account", "chat", "sender", "robot"],
        )
        assert alloc.episode_id.startswith("ep_v1_")
        assert len(alloc.aliases) == 1

    def test_deduplicates_dimensions(self) -> None:
        env = Envelope(channel="slack")
        alloc = allocate_episode(
            env, agent_id="main", dimensions=["channel", "CHANNEL", "Channel "]
        )
        assert alloc.scope.dimensions == ("channel",)

    def test_filters_invalid_dimensions(self) -> None:
        env = Envelope(channel="slack")
        alloc = allocate_episode(
            env, agent_id="main", dimensions=["invalid", "channel", "unknown"]
        )
        assert "invalid" not in alloc.scope.dimensions
        assert "unknown" not in alloc.scope.dimensions
        assert "channel" in alloc.scope.dimensions

    def test_empty_dimensions(self) -> None:
        env = Envelope(channel="slack")
        alloc = allocate_episode(env, agent_id="main", dimensions=[])
        assert alloc.scope.dimensions == ()

    def test_same_dimensions_any_order_produces_same_key(self) -> None:
        """allocate_episode should produce the same key regardless of dimension order."""
        env = Envelope(channel="slack", robot_id="r1")
        a = allocate_episode(env, agent_id="main", dimensions=["channel", "robot"])
        b = allocate_episode(env, agent_id="main", dimensions=["robot", "channel"])
        assert a.episode_id == b.episode_id, (
            f"Dimension order should not affect episode key: {a.episode_id} != {b.episode_id}"
        )

    def test_same_scope_produces_same_key(self) -> None:
        env1 = Envelope(channel="slack", robot_id="r1")
        env2 = Envelope(channel="slack", robot_id="r1")
        a = allocate_episode(env1, agent_id="main", dimensions=["channel", "robot"])
        b = allocate_episode(env2, agent_id="main", dimensions=["channel", "robot"])
        assert a.episode_id == b.episode_id


class TestDimensionValue:
    def test_channel_value(self) -> None:
        env = Envelope(channel="slack")
        assert _dimension_value(env, "channel", "main") == "slack"

    def test_channel_when_none(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "channel", "main") == "unknown"

    def test_account_value(self) -> None:
        env = Envelope(account_id="acct1")
        assert _dimension_value(env, "account", "main") == "acct1"

    def test_account_when_none(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "account", "main") == "default"

    def test_user_value(self) -> None:
        env = Envelope(user_id="user-123", sender_id="sender-123")
        assert _dimension_value(env, "user", "main") == "user-123"

    def test_user_fallback_to_sender(self) -> None:
        env = Envelope(sender_id="sender-123")
        assert _dimension_value(env, "user", "main") == "sender-123"

    def test_chat_value(self) -> None:
        env = Envelope(chat_id="C123", chat_type="group")
        assert _dimension_value(env, "chat", "main") == "group:C123"

    def test_chat_fallback_to_sender(self) -> None:
        env = Envelope(sender_id="U456")
        assert _dimension_value(env, "chat", "main") == "direct:U456"

    def test_chat_when_all_none(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "chat", "main") == "direct:direct"

    def test_sender_value(self) -> None:
        env = Envelope(sender_id="U456")
        assert _dimension_value(env, "sender", "main") == "U456"

    def test_sender_when_none(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "sender", "main") == "anonymous"

    def test_robot_value(self) -> None:
        env = Envelope(robot_id="r1")
        assert _dimension_value(env, "robot", "main") == "r1"

    def test_robot_when_none(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "robot", "main") == "none"

    def test_agent_value(self) -> None:
        env = Envelope()
        assert _dimension_value(env, "agent", "main") == "main"


class TestScopeToDict:
    def test_round_trip_via_asdict(self) -> None:
        from dataclasses import asdict

        scope = EpisodeScope(
            agent_id="main",
            dimensions=("channel", "robot"),
            values={"channel": "slack", "robot": "r1"},
        )
        d = scope_to_dict(scope)
        assert d == asdict(scope)
        assert d["version"] == "v1"
        assert d["agent_id"] == "main"
        assert d["dimensions"] == ("channel", "robot")
        assert d["values"] == {"channel": "slack", "robot": "r1"}


class TestValidDimensions:
    def test_contains_expected(self) -> None:
        assert "channel" in VALID_DIMENSIONS
        assert "account" in VALID_DIMENSIONS
        assert "user" in VALID_DIMENSIONS
        assert "chat" in VALID_DIMENSIONS
        assert "sender" in VALID_DIMENSIONS
        assert "robot" in VALID_DIMENSIONS
        assert "agent" in VALID_DIMENSIONS
