from __future__ import annotations

import asyncio
from pathlib import Path

from hey_robot.config import NotificationSpec
from hey_robot.episode import JsonlEpisodeStore
from hey_robot.notifications import (
    Notification,
    NotificationPolicy,
    NotificationService,
    NotificationTarget,
)
from hey_robot.protocol import AgentReply, Envelope, UserTurn


def test_notification_service_routes_explicit_target_and_dedupes(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    service = NotificationService(
        JsonlEpisodeStore(tmp_path / "episodes"),
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=60,
    )
    origin = Envelope(
        channel="web",
        chat_id="chat-a",
        sender_id="user-a",
        message_id="msg-1",
        reply_to_id="parent-1",
        episode_id="ep-1",
        robot_id="mock0",
        agent_id="main",
        trace_id="tr-1",
    )
    notification = Notification(
        kind="task_watchdog",
        title="Watchdog",
        body="skill heartbeat is stale",
        severity="warning",
        origin_envelope=origin,
        target=NotificationTarget(mode="explicit", reply_to_current=True),
        metadata={"source": "watchdog"},
    )

    first = asyncio.run(service.publish(notification))
    second = asyncio.run(service.publish(notification))

    assert first is True
    assert second is False
    assert published[0].text == "Watchdog\nskill heartbeat is stale"
    assert published[0].envelope.message_id == "msg-1"
    assert published[0].envelope.reply_to_id == "parent-1"
    assert published[0].metadata["notification"] is True
    assert published[0].metadata["severity"] == "warning"


def test_notification_service_falls_back_to_episode_history_and_handles_invalid_rows(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    episodes = JsonlEpisodeStore(tmp_path / "episodes")
    service = NotificationService(
        episodes, lambda reply: _collect_reply(published, reply), dedupe_window_sec=0
    )

    episode_id = "ep-2"
    history_turn = UserTurn(
        envelope=Envelope(
            channel="feishu",
            chat_id="oc_chat_1",
            sender_id="ou_user",
            episode_id=episode_id,
            robot_id="mock0",
            agent_id="main",
            trace_id="tr-history",
        ),
        text="hello",
    )
    episodes.append_user_turn(episode_id, history_turn)
    broken_path = tmp_path / "episodes" / f"{episode_id}.jsonl"
    broken_path.write_text(
        broken_path.read_text(encoding="utf-8")
        + '{"role":"assistant","content":"bad","timestamp":1,"payload":{"envelope":"bad"}}\n'
        + '{"role":"assistant","content":"bad2","timestamp":2,"payload":{"envelope":{"unknown":1}}}\n',
        encoding="utf-8",
    )

    published_ok = asyncio.run(
        service.publish(
            Notification(
                kind="task_update",
                body="task finished",
                episode_id=episode_id,
                robot_id="mock0",
                agent_id="main",
                trace_id="tr-notify",
            )
        )
    )
    published_missing = asyncio.run(
        service.publish(Notification(kind="task_update", body="missing target"))
    )

    assert published_ok is True
    assert published_missing is False
    assert published[0].envelope.channel == "feishu"
    assert published[0].envelope.chat_id == "oc_chat_1"
    assert published[0].envelope.episode_id == episode_id
    assert published[0].metadata["notification_kind"] == "task_update"


def test_notification_service_uses_origin_envelope_for_episode_mode(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    service = NotificationService(
        JsonlEpisodeStore(tmp_path / "episodes"),
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=0,
    )
    origin = Envelope(
        channel="cli",
        chat_id="local",
        sender_id="operator",
        episode_id="ep-origin",
        trace_id="tr-origin",
    )

    result = asyncio.run(
        service.publish(
            Notification(
                kind="task_update",
                body="done",
                origin_envelope=origin,
                robot_id="mock0",
                agent_id="main",
            )
        )
    )

    assert result is True
    assert published[0].envelope.channel == "cli"
    assert published[0].envelope.chat_id == "local"


def test_notification_service_returns_false_when_episode_history_has_no_routable_envelope(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    episodes = JsonlEpisodeStore(tmp_path / "episodes")
    path = tmp_path / "episodes" / "ep-empty.jsonl"
    path.write_text(
        '{"role":"assistant","content":"bad","timestamp":1,"payload":{"envelope":{"unknown":1}}}\n',
        encoding="utf-8",
    )
    service = NotificationService(
        episodes, lambda reply: _collect_reply(published, reply), dedupe_window_sec=0
    )

    result = asyncio.run(
        service.publish(
            Notification(kind="task_update", body="done", episode_id="ep-empty")
        )
    )

    assert result is False
    assert published == []


def test_notification_service_explicit_target_requires_channel_and_chat(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    service = NotificationService(
        JsonlEpisodeStore(tmp_path / "episodes"),
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=0,
    )

    result = asyncio.run(
        service.publish(
            Notification(
                kind="operator_alert",
                body="needs operator",
                target=NotificationTarget(mode="explicit", channel="web"),
            )
        )
    )

    assert result is False
    assert published == []


def test_notification_service_applies_policy_severity_and_fanout(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    policy = NotificationPolicy(
        NotificationSpec(
            defaults={"channels": ["web"]},
            channels={"feishu": {"chat_id": "oc_chat_1", "sender_id": "ou_user"}},
            kinds={
                "task_watchdog": {"severity": "critical", "channels": ["web", "feishu"]}
            },
        )
    )
    service = NotificationService(
        JsonlEpisodeStore(tmp_path / "episodes"),
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=0,
        policy=policy,
    )
    origin = Envelope(
        channel="web",
        chat_id="chat-a",
        sender_id="user-a",
        episode_id="ep-1",
        trace_id="tr-1",
    )

    result = asyncio.run(
        service.publish(
            Notification(
                kind="task_watchdog",
                body="heartbeat stale",
                severity="warning",
                origin_envelope=origin,
                metadata={"event": "task_watchdog"},
            )
        )
    )

    assert result is True
    assert [reply.envelope.channel for reply in published] == ["web", "feishu"]
    assert all(reply.metadata["severity"] == "critical" for reply in published)
    assert published[1].envelope.chat_id == "oc_chat_1"


def test_notification_service_routes_to_identity_linked_channel_targets(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    episodes = JsonlEpisodeStore(tmp_path / "episodes")
    episodes.append_user_turn(
        "ep-user",
        UserTurn(
            envelope=Envelope(
                channel="web",
                chat_id="chat-web",
                sender_id="web-user",
                episode_id="ep-user",
                user_id="owner",
            ),
            text="start task",
        ),
    )
    policy = NotificationPolicy(
        NotificationSpec(
            defaults={"channels": ["web", "feishu"]},
            channels={"web": {}, "feishu": {}},
            kinds={"task_update": {"channels": ["web", "feishu"]}},
        )
    )
    service = NotificationService(
        episodes,
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=0,
        policy=policy,
        linked_target_provider=lambda user_id, channel: (
            [
                Envelope(
                    channel="feishu",
                    chat_id="oc_chat_1",
                    sender_id="ou_user",
                    user_id=user_id,
                )
            ]
            if user_id == "owner" and channel == "feishu"
            else []
        ),
    )

    result = asyncio.run(
        service.publish(
            Notification(
                kind="task_update",
                body="task is still running",
                episode_id="ep-user",
            )
        )
    )

    assert result is True
    assert [reply.envelope.channel for reply in published] == ["web", "feishu"]
    assert published[1].envelope.chat_id == "oc_chat_1"
    assert published[1].envelope.user_id == "owner"


def test_notification_service_does_not_fallback_to_episode_when_configured_targets_are_unavailable(
    tmp_path: Path,
) -> None:
    published: list[AgentReply] = []
    episodes = JsonlEpisodeStore(tmp_path / "episodes")
    policy = NotificationPolicy(
        NotificationSpec(
            defaults={"channels": ["web"]},
            channels={"web": {}, "feishu": {"chat_id": ""}},
            kinds={
                "task_watchdog": {"severity": "critical", "channels": ["web", "feishu"]}
            },
        )
    )
    service = NotificationService(
        episodes,
        lambda reply: _collect_reply(published, reply),
        dedupe_window_sec=0,
        policy=policy,
    )
    episode_id = "ep-voice"
    episodes.append_user_turn(
        episode_id,
        UserTurn(
            envelope=Envelope(
                channel="voice", chat_id="xlerobot-voice", episode_id=episode_id
            ),
            text="你好",
        ),
    )

    result = asyncio.run(
        service.publish(
            Notification(
                kind="task_watchdog",
                body="robot status stale for 34.6s",
                episode_id=episode_id,
                robot_id="xlerobot",
            )
        )
    )

    assert result is False
    assert published == []


async def _collect_reply(bucket: list[AgentReply], reply: AgentReply) -> None:
    bucket.append(reply)
