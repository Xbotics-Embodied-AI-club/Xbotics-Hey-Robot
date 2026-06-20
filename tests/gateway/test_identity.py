from __future__ import annotations

from hey_robot.config import IdentitySpec
from hey_robot.gateway.identity import IdentityResolver
from hey_robot.protocol import Envelope


def test_identity_resolver_known_channels_includes_static_dynamic_and_claimed_bindings(
    tmp_path,
) -> None:
    resolver = IdentityResolver(
        IdentitySpec(
            enabled=True,
            bindings={
                "web:sender:web-user": "owner",
                "voice:sender:voice-user": "owner",
            },
        ),
        state_path=tmp_path / "bindings.json",
    )

    pending = resolver.create_binding(
        Envelope(channel="web", sender_id="web-user", chat_id="chat-web"), ttl_sec=300.0
    )
    resolver.claim_binding(
        pending.code,
        Envelope(channel="feishu", sender_id="ou_user_1", chat_id="oc_chat_1"),
    )

    assert resolver.known_channels("owner") == ["feishu", "voice", "web"]
