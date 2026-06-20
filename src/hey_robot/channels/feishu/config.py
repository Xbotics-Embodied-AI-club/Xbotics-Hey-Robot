from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class FeishuChannelConfig:
    app_id: str = ""
    app_id_env: str = "FEISHU_APP_ID"
    app_secret: str = ""
    app_secret_env: str = "FEISHU_APP_SECRET"  # noqa: S105
    encrypt_key: str = ""
    encrypt_key_env: str = "FEISHU_ENCRYPT_KEY"
    verification_token: str = ""
    verification_token_env: str = "FEISHU_VERIFICATION_TOKEN"  # noqa: S105
    sender_id: str = "feishu-user"
    allow_from: list[str] = field(default_factory=list)
    react_emoji: str = "THUMBSUP"
    group_policy: Literal["open", "mention"] = "mention"
    reply_to_message: bool = True
    domain: Literal["feishu", "lark"] = "feishu"
    media_root: str = "runtime/media/feishu"

    @property
    def resolved_app_id(self) -> str:
        return self.app_id or os.environ.get(self.app_id_env, "")

    @property
    def resolved_app_secret(self) -> str:
        return self.app_secret or os.environ.get(self.app_secret_env, "")

    @property
    def resolved_encrypt_key(self) -> str:
        return self.encrypt_key or os.environ.get(self.encrypt_key_env, "")

    @property
    def resolved_verification_token(self) -> str:
        return self.verification_token or os.environ.get(
            self.verification_token_env, ""
        )


def feishu_config_from_settings(settings: dict[str, Any]) -> FeishuChannelConfig:
    return FeishuChannelConfig(
        app_id=str(settings.get("app_id", settings.get("appId", "")) or ""),
        app_id_env=str(
            settings.get("app_id_env", settings.get("appIdEnv", "FEISHU_APP_ID"))
            or "FEISHU_APP_ID"
        ),
        app_secret=str(settings.get("app_secret", settings.get("appSecret", "")) or ""),
        app_secret_env=str(
            settings.get(
                "app_secret_env", settings.get("appSecretEnv", "FEISHU_APP_SECRET")
            )
            or "FEISHU_APP_SECRET"
        ),
        encrypt_key=str(
            settings.get("encrypt_key", settings.get("encryptKey", "")) or ""
        ),
        encrypt_key_env=str(
            settings.get(
                "encrypt_key_env", settings.get("encryptKeyEnv", "FEISHU_ENCRYPT_KEY")
            )
            or "FEISHU_ENCRYPT_KEY"
        ),
        verification_token=str(
            settings.get("verification_token", settings.get("verificationToken", ""))
            or ""
        ),
        verification_token_env=str(
            settings.get(
                "verification_token_env",
                settings.get("verificationTokenEnv", "FEISHU_VERIFICATION_TOKEN"),
            )
            or "FEISHU_VERIFICATION_TOKEN"
        ),
        sender_id=str(
            settings.get("sender_id", settings.get("senderId", "feishu-user"))
            or "feishu-user"
        ),
        allow_from=[
            str(item)
            for item in settings.get("allow_from", settings.get("allowFrom", [])) or []
        ],
        react_emoji=str(
            settings.get("react_emoji", settings.get("reactEmoji", "THUMBSUP"))
            or "THUMBSUP"
        ),
        group_policy=str(
            settings.get("group_policy", settings.get("groupPolicy", "mention"))
            or "mention"
        ),  # type: ignore[arg-type]
        reply_to_message=bool(
            settings.get("reply_to_message", settings.get("replyToMessage", True))
        ),
        domain=str(settings.get("domain", "feishu") or "feishu"),  # type: ignore[arg-type]
        media_root=str(
            settings.get(
                "media_root", settings.get("mediaRoot", "runtime/media/feishu")
            )
        ),
    )
