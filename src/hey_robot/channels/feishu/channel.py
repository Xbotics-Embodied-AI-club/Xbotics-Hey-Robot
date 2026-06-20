from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

from hey_robot.channels.base import ChannelContext, InboundHandler
from hey_robot.channels.feishu.config import (
    FeishuChannelConfig,
    feishu_config_from_settings,
)
from hey_robot.channels.feishu.inbound import (
    MSG_TYPE_MAP,
    extract_post_content,
    extract_share_card_content,
    resolve_mentions,
)
from hey_robot.channels.feishu.presenter import (
    format_outbound_message,
    format_outbound_reply,
)
from hey_robot.channels.feishu.transport import FeishuTransport, build_media_ref
from hey_robot.events import RuntimeEvent
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import AgentReply, Envelope, MediaRef, UserTurn

logger = HeyRobotLogger(name="feishu")

FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None


class FeishuChannel:
    name = "feishu"
    _IMAGE_EXTS: ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    )
    _AUDIO_EXTS: ClassVar[frozenset[str]] = frozenset(
        {".ogg", ".mp3", ".wav", ".m4a", ".aac", ".opus"}
    )
    _VIDEO_EXTS: ClassVar[frozenset[str]] = frozenset(
        {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )

    def __init__(self, context: ChannelContext) -> None:
        self.context = context
        self.name = context.name
        self.config = feishu_config_from_settings(context.spec.settings)
        self._client: Any | None = None
        self._ws_client: Any | None = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: InboundHandler | None = None
        self._running = False
        self._stopped = asyncio.Event()
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._bot_open_id: str | None = None

    async def start(self, handler: InboundHandler) -> None:
        package = sys.modules.get("hey_robot.channels.feishu")
        available = bool(getattr(package, "FEISHU_AVAILABLE", FEISHU_AVAILABLE))
        if not available:
            raise ImportError(
                "FeishuChannel requires lark-oapi. Install `hey-robot[agent]`."
            )
        app_id = self.config.resolved_app_id
        app_secret = self.config.resolved_app_secret
        if not app_id or not app_secret:
            raise ValueError(
                "Feishu channel requires app_id/app_secret or *_env settings."
            )

        import lark_oapi as lark
        from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN

        self._handler = handler
        self._running = True
        self._stopped.clear()
        self._loop = asyncio.get_running_loop()

        domain = LARK_DOMAIN if self.config.domain == "lark" else FEISHU_DOMAIN
        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.INFO)
            .build()
        )
        builder = lark.EventDispatcherHandler.builder(
            self.config.resolved_encrypt_key,
            self.config.resolved_verification_token,
        ).register_p2_im_message_receive_v1(self._on_message_sync)
        self._ws_client = lark.ws.Client(
            app_id,
            app_secret,
            domain=domain,
            event_handler=builder.build(),
            log_level=lark.LogLevel.INFO,
        )

        def run_ws() -> None:
            lark_ws_client = importlib.import_module("lark_oapi.ws.client")

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            setattr(lark_ws_client, "loop", ws_loop)
            try:
                while self._running and self._ws_client is not None:
                    try:
                        self._ws_client.start()
                    except Exception as exc:
                        logger.warning(
                            f"feishu WebSocket 错误: {type(exc).__name__}: {exc}"
                        )
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True, name="feishu-ws")
        self._ws_thread.start()
        self._bot_open_id = await asyncio.get_running_loop().run_in_executor(
            None, self._fetch_bot_open_id
        )
        logger.info(
            f"gateway channel [{self.name}] feishu 就绪 domain={self.config.domain}"
        )

        await self._stopped.wait()

    async def send(self, reply: AgentReply) -> None:
        if self._client is None:
            logger.debug("feishu 发送跳过: client 未初始化")
            return
        chat_id = reply.envelope.chat_id or ""
        if not chat_id:
            logger.debug("feishu 发送跳过: reply envelope 无 chat_id")
            return
        receive_id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
        target_message_id = reply.envelope.message_id
        loop = asyncio.get_running_loop()

        for media in reply.media:
            path = Path(media.uri)
            if not await asyncio.to_thread(path.is_file):
                logger.warning(f"feishu media 文件不存在: {path}")
                continue
            suffix = path.suffix.lower()
            if suffix in self._IMAGE_EXTS:
                image_key = await loop.run_in_executor(
                    None, self._upload_image_sync, path
                )
                if image_key:
                    image_content = json.dumps(
                        {"image_key": image_key}, ensure_ascii=False
                    )
                    await asyncio.to_thread(
                        self._send_or_reply_sync,
                        receive_id_type,
                        chat_id,
                        "image",
                        image_content,
                        target_message_id,
                    )
            else:
                file_key = await loop.run_in_executor(
                    None, self._upload_file_sync, path
                )
                if file_key:
                    if suffix in self._AUDIO_EXTS:
                        msg_type = "audio"
                    elif suffix in self._VIDEO_EXTS:
                        msg_type = "media"
                    else:
                        msg_type = "file"
                    file_content = json.dumps(
                        {"file_key": file_key}, ensure_ascii=False
                    )
                    await asyncio.to_thread(
                        self._send_or_reply_sync,
                        receive_id_type,
                        chat_id,
                        msg_type,
                        file_content,
                        target_message_id,
                    )

        text = (reply.text or "").strip()
        if not text:
            return
        msg_type, content = self._format_outbound_reply(reply)
        await asyncio.to_thread(
            self._send_or_reply_sync,
            receive_id_type,
            chat_id,
            msg_type,
            content,
            target_message_id,
        )

    async def on_event(self, _event: RuntimeEvent) -> None:
        return None

    async def stop(self) -> None:
        self._running = False
        self._stopped.set()

    def _on_message_sync(self, data: Any) -> None:
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        try:
            if self._handler is None:
                return
            event = data.event
            message = event.message
            sender = event.sender
            if getattr(sender, "sender_type", None) == "bot":
                return

            message_id = str(getattr(message, "message_id", "") or "")
            if not message_id or message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            sender_id_obj = getattr(sender, "sender_id", None)
            sender_id = str(getattr(sender_id_obj, "open_id", None) or "unknown")
            if not self._is_allowed(sender_id):
                return

            chat_id = str(getattr(message, "chat_id", "") or "")
            chat_type = str(getattr(message, "chat_type", "group") or "group")
            msg_type = str(getattr(message, "message_type", "text") or "text")
            if chat_type == "group" and not self._is_group_message_for_bot(message):
                return

            try:
                content_json = json.loads(getattr(message, "content", "") or "{}")
            except json.JSONDecodeError:
                content_json = {}

            content_parts: list[str] = []
            media: list[MediaRef] = []

            if msg_type == "text":
                text = str(content_json.get("text") or "")
                if text:
                    content_parts.append(
                        resolve_mentions(text, getattr(message, "mentions", None))
                    )
            elif msg_type == "post":
                text, image_keys = extract_post_content(content_json)
                if text:
                    content_parts.append(text)
                for image_key in image_keys:
                    saved = await self._download_and_save_media(
                        "image", {"image_key": image_key}, message_id
                    )
                    if saved is not None:
                        media.append(saved)
                        content_parts.append(f"[image: {Path(saved.uri).name}]")
            elif msg_type in {"image", "audio", "file", "media"}:
                saved = await self._download_and_save_media(
                    msg_type, content_json, message_id
                )
                if saved is not None:
                    media.append(saved)
                    content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))
            elif msg_type in {
                "share_chat",
                "share_user",
                "interactive",
                "share_calendar_event",
                "system",
                "merge_forward",
            }:
                text = extract_share_card_content(content_json, msg_type)
                if text:
                    content_parts.append(text)
            else:
                content_parts.append(MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]"))

            text = "\n".join(part for part in content_parts if part).strip()
            if not text and not media:
                return

            envelope = Envelope(
                channel=self.name,
                account_id=self.context.spec.account_id or self.name,
                chat_id=chat_id if chat_type == "group" else sender_id,
                chat_type=chat_type,
                sender_id=sender_id,
                message_id=message_id,
                reply_to_id=str(getattr(message, "parent_id", None) or "") or None,
                deployment_id=self.context.deployment_id,
                timestamp=time.time(),
            )
            await self._handler(
                UserTurn(
                    envelope=envelope,
                    text=text,
                    media=media,
                    metadata={
                        "message_id": message_id,
                        "raw_chat_id": chat_id,
                        "msg_type": msg_type,
                        "root_id": getattr(message, "root_id", None),
                        "thread_id": getattr(message, "thread_id", None),
                    },
                )
            )
        except Exception as exc:
            logger.warning(f"feishu 消息处理失败: {type(exc).__name__}: {exc}")

    def _fetch_bot_open_id(self) -> str | None:
        if self._client is None:
            return None
        return FeishuTransport(self._client).fetch_bot_open_id()

    def _is_allowed(self, sender_id: str) -> bool:
        allow = self.config.allow_from
        if not allow:
            return False
        if "*" in allow:
            return True
        return sender_id in allow

    def _is_group_message_for_bot(self, message: Any) -> bool:
        if self.config.group_policy == "open":
            return True
        return self._is_bot_mentioned(message)

    def _is_bot_mentioned(self, message: Any) -> bool:
        raw_content = str(getattr(message, "content", "") or "")
        if "@_all" in raw_content:
            return True
        for mention in getattr(message, "mentions", None) or []:
            mention_id = getattr(getattr(mention, "id", None), "open_id", None)
            if self._bot_open_id and mention_id == self._bot_open_id:
                return True
        return False

    @staticmethod
    def _resolve_mentions(text: str, mentions: list[Any] | None) -> str:
        return resolve_mentions(text, mentions)

    def _format_outbound_message(self, text: str) -> tuple[str, str]:
        return format_outbound_message(text)

    def _format_outbound_reply(self, reply: AgentReply) -> tuple[str, str]:
        return format_outbound_reply(reply)

    def _send_or_reply_sync(
        self,
        receive_id_type: str,
        chat_id: str,
        msg_type: str,
        content: str,
        message_id: str | None,
    ) -> None:
        if self._client is None:
            return
        FeishuTransport(self._client).send_or_reply(
            receive_id_type=receive_id_type,
            chat_id=chat_id,
            msg_type=msg_type,
            content=content,
            message_id=message_id,
            reply_to_message=self.config.reply_to_message,
        )

    def _reply_message_sync(self, message_id: str, msg_type: str, content: str) -> bool:
        if self._client is None:
            return False
        return FeishuTransport(self._client).reply_message(
            message_id, msg_type, content
        )

    def _send_message_sync(
        self, receive_id_type: str, chat_id: str, msg_type: str, content: str
    ) -> bool:
        if self._client is None:
            return False
        return FeishuTransport(self._client).send_message(
            receive_id_type, chat_id, msg_type, content
        )

    def _upload_image_sync(self, file_path: Path) -> str | None:
        if self._client is None:
            return None
        return FeishuTransport(self._client).upload_image(file_path)

    def _upload_file_sync(self, file_path: Path) -> str | None:
        if self._client is None:
            return None
        return FeishuTransport(self._client).upload_file(file_path)

    async def _download_and_save_media(
        self, msg_type: str, content_json: dict[str, Any], message_id: str
    ) -> MediaRef | None:
        if self._client is None:
            return None
        file_path = (
            self._media_dir() / f"{message_id}_{msg_type}{self._media_suffix(msg_type)}"
        )
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None,
            self._download_media_sync,
            msg_type,
            content_json,
            message_id,
            file_path,
        )
        if not success:
            return None
        return build_media_ref(file_path, msg_type)

    def _download_media_sync(
        self,
        msg_type: str,
        content_json: dict[str, Any],
        message_id: str,
        output_path: Path,
    ) -> bool:
        if self._client is None:
            return False
        return FeishuTransport(self._client).download_media(
            msg_type, content_json, message_id, output_path
        )

    def _media_dir(self) -> Path:
        root = Path(self.config.media_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _media_suffix(msg_type: str) -> str:
        return {
            "image": ".png",
            "audio": ".ogg",
            "media": ".mp4",
            "file": ".bin",
        }.get(msg_type, ".bin")


__all__ = ["FEISHU_AVAILABLE", "FeishuChannel", "FeishuChannelConfig"]
