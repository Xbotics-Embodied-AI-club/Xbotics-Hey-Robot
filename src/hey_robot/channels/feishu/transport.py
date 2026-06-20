from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from hey_robot.protocol import MediaRef


class FeishuTransport:
    def __init__(self, client: Any) -> None:
        self.client = client

    def fetch_bot_open_id(self) -> str | None:
        try:
            import lark_oapi as lark

            request = (
                lark.BaseRequest.builder()
                .http_method(lark.HttpMethod.GET)
                .uri("/open-apis/bot/v3/info")
                .token_types({lark.AccessTokenType.APP})
                .build()
            )
            response = self.client.request(request)
            if not response.success():
                return None
            payload = json.loads(response.raw.content)
            bot = (
                (payload.get("data") or payload).get("bot") or payload.get("bot") or {}
            )
            return bot.get("open_id")
        except Exception:
            return None

    def send_or_reply(
        self,
        *,
        receive_id_type: str,
        chat_id: str,
        msg_type: str,
        content: str,
        message_id: str | None,
        reply_to_message: bool,
    ) -> None:
        if (
            message_id
            and reply_to_message
            and self.reply_message(message_id, msg_type, content)
        ):
            return
        self.send_message(receive_id_type, chat_id, msg_type, content)

    def reply_message(self, message_id: str, msg_type: str, content: str) -> bool:
        try:
            from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

            request = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type(msg_type)
                    .content(content)
                    .reply_in_thread(False)
                    .build()
                )
                .build()
            )
            response = self.client.im.v1.message.reply(request)
            return bool(response.success())
        except Exception:
            return False

    def send_message(
        self, receive_id_type: str, chat_id: str, msg_type: str, content: str
    ) -> bool:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )
        response = self.client.im.v1.message.create(request)
        return bool(response.success())

    def upload_image(self, file_path: Path) -> str | None:
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        with file_path.open("rb") as handle:
            request = (
                CreateImageRequest.builder()
                .request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(handle)
                    .build()
                )
                .build()
            )
            response = self.client.im.v1.image.create(request)
        if not response.success():
            return None
        return getattr(getattr(response, "data", None), "image_key", None)

    def upload_file(self, file_path: Path) -> str | None:
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        with file_path.open("rb") as handle:
            request = (
                CreateFileRequest.builder()
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_name(file_path.name)
                    .file_type("stream")
                    .file(handle)
                    .build()
                )
                .build()
            )
            response = self.client.im.v1.file.create(request)
        if not response.success():
            return None
        return getattr(getattr(response, "data", None), "file_key", None)

    def download_media(
        self,
        msg_type: str,
        content_json: dict[str, Any],
        message_id: str,
        output_path: Path,
    ) -> bool:
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        if msg_type == "image":
            file_key = str(content_json.get("image_key") or "")
            resource_type = "image"
        else:
            file_key = str(content_json.get("file_key") or "")
            resource_type = "file"
        if not file_key:
            return False
        request = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(resource_type)
            .build()
        )
        response = self.client.im.v1.message_resource.get(request)
        if not response.success():
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw = getattr(getattr(response, "file", None), "read", None)
        if callable(raw):
            output_path.write_bytes(response.file.read())
            return True
        content = getattr(getattr(response, "raw", None), "content", None)
        if isinstance(content, (bytes, bytearray)):
            output_path.write_bytes(bytes(content))
            return True
        return False


def build_media_ref(file_path: Path, msg_type: str) -> MediaRef:
    content_type, _ = mimetypes.guess_type(file_path.name)
    return MediaRef(
        uri=str(file_path),
        media_type=msg_type,
        name=file_path.name,
        content_type=content_type,
        size_bytes=file_path.stat().st_size if file_path.exists() else None,
    )
