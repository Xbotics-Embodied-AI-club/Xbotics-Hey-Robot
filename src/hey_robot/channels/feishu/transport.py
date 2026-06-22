from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import MediaRef

logger = HeyRobotLogger(name="feishu")

_SEND_MAX_ATTEMPTS = 3
_SEND_RETRY_BASE_DELAY_SEC = 0.5
_SEND_RETRY_MAX_DELAY_SEC = 2.0


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
                _log_feishu_failure("bot.v3.info", response)
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
        if message_id and reply_to_message:
            if self.reply_message(message_id, msg_type, content):
                return
            logger.warning(
                "飞书 message.reply 失败，回退使用 message.create "
                f"receive_id_type={receive_id_type} message_id={message_id}"
            )
        if not self.send_message(receive_id_type, chat_id, msg_type, content):
            logger.warning(
                "飞书 message.create 发送失败 "
                f"receive_id_type={receive_id_type} target={_redact_id(chat_id)}"
            )

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
            success = bool(response.success())
            if not success:
                _log_feishu_failure("message.reply", response)
            return success
        except Exception as exc:
            logger.warning(f"飞书 message.reply 异常: {type(exc).__name__}: {exc}")
            return False

    def send_message(
        self, receive_id_type: str, chat_id: str, msg_type: str, content: str
    ) -> bool:
        for attempt in range(1, _SEND_MAX_ATTEMPTS + 1):
            try:
                from lark_oapi.api.im.v1 import (
                    CreateMessageRequest,
                    CreateMessageRequestBody,
                )

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
                success = bool(response.success())
                if success:
                    if attempt > 1:
                        logger.info(
                            "飞书 message.create 重试成功 "
                            f"attempt={attempt} receive_id_type={receive_id_type}"
                        )
                    return True
                _log_feishu_failure("message.create", response)
                if not _is_retryable_response(response):
                    return False
                if attempt == _SEND_MAX_ATTEMPTS:
                    return False
                _sleep_before_retry("message.create", attempt)
            except Exception as exc:
                logger.warning(
                    "飞书 message.create 异常 "
                    f"attempt={attempt}/{_SEND_MAX_ATTEMPTS}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt == _SEND_MAX_ATTEMPTS:
                    return False
                _sleep_before_retry("message.create", attempt)
        return False

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
            _log_feishu_failure("image.create", response)
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
            _log_feishu_failure("file.create", response)
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
            _log_feishu_failure("message_resource.get", response)
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


def _log_feishu_failure(api: str, response: Any) -> None:
    code = getattr(response, "code", None)
    msg = getattr(response, "msg", None) or getattr(response, "message", None)
    request_id = getattr(response, "request_id", None)
    raw = getattr(getattr(response, "raw", None), "content", None)
    raw_text = ""
    if isinstance(raw, (bytes, bytearray)):
        raw_text = bytes(raw).decode("utf-8", errors="replace")[:500]
    elif raw is not None:
        raw_text = str(raw)[:500]
    logger.warning(
        f"飞书 API 调用失败 api={api} code={code} msg={msg} "
        f"request_id={request_id} raw={raw_text}"
    )


def _is_retryable_response(response: Any) -> bool:
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(response, "raw", None), "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True

    code = getattr(response, "code", None)
    if code is None:
        return False
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return False
    return numeric_code == 429 or 500 <= numeric_code < 600


def _sleep_before_retry(api: str, attempt: int) -> None:
    delay = min(
        _SEND_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)),
        _SEND_RETRY_MAX_DELAY_SEC,
    )
    logger.warning(
        f"飞书发送准备重试 api={api} attempt={attempt + 1}/"
        f"{_SEND_MAX_ATTEMPTS} delay_sec={delay:.1f}"
    )
    time.sleep(delay)


def _redact_id(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
