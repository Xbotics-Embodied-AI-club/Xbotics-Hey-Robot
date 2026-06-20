from __future__ import annotations

import json
from typing import Any

MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
    "media": "[video]",
}


def extract_share_card_content(content_json: dict[str, Any], msg_type: str) -> str:
    if msg_type == "share_chat":
        return f"[shared chat: {content_json.get('chat_id', '')}]"
    if msg_type == "share_user":
        return f"[shared user: {content_json.get('user_id', '')}]"
    if msg_type == "share_calendar_event":
        return f"[shared calendar event: {content_json.get('event_key', '')}]"
    if msg_type == "system":
        return "[system message]"
    if msg_type == "merge_forward":
        return "[merged forward messages]"
    return extract_interactive_text(content_json) or f"[{msg_type}]"


def extract_interactive_text(content: Any) -> str:
    parts: list[str] = []
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return str(content).strip()
    if not isinstance(content, dict):
        return ""
    title = content.get("title")
    if isinstance(title, dict):
        title_text = str(title.get("content") or title.get("text") or "").strip()
        if title_text:
            parts.append(title_text)
    elif isinstance(title, str) and title.strip():
        parts.append(title.strip())
    for element in content.get("elements", []) or []:
        if isinstance(element, dict):
            parts.extend(extract_element_content(element))
    if isinstance(content.get("card"), dict):
        nested = extract_interactive_text(content["card"])
        if nested:
            parts.append(nested)
    if isinstance(content.get("header"), dict):
        header_title = content["header"].get("title")
        if isinstance(header_title, dict):
            header_text = str(
                header_title.get("content") or header_title.get("text") or ""
            ).strip()
            if header_text:
                parts.append(header_text)
    return "\n".join(part for part in parts if part).strip()


def extract_element_content(element: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    tag = str(element.get("tag") or "")
    if tag in {"markdown", "lark_md"}:
        content = str(element.get("content") or "").strip()
        if content:
            parts.append(content)
    elif tag == "div":
        text = element.get("text")
        if isinstance(text, dict):
            value = str(text.get("content") or text.get("text") or "").strip()
            if value:
                parts.append(value)
    elif tag == "a":
        text = str(element.get("text") or "").strip()
        href = str(element.get("href") or "").strip()
        if text:
            parts.append(text)
        if href:
            parts.append(f"link: {href}")
    elif tag == "button":
        text = element.get("text")
        if isinstance(text, dict):
            value = str(text.get("content") or "").strip()
            if value:
                parts.append(value)
        url = str(
            element.get("url") or element.get("multi_url", {}).get("url") or ""
        ).strip()
        if url:
            parts.append(f"link: {url}")
    elif tag == "img":
        alt = element.get("alt")
        if isinstance(alt, dict):
            parts.append(str(alt.get("content") or "[image]"))
        else:
            parts.append("[image]")
    for nested in element.get("elements", []) or []:
        if isinstance(nested, dict):
            parts.extend(extract_element_content(nested))
    return parts


def extract_post_content(content_json: dict[str, Any]) -> tuple[str, list[str]]:
    def parse_block(block: dict[str, Any]) -> tuple[str, list[str]]:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return "", []
        texts: list[str] = []
        images: list[str] = []
        if block.get("title"):
            texts.append(str(block["title"]))
        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for element in row:
                if not isinstance(element, dict):
                    continue
                tag = str(element.get("tag") or "")
                if tag in {"text", "a"}:
                    texts.append(str(element.get("text") or ""))
                elif tag == "at":
                    texts.append(f"@{element.get('user_name', 'user')}")
                elif tag == "code_block":
                    texts.append(
                        f"\n```{element.get('language', '')}\n{element.get('text', '')}\n```\n"
                    )
                elif tag == "img" and element.get("image_key"):
                    images.append(str(element["image_key"]))
        return " ".join(texts).strip(), images

    root: Any = (
        content_json.get("post")
        if isinstance(content_json.get("post"), dict)
        else content_json
    )
    if not isinstance(root, dict):
        return "", []
    if "content" in root:
        text, images = parse_block(root)
        if text or images:
            return text, images
    for locale in ("zh_cn", "en_us", "ja_jp"):
        if isinstance(root.get(locale), dict):
            text, images = parse_block(root[locale])
            if text or images:
                return text, images
    for value in root.values():
        if isinstance(value, dict):
            text, images = parse_block(value)
            if text or images:
                return text, images
    return "", []


def resolve_mentions(text: str, mentions: list[Any] | None) -> str:
    if not mentions:
        return text
    resolved = text
    for mention in mentions:
        key = getattr(mention, "key", None)
        mention_id = getattr(getattr(mention, "id", None), "open_id", None)
        name = getattr(mention, "name", None) or "user"
        if key and key in resolved:
            resolved = resolved.replace(
                key, f"@{name} ({mention_id})" if mention_id else f"@{name}"
            )
    return resolved
