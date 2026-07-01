import json
import mimetypes
from typing import Any

from a2a.types import FilePart, FileWithBytes
from langchain_core.messages import BaseMessage, ToolMessage

_LANGCHAIN_BINARY_BLOCK_TYPES: dict[str, str] = {
    "file": "attachment",
    "image": "image",
}


def _filename_from_text_block(block: dict[str, Any]) -> str | None:
    text = block.get("text")
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, dict):
        name = payload.get("filename")
        if isinstance(name, str) and name:
            return name
    return None


def extract_file_parts(messages: list[BaseMessage]) -> list[tuple[str, FilePart]]:
    parts: list[tuple[str, FilePart]] = []
    for message in messages:
        if not (isinstance(message, ToolMessage)
                and isinstance(message.content, list)):
            continue
        pending_name: str | None = None
        counters: dict[str, int] = {"file": 0, "image": 0}
        for block in message.content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if not isinstance(block_type, str):
                continue
            if block_type == "text":
                hint = _filename_from_text_block(block)
                if hint:
                    pending_name = hint
                continue
            kind = _LANGCHAIN_BINARY_BLOCK_TYPES.get(block_type)
            if kind is None:
                continue
            b64 = block.get("base64")
            if not isinstance(b64, str) or not b64:
                continue
            mime_type = block.get("mime_type") or "application/octet-stream"
            if pending_name is not None:
                name = pending_name
                pending_name = None
            else:
                index = counters[block_type]
                counters[block_type] = index + 1
                guessed_ext = mimetypes.guess_extension(mime_type)
                ext = f"-{guessed_ext}" if guessed_ext is not None else ""
                suffix = f"-{index}" if index > 0 else ""
                name = f"{kind}{suffix}{ext}"
            parts.append((name, FilePart(file=FileWithBytes(
                name=name, mime_type=mime_type, bytes=b64,
            ))))
    return parts
