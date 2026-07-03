import json
import mimetypes
from typing import Any

from a2a.types import FilePart, FileWithBytes, FileWithUri
from langchain_core.messages import BaseMessage, ToolMessage
from mcp.types import (BlobResourceContents, EmbeddedResource, ImageContent,
                       ResourceLink)

from .mcp_interceptors import NON_TEXT_CONTENT_KEY

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


def _synthetic_name(kind: str, index: int, mime_type: str) -> str:
    guessed_ext = mimetypes.guess_extension(mime_type)
    ext = guessed_ext if guessed_ext is not None else ""
    suffix = f"-{index}" if index > 0 else ""
    return f"{kind}{suffix}{ext}"


def _name_from_uri(uri: str, fallback_kind: str, index: int, mime_type: str) -> str:
    tail = uri.rsplit("/", 1)[-1]
    if tail:
        return tail
    return _synthetic_name(fallback_kind, index, mime_type)


def _extract_from_mcp_blocks(blocks: list[Any]) -> list[tuple[str, FilePart]]:
    out: list[tuple[str, FilePart]] = []
    counters: dict[str, int] = {"attachment": 0, "image": 0}
    for block in blocks:
        if isinstance(block, EmbeddedResource) and isinstance(block.resource, BlobResourceContents):
            mime_type = block.resource.mimeType or "application/octet-stream"
            uri = str(block.resource.uri) if block.resource.uri is not None else ""
            kind = "image" if mime_type.startswith("image/") else "attachment"
            if uri:
                name = _name_from_uri(uri, kind, counters[kind], mime_type)
            else:
                name = _synthetic_name(kind, counters[kind], mime_type)
            counters[kind] += 1
            out.append((name, FilePart(file=FileWithBytes(
                name=name, mime_type=mime_type, bytes=block.resource.blob,
            ))))
        elif isinstance(block, ImageContent):
            mime_type = block.mimeType or "application/octet-stream"
            name = _synthetic_name("image", counters["image"], mime_type)
            counters["image"] += 1
            out.append((name, FilePart(file=FileWithBytes(
                name=name, mime_type=mime_type, bytes=block.data,
            ))))
        elif isinstance(block, ResourceLink):
            mime_type = block.mimeType or "application/octet-stream"
            uri = str(block.uri)
            kind = "image" if mime_type.startswith("image/") else "attachment"
            name = _name_from_uri(uri, kind, counters[kind], mime_type)
            counters[kind] += 1
            out.append((name, FilePart(file=FileWithUri(
                name=name, mime_type=mime_type, uri=uri,
            ))))
    return out


def _extract_from_langchain_content_blocks(content: list[Any]) -> list[tuple[str, FilePart]]:
    out: list[tuple[str, FilePart]] = []
    pending_name: str | None = None
    counters: dict[str, int] = {"file": 0, "image": 0}
    for block in content:
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
            name = _synthetic_name(kind, index, mime_type)
        out.append((name, FilePart(file=FileWithBytes(
            name=name, mime_type=mime_type, bytes=b64,
        ))))
    return out


def _mcp_blocks_from_artifact(artifact: Any) -> list[Any] | None:
    if not isinstance(artifact, dict):
        return None
    structured = artifact.get("structured_content")
    if not isinstance(structured, dict):
        return None
    blocks = structured.get(NON_TEXT_CONTENT_KEY)
    if not isinstance(blocks, list) or not blocks:
        return None
    return blocks


def extract_file_parts(messages: list[BaseMessage]) -> list[tuple[str, FilePart]]:
    parts: list[tuple[str, FilePart]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        mcp_blocks = _mcp_blocks_from_artifact(message.artifact)
        if mcp_blocks is not None:
            parts.extend(_extract_from_mcp_blocks(mcp_blocks))
            continue

        if isinstance(message.content, list):
            parts.extend(_extract_from_langchain_content_blocks(message.content))
    return parts
