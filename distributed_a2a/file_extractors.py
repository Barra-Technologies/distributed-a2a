import json
import mimetypes
from typing import Any

from a2a.types import FilePart, FileWithBytes, FileWithUri
from langchain_core.messages import BaseMessage, ToolMessage
from mcp.types import (BlobResourceContents, EmbeddedResource, ImageContent,
                       ResourceLink)

from .mcp_interceptors import NON_TEXT_CONTENT_KEY

DELIVERED_ARTIFACT_KEY = "_a2a_delivered"

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
        result = _mcp_block_to_file_part(block, counters)
        if result is not None:
            out.append(result)
    return out


def _mcp_block_to_file_part(
    block: Any, counters: dict[str, int],
) -> tuple[str, FilePart] | None:
    if isinstance(block, EmbeddedResource) and isinstance(
        block.resource, BlobResourceContents,
    ):
        mime_type = block.resource.mimeType or "application/octet-stream"
        uri = str(block.resource.uri) if block.resource.uri is not None else ""
        return _blob_resource_to_file_part(
            mime_type, uri, block.resource.blob, counters,
        )
    if isinstance(block, ImageContent):
        return _image_bytes_to_file_part(
            block.mimeType or "application/octet-stream", block.data, counters,
        )
    if isinstance(block, ResourceLink):
        return _resource_link_to_file_part(
            block.mimeType or "application/octet-stream",
            str(block.uri),
            counters,
        )

    if not isinstance(block, dict):
        return None

    block_type = block.get("type")
    if block_type == "resource":
        resource = block.get("resource")
        if not isinstance(resource, dict):
            return None
        blob = resource.get("blob")
        if not isinstance(blob, str) or not blob:
            return None
        mime_type_str = _mime_type_or_default(resource.get("mimeType"))
        raw_uri = resource.get("uri")
        uri_str = str(raw_uri) if raw_uri else ""
        return _blob_resource_to_file_part(
            mime_type_str, uri_str, blob, counters,
        )
    if block_type == "image":
        data = block.get("data")
        if not isinstance(data, str) or not data:
            return None
        mime_type_str = _mime_type_or_default(block.get("mimeType"))
        return _image_bytes_to_file_part(mime_type_str, data, counters)
    if block_type == "resource_link":
        raw_uri = block.get("uri")
        if not raw_uri:
            return None
        mime_type_str = _mime_type_or_default(block.get("mimeType"))
        return _resource_link_to_file_part(
            mime_type_str, str(raw_uri), counters,
        )
    return None


def _mime_type_or_default(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "application/octet-stream"


def _blob_resource_to_file_part(
    mime_type: str, uri: str, blob: str, counters: dict[str, int],
) -> tuple[str, FilePart]:
    kind = "image" if mime_type.startswith("image/") else "attachment"
    if uri:
        name = _name_from_uri(uri, kind, counters[kind], mime_type)
    else:
        name = _synthetic_name(kind, counters[kind], mime_type)
    counters[kind] += 1
    return name, FilePart(file=FileWithBytes(
        name=name, mime_type=mime_type, bytes=blob,
    ))


def _image_bytes_to_file_part(
    mime_type: str, data: str, counters: dict[str, int],
) -> tuple[str, FilePart]:
    name = _synthetic_name("image", counters["image"], mime_type)
    counters["image"] += 1
    return name, FilePart(file=FileWithBytes(
        name=name, mime_type=mime_type, bytes=data,
    ))


def _resource_link_to_file_part(
    mime_type: str, uri: str, counters: dict[str, int],
) -> tuple[str, FilePart]:
    kind = "image" if mime_type.startswith("image/") else "attachment"
    name = _name_from_uri(uri, kind, counters[kind], mime_type)
    counters[kind] += 1
    return name, FilePart(file=FileWithUri(
        name=name, mime_type=mime_type, uri=uri,
    ))


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


def _is_delivered(message: ToolMessage) -> bool:
    return (
        isinstance(message.artifact, dict)
        and message.artifact.get(DELIVERED_ARTIFACT_KEY) is True
    )


def _mark_delivered(message: ToolMessage) -> None:
    if isinstance(message.artifact, dict):
        message.artifact[DELIVERED_ARTIFACT_KEY] = True
    else:
        message.artifact = {DELIVERED_ARTIFACT_KEY: True}


def extract_file_parts(
    messages: list[BaseMessage],
) -> tuple[list[tuple[str, FilePart]], list[ToolMessage]]:
    """Return the file parts to emit and the ``ToolMessage`` instances they
    came from. Each returned message is flagged in-place with
    `DELIVERED_ARTIFACT_KEY` so a subsequent call over the same
    checkpointed history skips it. Callers are expected to persist those
    mutations back to the checkpoint.
    """
    parts: list[tuple[str, FilePart]] = []
    delivered_sources: list[ToolMessage] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if _is_delivered(message):
            continue

        mcp_blocks = _mcp_blocks_from_artifact(message.artifact)
        if mcp_blocks is not None:
            msg_parts = _extract_from_mcp_blocks(mcp_blocks)
        elif isinstance(message.content, list):
            msg_parts = _extract_from_langchain_content_blocks(message.content)
        else:
            msg_parts = []

        if msg_parts:
            parts.extend(msg_parts)
            _mark_delivered(message)
            delivered_sources.append(message)
    return parts, delivered_sources
