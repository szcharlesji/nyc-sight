"""WebSocket message protocol.

All messages over the single WebSocket are binary, prefixed with a one-byte
type tag.  This module defines the tags and provides pack/unpack helpers.

iPhone → Server:
    0x01  JPEG camera frame
    0x02  PCM audio chunk (16 kHz, 16-bit mono)

Server → iPhone:
    0x03  WAV TTS audio
    0x04  JSON status update (UTF-8)
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any


class MessageType(IntEnum):
    """One-byte type tags for the unified WebSocket protocol."""

    FRAME = 0x01
    AUDIO = 0x02
    TTS = 0x03
    STATUS = 0x04


def pack_message(msg_type: MessageType, payload: bytes) -> bytes:
    """Prefix *payload* with a one-byte type tag."""
    return bytes([msg_type]) + payload


def pack_status(data: dict[str, Any]) -> bytes:
    """Serialize a status dict as a type-prefixed JSON message."""
    return pack_message(
        MessageType.STATUS,
        json.dumps(data, separators=(",", ":")).encode(),
    )


def pack_tts(wav_bytes: bytes) -> bytes:
    """Wrap TTS WAV audio with the type prefix."""
    return pack_message(MessageType.TTS, wav_bytes)


def unpack_message(data: bytes) -> tuple[MessageType, bytes]:
    """Split a raw WebSocket message into (type, payload).

    Raises ``ValueError`` if the message is empty or the type tag is unknown.
    """
    if len(data) < 1:
        raise ValueError("Empty WebSocket message")
    try:
        msg_type = MessageType(data[0])
    except ValueError:
        raise ValueError(f"Unknown message type: 0x{data[0]:02x}") from None
    return msg_type, data[1:]
