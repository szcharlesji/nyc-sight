"""WebSocket message protocol.

All messages over the single WebSocket are binary, prefixed with a one-byte
type tag.  This module defines the tags and provides pack/unpack helpers.

Phone → Server (upstream):
    0x01  JPEG camera frame
    0x02  PCM audio chunk (16 kHz, 16-bit mono)  — web client (server-side ASR)
    0x05  UTF-8 transcript text                   — iOS client (on-device STT)
    0x08  UTF-8 JSON location update              — iOS client GPS

Server → Phone (downstream):
    0x03  WAV TTS audio           — web client (server-side TTS via Magpie)
    0x04  JSON status update      — both clients (UTF-8)
    0x06  UTF-8 text response     — iOS client (on-device TTS via AVSpeech)
    0x07  UTF-8 JSON warning      — iOS client (urgent, interrupts TTS)
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any


class MessageType(IntEnum):
    """One-byte type tags for the unified WebSocket protocol."""

    # Upstream (client → server)
    FRAME = 0x01
    AUDIO = 0x02          # PCM audio for server-side Parakeet ASR
    TRANSCRIPT = 0x05     # Pre-transcribed text from on-device WhisperKit
    LOCATION = 0x08       # JSON: {"lat": float, "lng": float}

    # Downstream (server → client)
    TTS = 0x03            # WAV audio bytes (Magpie TTS) — web client
    STATUS = 0x04         # JSON status update — both clients
    TEXT_RESPONSE = 0x06  # Plain text for on-device TTS — iOS client
    WARNING_TEXT = 0x07   # JSON: {"text": str, "urgency": str} — iOS client


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


def pack_text_response(text: str, *, final: bool = True) -> bytes:
    """Pack a text response for on-device TTS (iOS client).

    The payload is a JSON object so the client knows whether more chunks
    are coming (for streaming/progressive TTS).
    """
    payload = json.dumps(
        {"text": text, "final": final}, separators=(",", ":")
    ).encode()
    return pack_message(MessageType.TEXT_RESPONSE, payload)


def pack_warning_text(text: str, urgency: str = "high") -> bytes:
    """Pack a text warning for on-device TTS (iOS client).

    Warnings interrupt any current speech on the iOS side.
    """
    payload = json.dumps(
        {"text": text, "urgency": urgency}, separators=(",", ":")
    ).encode()
    return pack_message(MessageType.WARNING_TEXT, payload)


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
