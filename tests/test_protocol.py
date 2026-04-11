"""Tests for the WebSocket message protocol helpers."""

from __future__ import annotations

import json

import pytest

from spark_sight.server.protocol import (
    MessageType,
    pack_message,
    pack_status,
    pack_tts,
    unpack_message,
)


class TestPackUnpack:
    def test_round_trip_frame(self) -> None:
        payload = b"\xff\xd8jpeg-data"
        packed = pack_message(MessageType.FRAME, payload)
        msg_type, unpacked = unpack_message(packed)
        assert msg_type == MessageType.FRAME
        assert unpacked == payload

    def test_round_trip_audio(self) -> None:
        payload = b"\x00\x01\x02pcm-audio"
        packed = pack_message(MessageType.AUDIO, payload)
        msg_type, unpacked = unpack_message(packed)
        assert msg_type == MessageType.AUDIO
        assert unpacked == payload

    def test_pack_status(self) -> None:
        data = {"type": "status", "signal": "CLEAR", "mode": "patrol"}
        packed = pack_status(data)
        msg_type, payload = unpack_message(packed)
        assert msg_type == MessageType.STATUS
        parsed = json.loads(payload)
        assert parsed["signal"] == "CLEAR"

    def test_pack_tts(self) -> None:
        wav = b"RIFF\x00\x00\x00\x00WAVEfake"
        packed = pack_tts(wav)
        msg_type, payload = unpack_message(packed)
        assert msg_type == MessageType.TTS
        assert payload == wav

    def test_unpack_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            unpack_message(b"")

    def test_unpack_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            unpack_message(b"\xff\x00\x00")
