"""Tests for the camera frame ring buffer."""

from __future__ import annotations

import base64

from spark_sight.server.frame_buffer import FrameBuffer


class TestFrameBuffer:
    def test_empty_buffer_returns_none(self) -> None:
        buf = FrameBuffer()
        assert buf.latest() is None
        assert buf.latest_base64() is None

    def test_push_and_latest(self) -> None:
        buf = FrameBuffer()
        buf.push(b"\xff\xd8fake-jpeg")
        frame = buf.latest()
        assert frame is not None
        assert frame.jpeg == b"\xff\xd8fake-jpeg"
        assert frame.timestamp > 0

    def test_latest_base64(self) -> None:
        buf = FrameBuffer()
        data = b"hello-frame"
        buf.push(data)
        b64 = buf.latest_base64()
        assert b64 is not None
        assert base64.b64decode(b64) == data

    def test_overflow_evicts_oldest(self) -> None:
        buf = FrameBuffer(max_size=3)
        for i in range(5):
            buf.push(f"frame-{i}".encode())
        assert buf.size == 3
        assert buf.count == 5
        # Latest should be the last pushed.
        frame = buf.latest()
        assert frame is not None
        assert frame.jpeg == b"frame-4"

    def test_clear(self) -> None:
        buf = FrameBuffer()
        buf.push(b"data")
        buf.clear()
        assert buf.size == 0
        assert buf.latest() is None

    def test_max_size(self) -> None:
        buf = FrameBuffer(max_size=10)
        assert buf.max_size == 10
