"""Tests for the FastAPI server (HTTP + WebSocket)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from spark_sight.server.app import create_app
from spark_sight.server.frame_buffer import FrameBuffer
from spark_sight.server.protocol import MessageType, pack_message


@pytest.fixture
def app():
    buf = FrameBuffer()
    return create_app(buf, debug=True)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHTTP:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["debug"] is True

    def test_client_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Spark Sight" in resp.text

    def test_debug_endpoint(self, client: TestClient) -> None:
        resp = client.get("/debug")
        assert resp.status_code == 200
        data = resp.json()
        assert "frame_buffer" in data
        assert "queues" in data


class TestWebSocket:
    def test_connect_and_send_frame(self, client: TestClient, app) -> None:
        jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data"
        msg = pack_message(MessageType.FRAME, jpeg)

        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(msg)

        # Frame should be in the buffer now.
        buf: FrameBuffer = app.state.frame_buffer
        assert buf.count >= 1
        frame = buf.latest()
        assert frame is not None
        assert frame.jpeg == jpeg

    def test_connect_and_send_transcript(self, client: TestClient) -> None:
        text = "What do you see?"
        encoded = text.encode("utf-8")
        msg = pack_message(MessageType.TRANSCRIPT, encoded)

        # Should not raise — transcript is handled gracefully even without orchestrator.
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(msg)
