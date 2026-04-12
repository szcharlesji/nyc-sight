"""Tests for the configuration module."""

from __future__ import annotations

from spark_sight.config import Settings, get_settings


class TestSettings:
    def test_get_settings_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_has_all_services(self) -> None:
        s = get_settings()
        assert s.cosmos.nim_url
        assert s.cosmos.model
        assert s.nemotron.nim_url
        assert s.nemotron.model
        assert s.parakeet.ws_url
        assert s.tts.base_url
        assert s.tts.model

    def test_cosmos_model_default(self) -> None:
        s = get_settings()
        assert "cosmos" in s.cosmos.model.lower()

    def test_nemotron_model_default(self) -> None:
        s = get_settings()
        assert "nemotron" in s.nemotron.model.lower()

    def test_nim_urls_contain_v1(self) -> None:
        """NIM agent URLs should end with /v1."""
        s = get_settings()
        assert s.cosmos.nim_url.endswith("/v1")
        assert s.nemotron.nim_url.endswith("/v1")

    def test_tts_url_contains_v1(self) -> None:
        """Kokoro TTS URL should end with /v1."""
        s = get_settings()
        assert s.tts.base_url.endswith("/v1")

    def test_parakeet_ws_url(self) -> None:
        """Parakeet-EOU WebSocket URL should start with ws://."""
        s = get_settings()
        assert s.parakeet.ws_url.startswith("ws")
