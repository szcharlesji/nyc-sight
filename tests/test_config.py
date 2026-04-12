"""Tests for the configuration module."""

from __future__ import annotations

from spark_sight.config import Settings, get_settings


class TestSettings:
    def test_get_settings_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_has_nim_models(self) -> None:
        s = get_settings()
        assert s.cosmos.nim_url
        assert s.cosmos.model
        assert s.nemotron.nim_url
        assert s.nemotron.model

    def test_urls_are_strings(self) -> None:
        s = get_settings()
        assert isinstance(s.cosmos.nim_url, str)
        assert isinstance(s.nemotron.nim_url, str)
