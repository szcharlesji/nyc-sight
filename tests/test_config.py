"""Tests for the configuration module."""

from __future__ import annotations

from spark_sight.config import Settings, get_settings


class TestSettings:
    def test_get_settings_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_has_all_four_models(self) -> None:
        s = get_settings()
        assert s.cosmos.nim_url
        assert s.cosmos.model
        assert s.nemotron.nim_url
        assert s.nemotron.model
        assert s.parakeet.nim_url
        assert s.parakeet.model
        assert s.magpie.nim_url
        assert s.magpie.model

    def test_cosmos_model_default(self) -> None:
        s = get_settings()
        assert "cosmos" in s.cosmos.model.lower()

    def test_nemotron_model_default(self) -> None:
        s = get_settings()
        assert "nemotron" in s.nemotron.model.lower()

    def test_urls_contain_v1(self) -> None:
        """All NIM URLs should end with /v1."""
        s = get_settings()
        assert s.cosmos.nim_url.endswith("/v1")
        assert s.nemotron.nim_url.endswith("/v1")
        assert s.parakeet.nim_url.endswith("/v1")
        assert s.magpie.nim_url.endswith("/v1")
