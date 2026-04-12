"""Spark Sight configuration — reads NIM endpoints from environment variables.

Loads a ``.env`` file (if present) via ``python-dotenv``, then reads
``os.environ``.  Every setting has a sensible default so the server can
start without a ``.env`` file (falling back to localhost NIM endpoints).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True, slots=True)
class CosmosSettings:
    """Cosmos Reason2-8B — Ambient Agent (VLM)."""

    nim_url: str = field(default_factory=lambda: _env("COSMOS_NIM_URL", "http://localhost:8000/v1"))
    model: str = field(default_factory=lambda: _env("COSMOS_MODEL", "nvidia/cosmos-reason2-8b"))


@dataclass(frozen=True, slots=True)
class NemotronSettings:
    """Nemotron-3-Nano-30B — Planning Agent (LLM)."""

    nim_url: str = field(default_factory=lambda: _env("NEMOTRON_NIM_URL", "http://localhost:8005/v1"))
    model: str = field(default_factory=lambda: _env("NEMOTRON_MODEL", "nemotron-nano"))


@dataclass(frozen=True, slots=True)
class ParakeetSettings:
    """Parakeet-EOU-120M — Streaming Speech-to-Text via WebSocket."""

    ws_url: str = field(default_factory=lambda: _env("PARAKEET_WS_URL", "ws://localhost:3030/asr"))


@dataclass(frozen=True, slots=True)
class TTSSettings:
    """Kokoro-82M — Text-to-Speech (OpenAI-compatible)."""

    base_url: str = field(default_factory=lambda: _env("TTS_BASE_URL", "http://localhost:8880/v1"))
    model: str = field(default_factory=lambda: _env("TTS_MODEL", "kokoro"))
    voice: str = field(default_factory=lambda: _env("TTS_VOICE", "af_bella"))


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level settings container for all four NIM models."""

    cosmos: CosmosSettings = field(default_factory=CosmosSettings)
    nemotron: NemotronSettings = field(default_factory=NemotronSettings)
    parakeet: ParakeetSettings = field(default_factory=ParakeetSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)


def get_settings() -> Settings:
    """Return a Settings instance populated from the environment."""
    return Settings()
