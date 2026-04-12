"""Central configuration for the Kokoro-82M TTS server.

All tunables live here. Import-only: no side effects, no logging, no CUDA.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8880

MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_LOCAL_DIR = Path("./models/kokoro-82m")
VOICES_SUBDIR = "voices"

SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

MAX_CHUNK_TOKENS = 510  # 512 context − 2 pad tokens
MIN_CHUNK_MERGE = 1

DEFAULT_DTYPE: Literal["fp32", "fp16", "bf16"] = "fp16"
ALLOWED_DTYPES = ("fp32", "fp16", "bf16")

DEFAULT_BATCH_SIZE = 4
DEFAULT_FIRST_BATCH_SIZE = 1  # TTFA optimization
DEFAULT_LANG_CODE = "a"  # 'a' = American English in kokoro
DEFAULT_VOICE = "af_bella"

ENABLE_TORCH_COMPILE = False
MAX_CONCURRENT_REQUESTS = 4

WARMUP_TEXT = "Hello world. This is a warmup pass used to trigger CUDA kernel JIT."

SUPPORTED_FORMATS = ("pcm", "wav", "mp3", "opus", "flac")
DEFAULT_FORMAT: Literal["pcm", "wav", "mp3", "opus", "flac"] = "pcm"

MP3_BITRATE = 96_000
OPUS_BITRATE = 48_000
FLAC_COMPRESSION = 5

MIME_TYPES = {
    "pcm": "application/octet-stream",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "flac": "audio/flac",
}


@dataclass
class ServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model_dir: Path = MODEL_LOCAL_DIR
    lang_code: str = DEFAULT_LANG_CODE
    dtype: str = DEFAULT_DTYPE
    batch_size: int = DEFAULT_BATCH_SIZE
    first_batch_size: int = DEFAULT_FIRST_BATCH_SIZE
    max_chunk_tokens: int = MAX_CHUNK_TOKENS
    enable_torch_compile: bool = ENABLE_TORCH_COMPILE
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    verify_patch: bool = False
    log_level: str = "info"
    device: str = "cuda"  # "cuda" | "cpu"; backend auto-falls back

    mp3_bitrate: int = MP3_BITRATE
    opus_bitrate: int = OPUS_BITRATE
    flac_compression: int = FLAC_COMPRESSION
    sample_rate: int = SAMPLE_RATE

    voices_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.voices_dir = self.model_dir / VOICES_SUBDIR
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"dtype must be one of {ALLOWED_DTYPES}, got {self.dtype!r}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.first_batch_size < 1 or self.first_batch_size > self.batch_size:
            raise ValueError("first_batch_size must be in [1, batch_size]")

    # -------- builders ------------------------------------------------------

    @classmethod
    def from_cli(cls, ns: argparse.Namespace) -> "ServerConfig":
        overrides: dict = {}
        for name in (
            "host",
            "port",
            "model_dir",
            "lang_code",
            "dtype",
            "batch_size",
            "first_batch_size",
            "enable_torch_compile",
            "max_concurrent_requests",
            "verify_patch",
            "log_level",
            "device",
        ):
            val = getattr(ns, name, None)
            if val is not None:
                overrides[name] = val
        return cls(**overrides)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        overrides: dict = {}
        if v := os.environ.get("KOKORO_HOST"):
            overrides["host"] = v
        if v := os.environ.get("KOKORO_PORT"):
            overrides["port"] = int(v)
        if v := os.environ.get("KOKORO_MODEL_DIR"):
            overrides["model_dir"] = Path(v)
        if v := os.environ.get("KOKORO_DTYPE"):
            overrides["dtype"] = v
        if v := os.environ.get("KOKORO_BATCH_SIZE"):
            overrides["batch_size"] = int(v)
        if v := os.environ.get("KOKORO_DEVICE"):
            overrides["device"] = v
        return cls(**overrides)

    def with_overrides(self, **kw) -> "ServerConfig":
        return replace(self, **kw)


def build_cli_parser(prog: str = "kokoro-tts-server") -> argparse.ArgumentParser:
    """Shared argparse for `server.py` and `benchmark.py`."""
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model-dir", type=Path, default=MODEL_LOCAL_DIR, dest="model_dir")
    p.add_argument("--lang-code", default=DEFAULT_LANG_CODE, dest="lang_code")
    p.add_argument(
        "--dtype",
        choices=ALLOWED_DTYPES,
        default=DEFAULT_DTYPE,
        help="Model precision (voice style vectors stay FP32).",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, dest="batch_size")
    p.add_argument(
        "--first-batch-size",
        type=int,
        default=DEFAULT_FIRST_BATCH_SIZE,
        dest="first_batch_size",
        help="Batch size of the first chunk group (smaller = lower TTFA).",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        default=ENABLE_TORCH_COMPILE,
        dest="enable_torch_compile",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        dest="max_concurrent_requests",
    )
    p.add_argument(
        "--verify-patch",
        action="store_true",
        default=False,
        dest="verify_patch",
        help="On warmup, assert the NimbleEdge alignment patch matches the unpatched FP32 forward.",
    )
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--log-level", default="info", dest="log_level")
    return p
