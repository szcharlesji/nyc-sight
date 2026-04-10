from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

PIPER_MODEL = "en_US-lessac-medium"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "piper"


class PiperTTS:
    def __init__(self, no_tts: bool = False) -> None:
        self._no_tts = no_tts
        self._model_path: Path | None = None
        if not no_tts:
            self._ensure_model()

    def _ensure_model(self) -> None:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        onnx = ASSETS_DIR / f"{PIPER_MODEL}.onnx"
        json_cfg = ASSETS_DIR / f"{PIPER_MODEL}.onnx.json"
        if onnx.exists() and json_cfg.exists():
            self._model_path = onnx
            return

        logger.info("Downloading Piper model %s …", PIPER_MODEL)
        base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{PIPER_MODEL}.onnx"
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(onnx), base],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["curl", "-L", "-o", str(json_cfg), base + ".json"],
                check=True, capture_output=True,
            )
            self._model_path = onnx
            logger.info("Piper model downloaded")
        except Exception:
            logger.exception("Failed to download Piper model")

    def speak(self, text: str) -> str | None:
        if self._no_tts:
            logger.info("[TTS disabled] %s", text)
            return None

        if not self._model_path or not self._model_path.exists():
            logger.warning("Piper model not available")
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        try:
            proc = subprocess.run(
                [
                    "piper",
                    "--model", str(self._model_path),
                    "--output_file", tmp.name,
                ],
                input=text.encode(),
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return tmp.name
            logger.warning("Piper failed: %s", proc.stderr.decode())
        except FileNotFoundError:
            logger.warning("piper binary not found, falling back to text output")
        except Exception:
            logger.exception("TTS error")

        os.unlink(tmp.name)
        return None
