"""Streaming audio encoders for /v1/audio/speech.

Each encoder presents the same interface:

    enc = make_encoder("mp3", sample_rate=24000)
    enc.feed(int16_numpy_array) -> bytes   # may be b"" if nothing drained yet
    enc.close() -> bytes                   # final flush

The PCM passthrough and FLAC/MP3/Opus paths are true streaming (they emit
bytes as soon as enough samples have been accumulated by the underlying
codec). WAV is BUFFERED — a WAV file needs its total sample count in the
header, so we collect all PCM and emit the file at ``close()``.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


log = logging.getLogger(__name__)


class BaseEncoder(ABC):
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    @abstractmethod
    def feed(self, samples: np.ndarray) -> bytes: ...

    @abstractmethod
    def close(self) -> bytes: ...


# ---------------------------------------------------------------------------
# PCM
# ---------------------------------------------------------------------------


class PCMPassthrough(BaseEncoder):
    def feed(self, samples: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(samples, dtype=np.int16)
        return arr.tobytes()

    def close(self) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# WAV (buffered)
# ---------------------------------------------------------------------------


class WavBuffered(BaseEncoder):
    def __init__(self, sample_rate: int) -> None:
        super().__init__(sample_rate)
        self._parts: List[np.ndarray] = []

    def feed(self, samples: np.ndarray) -> bytes:
        self._parts.append(np.ascontiguousarray(samples, dtype=np.int16))
        return b""

    def close(self) -> bytes:
        import soundfile as sf

        if not self._parts:
            audio = np.zeros(0, dtype=np.int16)
        else:
            audio = np.concatenate(self._parts)
        buf = io.BytesIO()
        sf.write(
            buf,
            audio,
            samplerate=self.sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return buf.getvalue()


# ---------------------------------------------------------------------------
# FLAC (streaming via soundfile)
# ---------------------------------------------------------------------------


class _GrowBuffer(io.RawIOBase):
    """File-like sink that lets us drain appended bytes between feed() calls."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def write(self, data) -> int:  # type: ignore[override]
        self._buf.extend(bytes(data))
        return len(data)

    def drain(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out


class FlacStreaming(BaseEncoder):
    def __init__(self, sample_rate: int, compression: int = 5) -> None:
        super().__init__(sample_rate)
        import soundfile as sf  # noqa: F401 (ensures import failure is immediate)

        self._sink = _GrowBuffer()
        self._sf = __import__("soundfile").SoundFile(
            self._sink,
            mode="w",
            samplerate=sample_rate,
            channels=1,
            format="FLAC",
            subtype="PCM_16",
        )
        self._compression = compression

    def feed(self, samples: np.ndarray) -> bytes:
        self._sf.write(np.ascontiguousarray(samples, dtype=np.int16))
        self._sf.flush()
        return self._sink.drain()

    def close(self) -> bytes:
        self._sf.close()
        return self._sink.drain()


# ---------------------------------------------------------------------------
# MP3 / Opus via PyAV
# ---------------------------------------------------------------------------


class _PyAVStreaming(BaseEncoder):
    CODEC: str = ""  # override
    FORMAT: str = ""  # container format
    BITRATE: int = 96_000
    CODEC_RATE: Optional[int] = None  # if set, resample to this before encoding

    def __init__(self, sample_rate: int, bitrate: Optional[int] = None) -> None:
        super().__init__(sample_rate)
        import av  # pyav

        self._av = av
        self._sink = _GrowBuffer()
        self._container = av.open(self._sink, mode="w", format=self.FORMAT)
        codec_rate = self.CODEC_RATE or sample_rate
        self._stream = self._container.add_stream(self.CODEC, rate=codec_rate)
        self._stream.bit_rate = bitrate or self.BITRATE
        try:
            # One channel; most codecs accept 'mono'.
            self._stream.layout = "mono"
        except Exception:  # pragma: no cover - layout attr varies by codec
            pass
        self._codec_rate = codec_rate
        self._need_resample = codec_rate != sample_rate
        self._resampler = None
        if self._need_resample:
            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=codec_rate)
        self._closed = False

    def _make_frame(self, samples: np.ndarray):
        av = self._av
        samples = np.ascontiguousarray(samples, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        return frame

    def _encode_and_mux(self, frame) -> None:
        if self._resampler is not None and frame is not None:
            frames = self._resampler.resample(frame)
            if not isinstance(frames, list):
                frames = [frames]
        else:
            frames = [frame] if frame is not None else [None]
        for f in frames:
            for packet in self._stream.encode(f):
                self._container.mux(packet)

    def feed(self, samples: np.ndarray) -> bytes:
        if self._closed:
            return b""
        if samples.size == 0:
            return b""
        frame = self._make_frame(samples)
        try:
            self._encode_and_mux(frame)
        except Exception as exc:  # pragma: no cover
            log.warning("%s encode failed: %s", self.CODEC, exc)
            return b""
        return self._sink.drain()

    def close(self) -> bytes:
        if self._closed:
            return b""
        try:
            # Flush resampler, then the codec.
            if self._resampler is not None:
                tail = self._resampler.resample(None)
                if tail is None:
                    tail = []
                if not isinstance(tail, list):
                    tail = [tail]
                for f in tail:
                    for packet in self._stream.encode(f):
                        self._container.mux(packet)
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        except Exception as exc:  # pragma: no cover
            log.warning("%s flush failed: %s", self.CODEC, exc)
        try:
            self._container.close()
        except Exception:  # pragma: no cover
            pass
        self._closed = True
        return self._sink.drain()


class Mp3Streaming(_PyAVStreaming):
    CODEC = "libmp3lame"
    FORMAT = "mp3"
    BITRATE = 96_000


class OpusStreaming(_PyAVStreaming):
    CODEC = "libopus"
    FORMAT = "ogg"
    BITRATE = 48_000
    CODEC_RATE = 48_000  # libopus prefers 48k; resample if model is 24k


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_encoder(
    fmt: str,
    sample_rate: int,
    *,
    mp3_bitrate: int = 96_000,
    opus_bitrate: int = 48_000,
    flac_compression: int = 5,
) -> BaseEncoder:
    fmt = fmt.lower()
    if fmt == "pcm":
        return PCMPassthrough(sample_rate)
    if fmt == "wav":
        return WavBuffered(sample_rate)
    if fmt == "flac":
        return FlacStreaming(sample_rate, compression=flac_compression)
    if fmt == "mp3":
        return Mp3Streaming(sample_rate, bitrate=mp3_bitrate)
    if fmt == "opus":
        return OpusStreaming(sample_rate, bitrate=opus_bitrate)
    raise ValueError(f"unsupported format: {fmt}")
