"""Thread-safe ring buffer for camera frames.

The WebSocket handler pushes raw JPEG bytes.  The Ambient Agent consumes the
latest frame via ``latest()`` or ``latest_base64()``.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Frame:
    """A single buffered camera frame."""

    jpeg: bytes
    """Raw JPEG bytes as received from the iPhone."""
    timestamp: float
    """``time.time()`` when the frame was received."""


class FrameBuffer:
    """Fixed-size ring buffer that stores the most recent camera frames.

    Parameters
    ----------
    max_size:
        Maximum number of frames to retain.  Oldest frames are evicted
        when the buffer is full.  Default 30 (~7.5 s at 4 FPS).
    """

    def __init__(self, max_size: int = 30) -> None:
        self._buf: deque[Frame] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._frame_count = 0

    def push(self, jpeg_bytes: bytes) -> None:
        """Store a new JPEG frame, evicting the oldest if at capacity."""
        frame = Frame(jpeg=jpeg_bytes, timestamp=time.time())
        with self._lock:
            self._buf.append(frame)
            self._frame_count += 1

    def latest(self) -> Frame | None:
        """Return the most recently pushed frame, or ``None`` if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def latest_base64(self) -> str | None:
        """Return the latest frame's JPEG bytes as a base64 string.

        This is the format the Ambient Agent expects in ``input_data``.
        """
        frame = self.latest()
        if frame is None:
            return None
        return base64.b64encode(frame.jpeg).decode("ascii")

    @property
    def count(self) -> int:
        """Total number of frames received since creation."""
        with self._lock:
            return self._frame_count

    @property
    def size(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buf)

    @property
    def max_size(self) -> int:
        return self._buf.maxlen or 0

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
