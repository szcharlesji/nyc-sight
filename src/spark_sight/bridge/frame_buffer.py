"""Thread-safe ring buffer for camera frames.

Stores the last *N* base64-encoded JPEG frames from the iPhone camera.
The Ambient Agent reads the latest frame each loop iteration; the
Orchestrator reads it for inspect queries.
"""

from __future__ import annotations

import collections
import threading


class FrameBuffer:
    """Fixed-size ring buffer holding base64-encoded JPEG frames.

    Parameters
    ----------
    maxlen:
        Maximum number of frames to retain (default 30, per spec).
    """

    def __init__(self, maxlen: int = 30) -> None:
        self._buffer: collections.deque[str] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame_base64: str) -> None:
        """Append a frame.  Oldest frame is evicted if at capacity."""
        with self._lock:
            self._buffer.append(frame_base64)

    def latest(self) -> str | None:
        """Return the most recent frame, or ``None`` if the buffer is empty."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
