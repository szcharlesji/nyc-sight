from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from time import time

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float = field(default_factory=time)


class FrameBuffer:
    def __init__(self, maxlen: int = 30) -> None:
        self._buf: deque[Frame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._new_frame_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._new_frame_event = asyncio.Event()

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._buf.append(frame)
        if self._loop and self._new_frame_event:
            self._loop.call_soon_threadsafe(self._new_frame_event.set)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    async def wait_for_new(self) -> Frame:
        if self._new_frame_event is None:
            raise RuntimeError("Call set_loop() before wait_for_new()")
        self._new_frame_event.clear()
        await self._new_frame_event.wait()
        frame = self.latest()
        assert frame is not None
        return frame
