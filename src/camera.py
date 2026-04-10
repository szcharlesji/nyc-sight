from __future__ import annotations

import logging
import threading
import time

import cv2

from src.frame_buffer import Frame, FrameBuffer

logger = logging.getLogger(__name__)


class CameraLoop:
    def __init__(
        self,
        buffer: FrameBuffer,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self._buffer = buffer
        self._source = source
        self._width = width
        self._height = height
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera thread started (source=%s)", self._source)

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        while self._running:
            try:
                if cap is None or not cap.isOpened():
                    logger.info("Opening camera source: %s", self._source)
                    cap = cv2.VideoCapture(self._source)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    if not cap.isOpened():
                        logger.warning("Camera not available, retrying in 2s…")
                        time.sleep(2)
                        continue

                ok, img = cap.read()
                if not ok:
                    logger.warning("Frame read failed, reopening camera…")
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue

                self._buffer.push(Frame(image=img))

            except Exception:
                logger.exception("Camera loop error")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(2)

        if cap:
            cap.release()
        logger.info("Camera thread stopped")
