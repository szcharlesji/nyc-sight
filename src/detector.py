from __future__ import annotations

import asyncio
import logging
import time

from ultralytics import YOLO

from src.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

DANGER_CLASSES = {
    "person", "bicycle", "car", "bus", "truck",
    "motorcycle", "dog", "fire hydrant", "stop sign",
}
PROXIMITY_THRESHOLD = 0.35
ALERT_COOLDOWN = 2.0  # seconds between repeated alerts for same class


class YOLODetector:
    def __init__(
        self,
        buffer: FrameBuffer,
        alert_queue: asyncio.Queue[str],
        vlm_trigger_queue: asyncio.Queue[str],
        model_name: str = "yolov8n.pt",
    ) -> None:
        self._buffer = buffer
        self._alert_queue = alert_queue
        self._vlm_trigger_queue = vlm_trigger_queue
        self._model = YOLO(model_name)
        self._last_alert: dict[str, float] = {}

    async def run(self) -> None:
        logger.info("YOLO detector started")
        loop = asyncio.get_running_loop()
        while True:
            frame = self._buffer.latest()
            if frame is None:
                await asyncio.sleep(0.03)
                continue

            results = await loop.run_in_executor(
                None, lambda: self._model(frame.image, verbose=False)
            )

            h = frame.image.shape[0]
            now = time.time()

            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if conf < 0.4:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_h = y2 - y1
                    ratio = bbox_h / h

                    if cls_name in DANGER_CLASSES and ratio > PROXIMITY_THRESHOLD:
                        last = self._last_alert.get(cls_name, 0)
                        if now - last > ALERT_COOLDOWN:
                            direction = self._direction(x1, x2, frame.image.shape[1])
                            dist = self._approx_distance(ratio)
                            alert = f"Warning: {cls_name} {direction}, approximately {dist}"
                            await self._alert_queue.put(alert)
                            self._last_alert[cls_name] = now
                            logger.info("YOLO alert: %s", alert)

                    unusual = cls_name not in {"person", "car", "truck", "bus"}
                    if unusual and conf > 0.5 and ratio > 0.2:
                        last = self._last_alert.get(f"vlm_{cls_name}", 0)
                        if now - last > 10:
                            await self._vlm_trigger_queue.put(
                                f"YOLO detected unusual object: {cls_name}"
                            )
                            self._last_alert[f"vlm_{cls_name}"] = now

            await asyncio.sleep(0.03)

    @staticmethod
    def _direction(x1: float, x2: float, w: int) -> str:
        cx = (x1 + x2) / 2
        rel = cx / w
        if rel < 0.33:
            return "on your left"
        elif rel > 0.66:
            return "on your right"
        return "directly ahead"

    @staticmethod
    def _approx_distance(ratio: float) -> str:
        if ratio > 0.7:
            return "very close, within 3 feet"
        elif ratio > 0.5:
            return "about 5 feet away"
        return "about 8 feet away"
