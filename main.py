from __future__ import annotations

import argparse
import asyncio
import logging
import threading

import cv2
import gradio as gr
import numpy as np

from src.asr import ASREngine
from src.camera import CameraLoop
from src.detector import YOLODetector
from src.frame_buffer import FrameBuffer
from src.nyc_data import NYCData
from src.orchestrator import Orchestrator
from src.tts import PiperTTS
from src.vlm import VLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.buffer = FrameBuffer(maxlen=30)
        self.alert_queue: asyncio.Queue[str] = asyncio.Queue()
        self.voice_queue: asyncio.Queue[str] = asyncio.Queue()
        self.vlm_trigger_queue: asyncio.Queue[str] = asyncio.Queue()

        source: int | str = args.video if args.video else 0
        self.camera = CameraLoop(self.buffer, source=source)

        self.vlm = VLMClient()
        self.tts = PiperTTS(no_tts=args.no_tts)
        self.nyc = NYCData()

        self._ui_state: dict = {
            "response": "",
            "alert": "",
            "audio": None,
            "status": "Starting…",
        }
        self._ui_lock = threading.Lock()

        self.orchestrator = Orchestrator(
            buffer=self.buffer,
            alert_queue=self.alert_queue,
            voice_queue=self.voice_queue,
            vlm_trigger_queue=self.vlm_trigger_queue,
            vlm=self.vlm,
            tts=self.tts,
            nyc=self.nyc,
            ui_callback=self._ui_update,
        )

    def _ui_update(self, **kwargs) -> None:
        with self._ui_lock:
            for k, v in kwargs.items():
                if k in self._ui_state:
                    self._ui_state[k] = v

    def _get_frame_jpeg(self) -> np.ndarray | None:
        frame = self.buffer.latest()
        if frame is None:
            return None
        return cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

    def _start_backend(self) -> None:
        loop = asyncio.new_event_loop()
        self.buffer.set_loop(loop)

        self.camera.start()

        detector = YOLODetector(
            buffer=self.buffer,
            alert_queue=self.alert_queue,
            vlm_trigger_queue=self.vlm_trigger_queue,
        )
        asr = ASREngine(
            voice_queue=self.voice_queue,
            no_mic=self.args.no_mic,
        )

        async def _run_all():
            await asyncio.gather(
                detector.run(),
                asr.run(),
                self.orchestrator.run(),
            )

        loop.run_until_complete(_run_all())

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="NYC Sight — Visual Accessibility Assistant") as demo:
            gr.Markdown("# NYC Sight — Visual Accessibility Assistant")
            gr.Markdown(
                "Point a camera at your surroundings, speak questions, and hear spoken answers."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    webcam = gr.Image(
                        label="Camera Feed",
                        streaming=True,
                        type="numpy",
                    )
                with gr.Column(scale=1):
                    status_box = gr.Textbox(
                        label="Status",
                        value="Starting…",
                        interactive=False,
                    )
                    alert_box = gr.Textbox(
                        label="Latest YOLO Alert",
                        value="",
                        interactive=False,
                    )
                    response_box = gr.Textbox(
                        label="Latest VLM Response",
                        value="",
                        lines=5,
                        interactive=False,
                    )
                    audio_out = gr.Audio(
                        label="TTS Output",
                        autoplay=True,
                        type="filepath",
                    )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Text Command (use when --no-mic)",
                    placeholder="Type a question, e.g. 'What's in front of me?'",
                )
                send_btn = gr.Button("Send", variant="primary")

            with gr.Row():
                lat_input = gr.Number(label="Latitude", value=40.7780, precision=6)
                lon_input = gr.Number(label="Longitude", value=-73.9812, precision=6)

            def refresh():
                img = self._get_frame_jpeg()
                with self._ui_lock:
                    state = dict(self._ui_state)
                return (
                    img,
                    state.get("status", ""),
                    state.get("alert", ""),
                    state.get("response", ""),
                    state.get("audio"),
                )

            timer = gr.Timer(0.5)
            timer.tick(
                fn=refresh,
                outputs=[webcam, status_box, alert_box, response_box, audio_out],
            )

            def on_send(text):
                if text.strip():
                    self.voice_queue.put_nowait(text.strip())
                return ""

            send_btn.click(fn=on_send, inputs=[text_input], outputs=[text_input])
            text_input.submit(fn=on_send, inputs=[text_input], outputs=[text_input])

        return demo

    def run(self) -> None:
        backend_thread = threading.Thread(target=self._start_backend, daemon=True)
        backend_thread.start()

        demo = self.build_ui()
        demo.launch(server_name="0.0.0.0", server_port=7860)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC Sight — Visual Accessibility Assistant"
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Video file path instead of camera"
    )
    parser.add_argument(
        "--no-mic", action="store_true", help="Disable microphone, use text input"
    )
    parser.add_argument(
        "--no-tts", action="store_true", help="Disable TTS, print text instead"
    )
    args = parser.parse_args()

    app = App(args)
    app.run()


if __name__ == "__main__":
    main()
