"""Ambient Agent — continuous visual monitor powered by Cosmos Reason2-8B.

Processes camera frames against a dynamic goal prompt and emits structured
signals (CLEAR, WARNING, PROGRESS, CORRECTION, GOAL_REACHED, FAILURE).
Runs on a local NIM endpoint via the OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from spark_sight.agents.base import BaseAgent
from spark_sight.bridge.models import AmbientResponse, AmbientSignal
from spark_sight.bridge.prompt_state import PromptState
from spark_sight.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_DEFAULT_NIM_BASE_URL = _settings.cosmos.nim_url
_DEFAULT_MODEL = _settings.cosmos.model

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Ambient Agent for Spark Sight, an accessibility assistant for \
visually impaired users navigating New York City. You are their eyes.

The camera is chest-mounted on the user, giving a first-person perspective \
pointing slightly downward. Objects at the center-bottom are directly in \
front of the user; objects at the top of the frame are farther away.

You receive one camera frame and a goal prompt. \
Evaluate the scene and respond with EXACTLY ONE JSON object.

IMPORTANT: Output ONLY the JSON object. No thinking, no explanation, no \
markdown fences. Just the raw JSON.

{"signal": "<SIGNAL>", "message": "<text or empty string>", "reasoning": "<brief>"}

SIGNAL must be one of:
- CLEAR: nothing to report. message MUST be "". Use this most of the time.
- WARNING: immediate safety hazard (cyclist, obstacle, construction). \
message describes the danger. Always fires regardless of mode.
- GOAL_REACHED: the ACTIVE GOAL condition is FULLY satisfied. \
The user is blind — they must be close enough to physically reach or \
touch the target. Do NOT signal GOAL_REACHED just because the target \
is visible in the distance. Use PROGRESS to guide them closer first. \
Only use GOAL_REACHED when the target is within arm's reach or the \
user has clearly arrived. message confirms what was found/achieved.
- PROGRESS: meaningful step toward the goal but NOT yet achieved. \
Use sparingly — only for significant milestones (e.g. "subway sign \
visible 50 feet ahead").
- CORRECTION: user needs to change direction to reach the goal.
- FAILURE: the goal CANNOT be achieved (path fully blocked, target gone). \
message explains why.

Rules:
1. When there is NO "ACTIVE GOAL" section below, only emit CLEAR or WARNING.
2. When there IS an "ACTIVE GOAL", check if the goal condition is MET in this \
frame. If yes → GOAL_REACHED immediately. Do not use PROGRESS or CORRECTION \
if the goal is already satisfied.
3. Safety warnings (WARNING) ALWAYS fire regardless of mode.
4. Prefer CLEAR. Only speak when there is actionable information.
5. Messages must be concise (1-2 sentences), spoken aloud to a blind person. \
Describe position and content, not colors.
6. Output valid JSON only. No markdown, no explanation outside the JSON.
7. DO NOT repeat instructions you already gave recently. Check the \
RECENT HISTORY section below — if you already said something similar \
within the last 30 seconds, emit CLEAR instead. Only speak again if \
the situation has CHANGED or new information is available.
"""

_INSPECT_SYSTEM_PROMPT = """\
You are a visual assistant for a visually impaired user. The image is from a \
chest-mounted camera (first-person view, slightly downward angle). Answer the \
following question concisely in 1-3 sentences. Describe positions and content, \
never reference colors alone.

Respond with EXACTLY ONE JSON object:
{
  "signal": "CLEAR",
  "message": "<your answer>",
  "reasoning": "<brief reasoning>"
}

Output valid JSON only. No markdown, no explanation outside the JSON.
"""


class AmbientAgent(BaseAgent):
    """Evaluates camera frames against the current goal prompt.

    Each call to :meth:`process` takes one frame and returns an
    :class:`AmbientResponse` with a signal + optional message.

    Parameters
    ----------
    prompt_state:
        Shared state — the agent reads the compiled prompt each frame.
    nim_base_url:
        Base URL of the local NIM inference endpoint.
    model:
        Model identifier to pass to the NIM API.
    """

    def __init__(
        self,
        prompt_state: PromptState,
        *,
        nim_base_url: str = _DEFAULT_NIM_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._state = prompt_state
        self._nim_base_url = nim_base_url
        self._model = model
        self._client: AsyncOpenAI | None = None
        # Recent non-CLEAR messages with timestamps (max 10, last 60s).
        self._history: list[tuple[float, str, str]] = []  # (timestamp, signal, message)

    @property
    def name(self) -> str:
        return "AmbientAgent"

    async def start(self) -> None:
        """Initialise the API client."""
        import os
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key=os.environ.get("GEMINI_API_KEY") or "not-needed",
        )
        logger.info("%s started (model=%s)", self.name, self._model)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("%s stopped", self.name)

    async def process(self, input_data: dict[str, Any]) -> AmbientResponse:
        """Evaluate a single camera frame.

        Parameters
        ----------
        input_data:
            ``{"frame_base64": str}`` — a base64-encoded JPEG frame.
            Optionally ``{"prompt_override": str}`` for inspect mode.

        Returns
        -------
        AmbientResponse
        """
        frame_b64: str | None = input_data.get("frame_base64")
        prompt_override: str | None = input_data.get("prompt_override")

        if self._client is None or frame_b64 is None:
            return AmbientResponse(signal=AmbientSignal.CLEAR)

        # Build system content: inspect override or normal compiled prompt.
        if prompt_override:
            system_content = (
                _INSPECT_SYSTEM_PROMPT + f"\n\nQuestion: {prompt_override}"
            )
        else:
            compiled = self._state.get_compiled_prompt()
            history_block = self._format_history()
            system_content = (
                _SYSTEM_PROMPT
                + f"\n\n--- GOAL PROMPT ---\n{compiled}"
                + (f"\n\n--- RECENT HISTORY ---\n{history_block}" if history_block else "")
            )

        mode = "inspect" if prompt_override else "ambient"
        logger.debug("[Cosmos] %s call (frame=%d bytes)", mode, len(frame_b64))

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=128,
                temperature=0.1,
            )
            result = self._parse_response(response)
            if not prompt_override and result.signal != AmbientSignal.CLEAR:
                self._record_history(result)
            return result
        except Exception:
            logger.exception("[Cosmos] NIM inference failed")
            if prompt_override:
                return AmbientResponse(
                    signal=AmbientSignal.CLEAR,
                    message="Vision service timed out — please try again.",
                )
            return AmbientResponse(signal=AmbientSignal.CLEAR)

    async def inspect(self, frame_base64: str, prompt: str) -> AmbientResponse:
        """One-shot inspection query (called by Orchestrator for inspect action).

        Parameters
        ----------
        frame_base64:
            Base64-encoded JPEG frame to inspect.
        prompt:
            The question to answer about the frame.
        """
        return await self.process({
            "frame_base64": frame_base64,
            "prompt_override": prompt,
        })

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _record_history(self, result: AmbientResponse) -> None:
        """Record a non-CLEAR response with timestamp."""
        import time
        self._history.append((time.time(), str(result.signal), result.message))
        # Keep only last 10 entries.
        if len(self._history) > 10:
            self._history = self._history[-10:]

    def _format_history(self) -> str:
        """Format recent history (last 60s) for inclusion in the prompt."""
        import time
        now = time.time()
        # Prune entries older than 60s.
        self._history = [(t, s, m) for t, s, m in self._history if now - t < 60]
        if not self._history:
            return ""
        lines = []
        for ts, signal, msg in self._history:
            ago = int(now - ts)
            lines.append(f"- {ago}s ago [{signal}]: {msg}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove chain-of-thought ``<think>`` blocks from Cosmos output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if "</thought>" in text:
            text = text.split("</thought>", 1)[1].strip()
        return text

    def _parse_response(self, response: Any) -> AmbientResponse:
        """Parse a NIM chat completion into an :class:`AmbientResponse`."""
        raw: str = response.choices[0].message.content or ""
        logger.debug("[Cosmos raw] %s", raw[:300])

        raw = self._strip_think(raw)

        # VLMs sometimes wrap JSON in markdown code fences.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
            signal = AmbientSignal(data.get("signal", "CLEAR"))
            result = AmbientResponse(
                signal=signal,
                message=data.get("message", ""),
                reasoning=data.get("reasoning", ""),
            )
            if signal != AmbientSignal.CLEAR:
                logger.info(
                    "[Cosmos] %s: %s", signal, result.message[:100]
                )
            return result
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("[Cosmos] Failed to parse: %s", raw[:300])
            return AmbientResponse(signal=AmbientSignal.CLEAR)
