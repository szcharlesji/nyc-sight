"""Ambient Agent — continuous visual monitor powered by Cosmos Reason2-8B.

Processes camera frames against a dynamic goal prompt and emits structured
signals (CLEAR, WARNING, PROGRESS, CORRECTION, GOAL_REACHED, FAILURE).
Runs on a local NIM endpoint via the OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
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

You receive one camera frame (from a chest-mounted iPhone) and a goal prompt. \
Evaluate the scene and respond with EXACTLY ONE JSON object:

{
  "signal": "<SIGNAL>",
  "message": "<text to speak to the user, or empty string>",
  "reasoning": "<brief internal reasoning, not spoken>"
}

SIGNAL must be one of:
- CLEAR: nothing to report. message MUST be "".
- WARNING: immediate safety hazard or important information. message MUST \
describe the danger concisely.
- PROGRESS: meaningful progress toward the active goal (only when an ACTIVE \
GOAL is present). message describes the progress.
- CORRECTION: user needs to adjust course (only when an ACTIVE GOAL is \
present). message gives the correction.
- GOAL_REACHED: the active goal has been achieved (only when an ACTIVE GOAL \
is present). message confirms arrival.
- FAILURE: the goal cannot be achieved (only when an ACTIVE GOAL is present). \
message explains why.

Rules:
1. When there is NO "ACTIVE GOAL" section below, only emit CLEAR or WARNING.
2. When there IS an "ACTIVE GOAL", you may emit any signal.
3. Safety warnings (WARNING) ALWAYS fire regardless of mode. A fast-approaching \
cyclist, an open manhole, construction hazard — these override everything.
4. Prefer CLEAR. Only speak when there is actionable information. The user is \
walking and does not want constant narration.
5. Messages must be concise (1-2 sentences), spoken aloud to a blind person. \
No visual references like "the red sign" — describe position and content.
6. Output valid JSON only. No markdown, no explanation outside the JSON.
"""

_INSPECT_SYSTEM_PROMPT = """\
You are a visual assistant for a visually impaired user. Answer the following \
question about the image concisely in 1-3 sentences. Describe positions and \
content, never reference colors alone.

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

    @property
    def name(self) -> str:
        return "AmbientAgent"

    async def start(self) -> None:
        """Initialise the NIM client."""
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key="not-needed",  # local NIM doesn't require a key
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
            system_content = (
                _SYSTEM_PROMPT + f"\n\n--- GOAL PROMPT ---\n{compiled}"
            )

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
                max_tokens=256,
                temperature=0.1,
            )
            return self._parse_response(response)
        except Exception:
            logger.exception("NIM inference failed")
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> AmbientResponse:
        """Parse a NIM chat completion into an :class:`AmbientResponse`."""
        raw: str = response.choices[0].message.content or ""
        raw = raw.strip()

        # VLMs sometimes wrap JSON in markdown code fences.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
            signal = AmbientSignal(data.get("signal", "CLEAR"))
            return AmbientResponse(
                signal=signal,
                message=data.get("message", ""),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("Failed to parse Ambient response: %s", raw[:200])
            return AmbientResponse(signal=AmbientSignal.CLEAR)
