"""Ambient Agent — continuous visual monitor powered by Cosmos Reason2-8B.

This module contains the agent stub.  Actual NIM inference is behind a
placeholder that will be wired to a local Cosmos Reason2 NIM endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from nyc_sight.agents.base import BaseAgent
from nyc_sight.bridge.models import AmbientResponse, AmbientSignal
from nyc_sight.bridge.prompt_state import PromptState

logger = logging.getLogger(__name__)

# Default NIM endpoint for Cosmos Reason2 running locally on the GB10.
_DEFAULT_NIM_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "nvidia/cosmos-reason2-8b"


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
            (In the stub, the frame is ignored and a CLEAR signal is returned.)

        Returns
        -------
        AmbientResponse
        """
        _prompt = self._state.get_compiled_prompt()  # used by NIM inference
        frame_b64: str | None = input_data.get("frame_base64")

        if self._client is None or frame_b64 is None:
            # Stub: no model loaded or no frame — return CLEAR.
            return AmbientResponse(signal=AmbientSignal.CLEAR)

        # ----- NIM inference (placeholder) ------------------------------------
        # When the Cosmos Reason2 NIM container is running, this will send the
        # frame as a vision message and parse the structured response.
        #
        # response = await self._client.chat.completions.create(
        #     model=self._model,
        #     messages=[
        #         {"role": "system", "content": prompt},
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        #                 }
        #             ],
        #         },
        #     ],
        #     max_tokens=256,
        # )
        # return self._parse_response(response)
        # ----------------------------------------------------------------------

        logger.debug("Ambient frame processed (stub) — CLEAR")
        return AmbientResponse(signal=AmbientSignal.CLEAR)
