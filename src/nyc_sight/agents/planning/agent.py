"""Planning Agent — intent parser & route planner powered by Nemotron Super 120B.

This module contains the agent stub.  Actual NIM inference is behind a
placeholder that will be wired to a local Nemotron Super NIM endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from nyc_sight.agents.base import BaseAgent
from nyc_sight.bridge.models import PlanningAction, PlanningResponse
from nyc_sight.bridge.prompt_state import PromptState

logger = logging.getLogger(__name__)

# Default NIM endpoint for Nemotron Super running locally on the GB10.
_DEFAULT_NIM_BASE_URL = "http://localhost:8001/v1"
_DEFAULT_MODEL = "nvidia/nemotron-super-120b"

_SYSTEM_PROMPT = """\
You are the Planning Agent for Spark Sight, an accessibility assistant for \
visually impaired users navigating New York City.

You receive the user's spoken request (transcribed) and the current system state.
You must decide on ONE action and return a JSON object with these fields:

  action:         one of "set_goal", "inspect", "answer", "reset", "replan"
  message:        text to speak to the user (empty string if silent)
  goal:           new goal text (only for set_goal / replan, else null)
  nyc_context:    NYC data context string (only for set_goal / replan, else null)
  inspect_prompt: one-shot prompt for the Ambient Agent (only for inspect, else null)

Available tools (call via function_call):
  - nyc_accessibility_lookup(lat, lon, radius_m) → nearby accessibility data
  - set_ambient_goal(goal, nyc_context) → updates the Ambient Agent's prompt
  - reset_ambient_goal() → clears the Ambient Agent's goal
  - web_search(query) → web search (only when WiFi available)
"""


class PlanningAgent(BaseAgent):
    """Parses user intent, queries NYC data, and programs the Ambient Agent.

    Activated on-demand by voice input or by FAILURE signals from the
    Ambient Agent.

    Parameters
    ----------
    prompt_state:
        Shared state — the agent reads it for context and writes via the
        Orchestrator.
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
        return "PlanningAgent"

    async def start(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key="not-needed",
        )
        logger.info("%s started (model=%s)", self.name, self._model)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("%s stopped", self.name)

    async def process(self, input_data: dict[str, Any]) -> PlanningResponse:
        """Process a user voice transcript or a FAILURE replan trigger.

        Parameters
        ----------
        input_data:
            ``{"transcript": str}`` for voice input, or
            ``{"failure_reason": str}`` for replan triggers.

        Returns
        -------
        PlanningResponse
        """
        transcript = input_data.get("transcript", "")
        failure_reason = input_data.get("failure_reason", "")
        trigger = transcript or failure_reason

        if not trigger:
            return PlanningResponse(action=PlanningAction.ANSWER)

        # Build context from current state.
        _state_snapshot = self._state.get_snapshot()  # used by NIM inference

        if self._client is None:
            # Stub: echo back a placeholder answer.
            logger.debug("Planning processed (stub): %s", trigger[:80])
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message=f"I heard: {trigger}. (Planning Agent stub — NIM not connected.)",
            )

        # ----- NIM inference (placeholder) ------------------------------------
        # When Nemotron Super NIM is running, this will send the transcript
        # along with system state and parse the structured JSON response.
        #
        # messages = [
        #     {"role": "system", "content": _SYSTEM_PROMPT},
        #     {
        #         "role": "user",
        #         "content": (
        #             f"User said: {trigger}\n\n"
        #             f"Current mode: {state_snapshot.mode}\n"
        #             f"Active goal: {state_snapshot.active_goal or 'none'}\n"
        #             f"NYC context: {state_snapshot.nyc_context or 'none'}"
        #         ),
        #     },
        # ]
        # response = await self._client.chat.completions.create(
        #     model=self._model,
        #     messages=messages,
        #     response_format={"type": "json_object"},
        #     max_tokens=512,
        # )
        # return self._parse_response(response)
        # ----------------------------------------------------------------------

        logger.debug("Planning processed (stub): %s", trigger[:80])
        return PlanningResponse(
            action=PlanningAction.ANSWER,
            message=f"I heard: {trigger}. (Planning Agent stub — NIM not connected.)",
        )
