"""Planning Agent — intent parser & route planner powered by Nemotron-3-Nano-30B.

Parses user voice transcripts and FAILURE replan triggers via the
Nemotron NIM endpoint (OpenAI-compatible API).  Returns a structured
``PlanningResponse`` with an action and optional speech/goal.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from spark_sight.agents.base import BaseAgent
from spark_sight.bridge.models import PlanningAction, PlanningResponse
from spark_sight.bridge.prompt_state import PromptState
from spark_sight.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_DEFAULT_NIM_BASE_URL = _settings.nemotron.nim_url
_DEFAULT_MODEL = _settings.nemotron.model

_SYSTEM_PROMPT = """\
You are the Planning Agent for Spark Sight, an accessibility assistant for \
visually impaired users navigating New York City.

You receive the user's spoken request (transcribed) and the current system state.
You must decide on ONE action and respond with EXACTLY ONE JSON object:

{
  "action": "<ACTION>",
  "message": "<text to speak to the user, or empty string>",
  "goal": "<new goal text or null>",
  "nyc_context": "<NYC data context string or null>",
  "inspect_prompt": "<one-shot visual question or null>"
}

ACTION must be one of:
- set_goal: Set a new navigation goal for the Ambient Agent. Provide goal \
and optionally nyc_context. message should confirm the goal to the user.
- inspect: Ask the Ambient Agent to look at the current camera frame and \
answer a visual question. Provide inspect_prompt.
- answer: Answer the user directly without vision. Only message is needed.
- reset: Clear the current navigation goal. message should confirm.
- replan: Create a new plan after a failure. Provide goal and message.

Rules:
1. If the user asks about what they can see, use "inspect" with a clear prompt.
2. If the user wants to go somewhere or navigate, use "set_goal".
3. If the user says "cancel", "stop", or "never mind", use "reset".
4. If the input is a failure reason (from the Ambient Agent), use "replan".
5. For general questions not requiring vision, use "answer".
6. Messages must be concise (1-2 sentences), spoken aloud to a blind person.
7. Output valid JSON only. No markdown, no explanation outside the JSON.
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
        state_snapshot = self._state.get_snapshot()

        if self._client is None:
            logger.debug("Planning processed (stub): %s", trigger[:80])
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message=f"I heard: {trigger}. (Planning Agent stub — NIM not connected.)",
            )

        # Construct the user message with trigger + current system state.
        if failure_reason:
            user_content = (
                f"The Ambient Agent reported FAILURE: {failure_reason}\n\n"
                f"Current mode: {state_snapshot.mode}\n"
                f"Previous goal: {state_snapshot.active_goal or 'none'}\n"
                f"NYC context: {state_snapshot.nyc_context or 'none'}\n\n"
                f"Please replan with an alternative approach."
            )
        else:
            user_content = (
                f"User said: {transcript}\n\n"
                f"Current mode: {state_snapshot.mode}\n"
                f"Active goal: {state_snapshot.active_goal or 'none'}\n"
                f"NYC context: {state_snapshot.nyc_context or 'none'}"
            )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            return self._parse_response(response)
        except Exception:
            logger.exception("NIM inference failed for PlanningAgent")
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message="Sorry, I'm having trouble processing that right now.",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove ``<think>...</think>`` reasoning blocks from model output.

        Nemotron-3-Nano-30B uses chain-of-thought wrapped in ``<think>``
        tags.  This must be stripped before JSON parsing.
        """
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_response(self, response: Any) -> PlanningResponse:
        """Parse a NIM chat completion into a :class:`PlanningResponse`."""
        raw: str = response.choices[0].message.content or ""
        raw = self._strip_think(raw)

        # LLMs sometimes wrap JSON in markdown code fences.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
            action = PlanningAction(data.get("action", "answer"))
            return PlanningResponse(
                action=action,
                message=data.get("message", ""),
                goal=data.get("goal"),
                nyc_context=data.get("nyc_context"),
                inspect_prompt=data.get("inspect_prompt"),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("Failed to parse Planning response: %s", raw[:200])
            # Fall back to treating the raw text as a direct answer.
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message=raw[:200] if raw else "I couldn't understand that.",
            )
