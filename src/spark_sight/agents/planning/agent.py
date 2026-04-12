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
You must decide on ONE action and respond with EXACTLY ONE JSON object.

IMPORTANT: Output ONLY the JSON object. No thinking, no explanation, no \
markdown fences. Just the raw JSON.

{
  "action": "<ACTION>",
  "message": "<text to speak to the user, or empty string>",
  "goal": "<new goal text or null>",
  "nyc_context": "<NYC data context string or null>",
  "inspect_prompt": "<one-shot visual question or null>"
}

ACTION must be one of:
- set_goal: Program the Ambient Agent with a CONTINUOUS monitoring goal. \
The Ambient Agent will evaluate EVERY camera frame against this goal and \
report PROGRESS, WARNING, CORRECTION, GOAL_REACHED, or FAILURE. Use this \
for navigation ("take me to the subway"), watching for something ("tell me \
when you see a face", "warn me about obstacles"), or any ongoing task. \
Provide goal (clear instruction for the Ambient Agent) and optionally \
nyc_context. message should confirm to the user.
- inspect: One-shot visual question — look at the CURRENT camera frame \
RIGHT NOW and answer a question. Use ONLY for immediate queries like \
"what do you see?", "read that sign", "is there a door ahead?". \
Provide inspect_prompt. Does NOT persist — the Ambient Agent returns to \
its previous mode after answering.
- answer: Answer the user directly without vision. Only message is needed.
- reset: Clear the current goal. The Ambient Agent reverts to Patrol Mode.
- replan: Create a new goal after a FAILURE signal. Provide goal and message.
- find_restroom: Find the nearest public restroom using NYC Open Data and \
the user's GPS location. The system will look up nearby operational restrooms \
automatically. No extra fields needed — just set action to "find_restroom" \
and optionally a message like "Let me find the nearest restroom for you."
- find_closure: Search for active NYC street construction closures. Add \
"closure_street" (street name in ALL CAPS, e.g. "BROADWAY") and optionally \
"closure_borough" (single letter: M=Manhattan, X=Bronx, B=Brooklyn, \
Q=Queens, S=Staten Island) to your JSON output. Use when the user asks \
about blocked roads, construction activity, or whether a street is passable. \
Set message to a short spoken confirmation like \
"Let me check for closures on Broadway."

Rules:
1. If the user wants CONTINUOUS monitoring or to be notified about something \
("remind me", "tell me when", "watch for", "warn me", "guide me"), use \
"set_goal". The goal text should be a clear instruction for the Ambient \
Agent describing what to watch for and when to signal.
2. If the user asks a ONE-TIME visual question about what's in front of them \
right now ("what do you see?", "read that sign"), use "inspect".
3. If the user wants to go somewhere or navigate, use "set_goal".
4. If the user says "cancel", "stop", or "never mind", use "reset".
5. If the input is a failure reason (from the Ambient Agent), use "replan".
6. For general questions not requiring vision, use "answer".
7. If the user asks about restrooms, bathrooms, or "where can I go", use \
"find_restroom".
8. If the user asks about road closures, construction, blocked streets, or \
whether a specific street is passable, use "find_closure".
9. Messages must be concise (1-2 sentences), spoken aloud to a blind person.
10. Output valid JSON only. No markdown, no explanation outside the JSON.
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
            planning_response = self._parse_response(response)
        except Exception:
            logger.exception("NIM inference failed for PlanningAgent")
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message="Sorry, I'm having trouble processing that right now.",
            )

        # If Call 1 produced find_closure, intercept and run the two-step
        # data fetch + synthesis before returning to the orchestrator.
        # All other actions pass through immediately — no second LLM call.
        if planning_response.action == PlanningAction.FIND_CLOSURE:
            return await self._fetch_and_synthesize_closure(
                planning_response, trigger
            )

        return planning_response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove chain-of-thought reasoning from model output.

        Nemotron-3-Nano-30B emits thinking text that may or may not be
        wrapped in ``<think>`` tags.  Handles both cases:
        - ``<think>...reasoning...</think>{json}``
        - ``...reasoning...\n</think>\n{json}``  (no opening tag)
        """
        import re
        # Case 1: proper <think>...</think> tags.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
        # Case 2: no opening tag — everything before closing tag is thinking.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if "</thought>" in text:
            text = text.split("</thought>", 1)[1].strip()
        return text

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
                # Carry closure search params extracted by the LLM so the
                # synthesis step can use them without re-parsing.
                metadata={
                    "closure_street": data.get("closure_street"),
                    "closure_borough": data.get("closure_borough"),
                },
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("Failed to parse Planning response: %s", raw[:200])
            # Fall back to treating the raw text as a direct answer.
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message=raw[:200] if raw else "I couldn't understand that.",
            )

    async def _fetch_and_synthesize_closure(
        self,
        call1_response: PlanningResponse,
        original_query: str,
    ) -> PlanningResponse:
        """Two-step closure lookup: fetch data then synthesize a spoken reply.

        Called only when Call 1 returned ``find_closure``.  All other actions
        bypass this method entirely and go straight to the orchestrator.

        Step A — query the local closure data server.
        Step B — [Call 2] ask the LLM to convert the raw data into a natural
                 spoken sentence for the user.
        """
        from spark_sight.config import get_settings
        from spark_sight.data.closures import search_closures

        street: str = call1_response.metadata.get("closure_street") or ""
        borough: str | None = call1_response.metadata.get("closure_borough")

        if not street:
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message="I need a street name to check for closures. Which street?",
            )

        # --- Step A: query data server ---
        settings = get_settings()
        data = await search_closures(
            street=street,
            borough=borough,
            base_url=settings.closure.server_url,
        )
        logger.info(
            "[Closure] street=%r borough=%r → %d results (error=%s)",
            street, borough, data.get("count", 0), data.get("error"),
        )

        if data.get("error") == "server_unreachable":
            return PlanningResponse(
                action=PlanningAction.ANSWER,
                message="I can't reach the closure data right now. Please try again in a moment.",
            )

        # --- Step B: [Call 2] synthesize spoken response ---
        if data["count"] == 0:
            data_summary = f"No active construction closures found for {street}."
        else:
            rows = data["results"]
            lines = [
                f"{r['street']} from {r['from_street']} to {r['to_street']} "
                f"({r['borough_name']}), {r['purpose']}, until {r['end_date']}"
                for r in rows
            ]
            data_summary = (
                f"{data['count']} active closure(s) found"
                + (f" (showing {data['shown']})" if data["shown"] < data["count"] else "")
                + ":\n"
                + "\n".join(f"- {line}" for line in lines)
            )

        synthesis_system = (
            "You are a concise voice assistant for a blind user walking in NYC. "
            "Convert the street closure data below into 1-3 natural spoken sentences. "
            "Mention the street, cross streets, work type, and end date. "
            "If there are more results than shown, say so briefly. "
            "Output ONLY the spoken text — no JSON, no markdown, no preamble."
        )
        synthesis_user = (
            f'User asked: "{original_query}"\n\n'
            f"Data:\n{data_summary}"
        )

        try:
            synthesis_resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": synthesis_system},
                    {"role": "user",   "content": synthesis_user},
                ],
                max_tokens=160,
                temperature=0.3,
            )
            speech = self._strip_think(
                synthesis_resp.choices[0].message.content or ""
            ).strip()
            if not speech:
                speech = data_summary  # last-resort fallback
        except Exception:
            logger.exception("Call 2 synthesis failed — using raw data summary")
            speech = data_summary

        return PlanningResponse(action=PlanningAction.ANSWER, message=speech)
