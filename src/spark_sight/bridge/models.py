"""Data models for the Spark Sight agent bridge.

Defines the contract between agents: signal types, action types,
prompt state schema, and structured response formats.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentMode(StrEnum):
    """Operating mode for the Ambient Agent."""

    PATROL = "patrol"
    GOAL = "goal"


class AmbientSignal(StrEnum):
    """Signals emitted by the Ambient Agent after processing a frame.

    Patrol mode: CLEAR, WARNING
    Goal mode:   CLEAR, WARNING, PROGRESS, CORRECTION, GOAL_REACHED, FAILURE
    """

    CLEAR = "CLEAR"
    WARNING = "WARNING"
    PROGRESS = "PROGRESS"
    CORRECTION = "CORRECTION"
    GOAL_REACHED = "GOAL_REACHED"
    FAILURE = "FAILURE"


class PlanningAction(StrEnum):
    """Actions the Planning Agent can take in response to user input."""

    SET_GOAL = "set_goal"
    INSPECT = "inspect"
    ANSWER = "answer"
    RESET = "reset"
    REPLAN = "replan"
    FIND_RESTROOM = "find_restroom"


# ---------------------------------------------------------------------------
# Prompt State
# ---------------------------------------------------------------------------


class PromptStateData(BaseModel):
    """Snapshot of the shared prompt state between agents.

    The Planning Agent writes to this; the Ambient Agent reads it.
    """

    mode: AgentMode = AgentMode.PATROL
    base_goal: str = (
        "You are a navigation assistant for a visually impaired user. "
        "Detect safety hazards, warn about close obstacles, read signs. "
        "Speak only when something is actionable."
    )
    active_goal: str | None = None
    nyc_context: str = ""

    @property
    def compiled_prompt(self) -> str:
        """Build the full prompt the Ambient Agent sees each frame."""
        parts = [self.base_goal]
        if self.active_goal:
            parts.append(f"ACTIVE GOAL: {self.active_goal}")
        if self.nyc_context:
            parts.append(f"NYC CONTEXT: {self.nyc_context}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent Responses
# ---------------------------------------------------------------------------


class AmbientResponse(BaseModel):
    """Structured output from the Ambient Agent after processing one frame."""

    signal: AmbientSignal
    message: str = ""
    """Natural-language message to speak to the user (empty for CLEAR)."""
    reasoning: str = ""
    """Internal chain-of-thought (logged, not spoken)."""


class PlanningResponse(BaseModel):
    """Structured output from the Planning Agent after processing a trigger."""

    action: PlanningAction
    message: str = ""
    """Speech to deliver to the user (empty for silent actions)."""
    goal: str | None = None
    """New goal text (only for set_goal / replan)."""
    nyc_context: str | None = None
    """NYC data context to inject (only for set_goal / replan)."""
    inspect_prompt: str | None = None
    """One-shot prompt for the Ambient Agent (only for inspect)."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary extra data (tool call results, debug info, etc.)."""
