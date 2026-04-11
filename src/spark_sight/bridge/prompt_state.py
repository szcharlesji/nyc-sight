"""Thread-safe shared prompt state.

The PromptState is the single mutable contract between the Planning Agent
(writer) and the Ambient Agent (reader).  All mutations are atomic via a
threading lock so the Ambient Agent never sees a half-written state.
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import UTC, datetime

from spark_sight.bridge.models import AgentMode, PromptStateData

logger = logging.getLogger(__name__)


class PromptState:
    """Thread-safe wrapper around :class:`PromptStateData`.

    Usage::

        state = PromptState()
        state.set_goal("Guide user north on Broadway", nyc_context="...")
        prompt = state.get_compiled_prompt()   # Ambient Agent reads this
        state.reset_goal()                     # revert to patrol
    """

    def __init__(self, base_goal: str | None = None) -> None:
        self._lock = threading.Lock()
        init_kwargs: dict = {}
        if base_goal is not None:
            init_kwargs["base_goal"] = base_goal
        self._data = PromptStateData(**init_kwargs)
        self._last_updated = datetime.now(UTC)

    # -- reads (Ambient Agent) -----------------------------------------------

    def get_compiled_prompt(self) -> str:
        """Return the full prompt string for the current frame."""
        with self._lock:
            return self._data.compiled_prompt

    def get_mode(self) -> AgentMode:
        with self._lock:
            return self._data.mode

    def get_snapshot(self) -> PromptStateData:
        """Return a deep copy of the current state (safe to inspect)."""
        with self._lock:
            return deepcopy(self._data)

    @property
    def last_updated(self) -> datetime:
        with self._lock:
            return self._last_updated

    # -- writes (Planning Agent via Orchestrator) ----------------------------

    def set_goal(self, goal: str, nyc_context: str = "") -> None:
        """Atomically transition to Goal mode with a new active goal."""
        with self._lock:
            self._data.mode = AgentMode.GOAL
            self._data.active_goal = goal
            self._data.nyc_context = nyc_context
            self._last_updated = datetime.now(UTC)
            logger.info("Goal set: %s", goal)

    def reset_goal(self) -> None:
        """Atomically revert to Patrol mode, clearing the active goal."""
        with self._lock:
            self._data.mode = AgentMode.PATROL
            self._data.active_goal = None
            self._data.nyc_context = ""
            self._last_updated = datetime.now(UTC)
            logger.info("Goal reset — patrol mode")

    def update_nyc_context(self, nyc_context: str) -> None:
        """Update NYC context data without changing the goal."""
        with self._lock:
            self._data.nyc_context = nyc_context
            self._last_updated = datetime.now(UTC)
