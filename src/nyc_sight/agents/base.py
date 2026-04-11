"""Abstract base class for Spark Sight agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Interface that both the Ambient and Planning agents implement.

    Lifecycle:  ``start()`` → repeated ``process()`` calls → ``stop()``.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources (model loading, NIM client, etc.)."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources gracefully."""

    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> Any:
        """Run one processing step.

        For the **Ambient Agent** this is one frame evaluation.
        For the **Planning Agent** this is one trigger (voice transcript
        or failure signal).

        Parameters
        ----------
        input_data:
            Agent-specific payload.  See each agent's README for the schema.

        Returns
        -------
        An agent-specific response model (``AmbientResponse`` or
        ``PlanningResponse``).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name for logging."""
