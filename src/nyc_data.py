from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEMO_LAT = 40.7780
DEMO_LON = -73.9812


class NYCData:
    """Stub — NYC Open Data integration disabled for now."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        logger.info("NYC Open Data: stub mode (no data loaded)")

    def get_context(
        self, lat: float = DEMO_LAT, lon: float = DEMO_LON, radius: float = 0.002
    ) -> str:
        return "No nearby condition data available."
