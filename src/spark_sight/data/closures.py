"""NYC street closure data — local server client.

Queries the closure data server (closure-data/server.py) for active
street construction closures.  Returns dict results or error sentinels
so callers never need to handle exceptions.
"""

from __future__ import annotations

import datetime
import logging

import httpx

logger = logging.getLogger(__name__)


async def search_closures(
    street: str,
    borough: str | None = None,
    date: str | None = None,
    limit: int = 5,
    base_url: str = "http://localhost:8010",
) -> dict:
    """Query the local closure data server.

    Returns the parsed JSON on success (keys: ``count``, ``shown``,
    ``results``).  On failure returns a dict with an ``"error"`` key
    instead of raising.
    """
    if date is None:
        date = datetime.date.today().isoformat()

    params: dict[str, str | int] = {"street": street, "date": date, "limit": limit}
    if borough:
        params["borough"] = borough

    async with httpx.AsyncClient(timeout=3) as client:
        try:
            resp = await client.get(f"{base_url}/search", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            logger.warning("Closure data server unreachable at %s", base_url)
            return {"error": "server_unreachable"}
        except httpx.TimeoutException:
            logger.warning("Closure data server timed out")
            return {"error": "timeout"}
        except httpx.HTTPError:
            logger.exception("Closure data request failed")
            return {"error": "request_failed"}
