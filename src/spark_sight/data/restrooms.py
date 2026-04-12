"""NYC Open Data — public restroom lookup.

Queries the NYC Public Restrooms dataset (Socrata) and returns the nearest
operational restrooms to a given GPS coordinate.

Dataset: https://dev.socrata.com/foundry/data.cityofnewyork.us/i7jb-7jku
"""

from __future__ import annotations

import logging
import math
import os

import httpx
from dotenv import load_dotenv

load_dotenv(override=False)

logger = logging.getLogger(__name__)

_ENDPOINT = (
    "https://data.cityofnewyork.us/api/v3/views/i7jb-7jku/query.json"
)
_QUERY = "SELECT * WHERE (`status` = 'Operational')"


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in feet between two GPS coordinates."""
    R = 20_902_231  # Earth radius in feet
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def find_nearby_restrooms(
    lat: float,
    lng: float,
    limit: int = 5,
) -> list[dict]:
    """Fetch operational public restrooms sorted by distance.

    Returns up to *limit* restrooms, each a dict with keys:
    ``name``, ``address``, ``distance_ft``, ``accessible``, ``hours``.
    """
    key_id = os.environ.get("OPENDATA_KEY_ID", "")
    key_secret = os.environ.get("OPENDATA_KEY_SECRET", "")

    auth = httpx.BasicAuth(key_id, key_secret) if key_id and key_secret else None

    async with httpx.AsyncClient(timeout=10, auth=auth) as client:
        try:
            resp = await client.get(
                _ENDPOINT,
                params={"query": _QUERY},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("NYC Open Data restroom query failed")
            return []

    rows = data if isinstance(data, list) else data.get("rows", data.get("data", []))
    if not rows:
        logger.warning("No restroom data returned")
        return []

    results = []
    for row in rows:
        try:
            rlat = float(row.get("latitude") or row.get("lat") or 0)
            rlng = float(row.get("longitude") or row.get("lon") or row.get("lng") or 0)
        except (TypeError, ValueError):
            continue
        if rlat == 0 or rlng == 0:
            continue

        dist = _haversine_ft(lat, lng, rlat, rlng)
        results.append({
            "name": row.get("facility_name") or row.get("name") or "Public Restroom",
            "location_type": row.get("location_type") or "",
            "operator": row.get("operator") or "",
            "distance_ft": round(dist),
            "accessible": row.get("changing_stations") or "Unknown",
            "hours": row.get("hours_of_operation") or "Unknown",
            "open": row.get("open") or "",
            "lat": rlat,
            "lng": rlng,
        })

    results.sort(key=lambda r: r["distance_ft"])
    return results[:limit]
