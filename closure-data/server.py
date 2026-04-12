"""Street closure data server — hackathon MVP.

Loads the NYC Street Closures CSV into memory on startup and exposes
a single GET /search endpoint for the Planning Agent to query.

Run:
    python closure-data/server.py
    # or from project root:
    python -m uvicorn closure-data.server:app --port 8010

Endpoint:
    GET /search?street=BROADWAY&borough=M&date=2026-04-12&limit=5
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "Street_Closures_due_to_Construction_Activities_by_Block_20260412.csv"

app = FastAPI(title="Street Closures Data Server")

# Module-level DataFrame — loaded once at startup.
_df: pd.DataFrame | None = None

BOROUGH_NAMES = {
    "M": "Manhattan",
    "X": "Bronx",
    "B": "Brooklyn",
    "Q": "Queens",
    "S": "Staten Island",
}


@app.on_event("startup")
def load_data() -> None:
    global _df
    logger.info("Loading CSV from %s", DATA_PATH)
    raw = pd.read_csv(DATA_PATH)
    raw["WORK_START_DATE"] = pd.to_datetime(
        raw["WORK_START_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    raw["WORK_END_DATE"] = pd.to_datetime(
        raw["WORK_END_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    # Normalise street name columns to uppercase stripped strings.
    for col in ("ONSTREETNAME", "FROMSTREETNAME", "TOSTREETNAME"):
        raw[col] = raw[col].str.strip().str.upper()
    raw["BOROUGH_CODE"] = raw["BOROUGH_CODE"].str.strip().str.upper()
    _df = raw
    logger.info("Loaded %d rows", len(_df))


@app.get("/search")
def search(
    street: str | None = Query(None, description="Street name (partial, case-insensitive)"),
    borough: str | None = Query(None, description="Borough code: M X B Q S"),
    date: str | None = Query(None, description="ISO date YYYY-MM-DD — filter active closures on this day"),
    limit: int = Query(5, ge=1, le=50, description="Max results to return"),
) -> JSONResponse:
    """Search active street construction closures."""
    if _df is None:
        return JSONResponse({"error": "Data not loaded"}, status_code=503)

    result = _df

    if street:
        needle = street.strip().upper()
        mask = (
            result["ONSTREETNAME"].str.contains(needle, na=False)
            | result["FROMSTREETNAME"].str.contains(needle, na=False)
            | result["TOSTREETNAME"].str.contains(needle, na=False)
        )
        result = result[mask]

    if borough:
        result = result[result["BOROUGH_CODE"] == borough.strip().upper()]

    if date:
        try:
            dt = pd.Timestamp(date)
            result = result[
                (result["WORK_START_DATE"] <= dt) & (result["WORK_END_DATE"] >= dt)
            ]
        except Exception:
            return JSONResponse({"error": f"Invalid date: {date!r}"}, status_code=400)

    total = len(result)
    records = []
    for _, row in result.head(limit).iterrows():
        records.append(
            {
                "street": row["ONSTREETNAME"],
                "from_street": row["FROMSTREETNAME"],
                "to_street": row["TOSTREETNAME"],
                "borough": row["BOROUGH_CODE"],
                "borough_name": BOROUGH_NAMES.get(row["BOROUGH_CODE"], row["BOROUGH_CODE"]),
                "start_date": row["WORK_START_DATE"].strftime("%Y-%m-%d")
                if pd.notna(row["WORK_START_DATE"])
                else None,
                "end_date": row["WORK_END_DATE"].strftime("%Y-%m-%d")
                if pd.notna(row["WORK_END_DATE"])
                else None,
                "purpose": row["PURPOSE"],
            }
        )

    logger.info(
        "search street=%r borough=%r date=%r → %d/%d results",
        street, borough, date, len(records), total,
    )
    return JSONResponse({"count": total, "shown": len(records), "results": records})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rows": len(_df) if _df is not None else 0}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)
