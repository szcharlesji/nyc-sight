# Street Closure Data Server

Local HTTP service that exposes NYC street construction closure data to the Planning Agent. Loads the CSV into memory once at startup and answers queries in-process — no external network calls required.

---

## Files

| File | Description |
|---|---|
| `server.py` | FastAPI data server (port 8010) |
| `Street_Closures_due_to_Construction_Activities_by_Block_20260412.csv` | Source data — NYC Street Closures due to Construction Activities by Block, snapshot 2026-04-12 (~3 300 rows) |

---

## Running the Server

```bash
# From the project root
python closure-data/server.py
```

The server binds to `0.0.0.0:8010` by default. It prints a log line confirming how many rows were loaded:

```
INFO: Loading CSV from .../Street_Closures_due_to_Construction_Activities_by_Block_20260412.csv
INFO: Loaded 3314 rows
INFO: Uvicorn running on http://0.0.0.0:8010
```

---

## API

### `GET /health`

Returns server status and row count. Use this to verify the server is up and data is loaded.

```bash
curl http://localhost:8010/health
```

```json
{"status": "ok", "rows": 3314}
```

---

### `GET /search`

Search active street construction closures.

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `street` | string | no | Street name — partial, case-insensitive. Matched against the closed street, the from-street, and the to-street. |
| `borough` | string | no | Single-letter borough code: `M` Manhattan · `X` Bronx · `B` Brooklyn · `Q` Queens · `S` Staten Island |
| `date` | string | no | ISO date `YYYY-MM-DD`. Filters to closures active on that day (`start ≤ date ≤ end`). |
| `limit` | integer | no | Maximum results to return. Default `5`, max `50`. |

**Response**

```json
{
  "count": 12,
  "shown": 5,
  "results": [
    {
      "street": "BROADWAY",
      "from_street": "CANAL STREET",
      "to_street": "HOWARD STREET",
      "borough": "M",
      "borough_name": "Manhattan",
      "start_date": "2026-03-26",
      "end_date": "2026-04-30",
      "purpose": "DOT IN-HOUSE PAVING"
    }
  ]
}
```

- `count` — total rows matching the query (before `limit`)
- `shown` — number of results actually returned
- `results` — list of matching closure records

---

## Example Curl Tests

```bash
# Health check
curl -s "http://localhost:8010/health" | python3 -m json.tool

# By street name (partial match)
curl -s "http://localhost:8010/search?street=BROADWAY" | python3 -m json.tool

# Street + borough
curl -s "http://localhost:8010/search?street=SHORE+ROAD&borough=S" | python3 -m json.tool

# Active closures today
curl -s "http://localhost:8010/search?street=BROADWAY&date=2026-04-12" | python3 -m json.tool

# All active closures in Manhattan today, up to 10
curl -s "http://localhost:8010/search?borough=M&date=2026-04-12&limit=10" | python3 -m json.tool
```

---

## Data Source

NYC Open Data — [Street Closures due to Construction Activities by Block](https://data.cityofnewyork.us/Transportation/Street-Closures-due-to-Construction-Activities-by-/478a-yykk).  
Dataset columns used: `ONSTREETNAME`, `FROMSTREETNAME`, `TOSTREETNAME`, `BOROUGH_CODE`, `WORK_START_DATE`, `WORK_END_DATE`, `PURPOSE`.  
Geometry column (`the_geom`) and `UNIQUEID` / `SEGMENTID` / `OFT` are loaded but not exposed by this API.
