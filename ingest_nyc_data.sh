#!/usr/bin/env bash
# =============================================================================
# NYC Open Data → Postgres Ingest Pipeline
# Target: sparksight @ 100.90.132.109 (Tailscale)
#
# Downloads 8 NYC Open Data CSVs via SODA API, transforms them, and bulk-loads
# into a local Postgres instance with proper schemas, types, and indexes.
#
# Usage:
#   chmod +x ingest_nyc_data.sh
#   ./ingest_nyc_data.sh              # full run (all datasets)
#   ./ingest_nyc_data.sh 311 pluto    # selective run (named datasets)
#
# Prerequisites:
#   - psql (PostgreSQL client)
#   - curl
#   - Network access to 100.90.132.109:5432 via Tailscale
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
DB_HOST="100.90.132.109"
DB_PORT="5432"
DB_NAME="sparksight"
DB_USER="postgres"
DB_PASS="postgres"

export PGPASSWORD="$DB_PASS"
PSQL="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -v ON_ERROR_STOP=1"

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ingest_${TIMESTAMP}.log"

# SODA API limits — 311 is huge so we page it; others fit in one shot
SODA_BASE="https://data.cityofnewyork.us/api/views"
SODA_RESOURCE_BASE="https://data.cityofnewyork.us/resource"
SODA_APP_TOKEN="${SODA_APP_TOKEN:-}" # optional, set for higher rate limits

# 311 row limit per SODA page (API max is 50000 per request)
PAGE_LIMIT=50000

# ── Logging ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*" >&2; }
die() {
  err "$*"
  exit 1
}

# ── Utilities ────────────────────────────────────────────────────────────────

# Download CSV from SODA views endpoint (full export)
download_csv() {
  local name="$1" code="$2" url
  url="${SODA_BASE}/${code}/rows.csv?accessType=DOWNLOAD"
  local outfile="${DATA_DIR}/${name}.csv"

  if [[ -f "$outfile" && -s "$outfile" ]]; then
    log "  ↳ ${name}.csv already exists ($(wc -l <"$outfile") lines), skipping download"
    return 0
  fi

  log "  ↳ Downloading ${name} (${code})..."
  local curl_opts=(-fSL --retry 3 --retry-delay 5 -o "$outfile")
  [[ -n "$SODA_APP_TOKEN" ]] && curl_opts+=(-H "X-App-Token: $SODA_APP_TOKEN")

  if ! curl "${curl_opts[@]}" "$url" 2>>"$LOG_FILE"; then
    err "Failed to download ${name}"
    rm -f "$outfile"
    return 1
  fi

  local lines
  lines=$(wc -l <"$outfile")
  log "  ↳ Downloaded ${lines} lines"
}

# Download large dataset via SODA resource endpoint with pagination (for 311)
download_csv_paged() {
  local name="$1" code="$2"
  local outfile="${DATA_DIR}/${name}.csv"
  local tmpfile="${DATA_DIR}/${name}_tmp.csv"

  if [[ -f "$outfile" && -s "$outfile" ]]; then
    local existing_lines
    existing_lines=$(wc -l <"$outfile")
    if ((existing_lines > 1000000)); then
      log "  ↳ ${name}.csv already exists (${existing_lines} lines), skipping download"
      return 0
    fi
  fi

  log "  ↳ Downloading ${name} (${code}) with pagination..."

  local offset=0
  local first=true
  local page_rows=0
  local total_rows=0

  rm -f "$outfile"

  while true; do
    local url="${SODA_RESOURCE_BASE}/${code}.csv?\$limit=${PAGE_LIMIT}&\$offset=${offset}&\$order=:id"
    local curl_opts=(-fSL --retry 3 --retry-delay 5 -o "$tmpfile")
    [[ -n "$SODA_APP_TOKEN" ]] && curl_opts+=(-H "X-App-Token: $SODA_APP_TOKEN")

    if ! curl "${curl_opts[@]}" "$url" 2>>"$LOG_FILE"; then
      err "Failed to download ${name} at offset ${offset}"
      rm -f "$tmpfile"
      return 1
    fi

    page_rows=$(tail -n +2 "$tmpfile" | wc -l)

    if $first; then
      # Keep header from first page
      cat "$tmpfile" >>"$outfile"
      first=false
    else
      # Skip header on subsequent pages
      tail -n +2 "$tmpfile" >>"$outfile"
    fi

    total_rows=$((total_rows + page_rows))
    log "    ↳ Page at offset ${offset}: ${page_rows} rows (total: ${total_rows})"

    if ((page_rows < PAGE_LIMIT)); then
      break
    fi

    offset=$((offset + PAGE_LIMIT))
  done

  rm -f "$tmpfile"
  log "  ↳ Downloaded ${total_rows} total rows for ${name}"
}

# Run SQL from string
run_sql() {
  $PSQL -c "$1" 2>>"$LOG_FILE"
}

# Run SQL from file
run_sql_file() {
  $PSQL -f "$1" 2>>"$LOG_FILE"
}

# Sanitize CSV: fix encoding, remove NUL bytes, normalize line endings
sanitize_csv() {
  local file="$1"
  log "  ↳ Sanitizing ${file}..."

  # Remove NUL bytes and normalize line endings
  sed -i.bak $'s/\x00//g' "$file"
  rm -f "${file}.bak"

  # Convert Windows line endings if present
  if file "$file" | grep -q CRLF; then
    sed -i.bak 's/\r$//' "$file"
    rm -f "${file}.bak"
  fi
}

# Bulk load CSV into Postgres using \copy
bulk_load() {
  local table="$1" file="$2"
  local lines
  lines=$(($(wc -l <"$file") - 1)) # subtract header
  log "  ↳ Loading ${lines} rows into ${table}..."

  # Use \copy (client-side) so we don't need server filesystem access
  $PSQL -c "\copy ${table} FROM '${file}' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')" 2>>"$LOG_FILE"

  local count
  count=$(run_sql "SELECT COUNT(*) FROM ${table};" -t -A)
  log "  ↳ Loaded. Row count in ${table}: ${count}"
}

# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

create_schemas() {
  log "Creating schema 'nyc' and extensions..."

  run_sql "
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE SCHEMA IF NOT EXISTS nyc;
    "
}

# ── 1. 311 Service Requests ──────────────────────────────────────────────────
create_table_311() {
  run_sql "
        DROP TABLE IF EXISTS nyc.service_requests_311 CASCADE;
        CREATE TABLE nyc.service_requests_311 (
            unique_key              TEXT PRIMARY KEY,
            created_date            TIMESTAMP,
            closed_date             TIMESTAMP,
            agency                  TEXT,
            agency_name             TEXT,
            complaint_type          TEXT,
            descriptor              TEXT,
            location_type           TEXT,
            incident_zip            TEXT,
            incident_address        TEXT,
            street_name             TEXT,
            cross_street_1          TEXT,
            cross_street_2          TEXT,
            intersection_street_1   TEXT,
            intersection_street_2   TEXT,
            address_type            TEXT,
            city                    TEXT,
            landmark                TEXT,
            facility_type           TEXT,
            status                  TEXT,
            due_date                TIMESTAMP,
            resolution_description  TEXT,
            resolution_action_updated_date TIMESTAMP,
            community_board         TEXT,
            bbl                     TEXT,
            borough                 TEXT,
            x_coordinate            DOUBLE PRECISION,
            y_coordinate            DOUBLE PRECISION,
            open_data_channel_type  TEXT,
            park_facility_name      TEXT,
            park_borough            TEXT,
            vehicle_type            TEXT,
            taxi_company_borough    TEXT,
            taxi_pick_up_location   TEXT,
            bridge_highway_name     TEXT,
            bridge_highway_direction TEXT,
            road_ramp               TEXT,
            bridge_highway_segment  TEXT,
            latitude                DOUBLE PRECISION,
            longitude               DOUBLE PRECISION,
            location                TEXT
        );
    "
}

index_311() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_311_created     ON nyc.service_requests_311 (created_date);
        CREATE INDEX IF NOT EXISTS idx_311_complaint    ON nyc.service_requests_311 (complaint_type);
        CREATE INDEX IF NOT EXISTS idx_311_borough      ON nyc.service_requests_311 (borough);
        CREATE INDEX IF NOT EXISTS idx_311_zip          ON nyc.service_requests_311 (incident_zip);
        CREATE INDEX IF NOT EXISTS idx_311_status       ON nyc.service_requests_311 (status);
        CREATE INDEX IF NOT EXISTS idx_311_latlon       ON nyc.service_requests_311 (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_311_complaint_trgm ON nyc.service_requests_311
            USING gin (complaint_type gin_trgm_ops);
    "
}

# ── 2. Motor Vehicle Collisions ──────────────────────────────────────────────
create_table_collisions() {
  run_sql "
        DROP TABLE IF EXISTS nyc.motor_vehicle_collisions CASCADE;
        CREATE TABLE nyc.motor_vehicle_collisions (
            collision_id                    TEXT PRIMARY KEY,
            crash_date                      TEXT,
            crash_time                      TEXT,
            borough                         TEXT,
            zip_code                        TEXT,
            latitude                        DOUBLE PRECISION,
            longitude                       DOUBLE PRECISION,
            location                        TEXT,
            on_street_name                  TEXT,
            cross_street_name               TEXT,
            off_street_name                 TEXT,
            number_of_persons_injured       INTEGER,
            number_of_persons_killed        INTEGER,
            number_of_pedestrians_injured   INTEGER,
            number_of_pedestrians_killed    INTEGER,
            number_of_cyclist_injured       INTEGER,
            number_of_cyclist_killed        INTEGER,
            number_of_motorist_injured      INTEGER,
            number_of_motorist_killed       INTEGER,
            contributing_factor_vehicle_1   TEXT,
            contributing_factor_vehicle_2   TEXT,
            contributing_factor_vehicle_3   TEXT,
            contributing_factor_vehicle_4   TEXT,
            contributing_factor_vehicle_5   TEXT,
            vehicle_type_code_1             TEXT,
            vehicle_type_code_2             TEXT,
            vehicle_type_code_3             TEXT,
            vehicle_type_code_4             TEXT,
            vehicle_type_code_5             TEXT
        );
    "
}

index_collisions() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_collisions_date      ON nyc.motor_vehicle_collisions (crash_date);
        CREATE INDEX IF NOT EXISTS idx_collisions_borough    ON nyc.motor_vehicle_collisions (borough);
        CREATE INDEX IF NOT EXISTS idx_collisions_zip        ON nyc.motor_vehicle_collisions (zip_code);
        CREATE INDEX IF NOT EXISTS idx_collisions_latlon     ON nyc.motor_vehicle_collisions (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_collisions_ped_killed ON nyc.motor_vehicle_collisions (number_of_pedestrians_killed)
            WHERE number_of_pedestrians_killed > 0;
    "
}

# ── 3. Sidewalk Inspections ──────────────────────────────────────────────────
create_table_sidewalks() {
  run_sql "
        DROP TABLE IF EXISTS nyc.sidewalk_inspections CASCADE;
        CREATE TABLE nyc.sidewalk_inspections (
            objectid                TEXT,
            boro                    TEXT,
            block                   TEXT,
            lot                     TEXT,
            bbl                     TEXT,
            inspection_date         TEXT,
            result_date             TEXT,
            violation_number        TEXT,
            penalty_applied         TEXT,
            penalty_adjusted        TEXT,
            paid_late_penalty       TEXT,
            penalty_balance         TEXT,
            amount_invoiced         TEXT,
            amount_paid             TEXT,
            paid_date               TEXT,
            violation_status        TEXT,
            current_status          TEXT,
            status_date             TEXT,
            cf_number               TEXT
        );
    "
}

index_sidewalks() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_sidewalk_bbl        ON nyc.sidewalk_inspections (bbl);
        CREATE INDEX IF NOT EXISTS idx_sidewalk_boro        ON nyc.sidewalk_inspections (boro);
        CREATE INDEX IF NOT EXISTS idx_sidewalk_status      ON nyc.sidewalk_inspections (violation_status);
        CREATE INDEX IF NOT EXISTS idx_sidewalk_insp_date   ON nyc.sidewalk_inspections (inspection_date);
    "
}

# ── 4. Street Light Conditions ───────────────────────────────────────────────
create_table_streetlights() {
  run_sql "
        DROP TABLE IF EXISTS nyc.street_light_conditions CASCADE;
        CREATE TABLE nyc.street_light_conditions (
            unique_key              TEXT,
            created_date            TEXT,
            closed_date             TEXT,
            agency                  TEXT,
            agency_name             TEXT,
            complaint_type          TEXT,
            descriptor              TEXT,
            location_type           TEXT,
            incident_zip            TEXT,
            incident_address        TEXT,
            street_name             TEXT,
            cross_street_1          TEXT,
            cross_street_2          TEXT,
            intersection_street_1   TEXT,
            intersection_street_2   TEXT,
            address_type            TEXT,
            city                    TEXT,
            landmark                TEXT,
            facility_type           TEXT,
            status                  TEXT,
            due_date                TEXT,
            resolution_description  TEXT,
            resolution_action_updated_date TEXT,
            community_board         TEXT,
            borough                 TEXT,
            open_data_channel_type  TEXT,
            park_facility_name      TEXT,
            park_borough            TEXT,
            latitude                DOUBLE PRECISION,
            longitude               DOUBLE PRECISION,
            location                TEXT
        );
    "
}

index_streetlights() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_streetlight_status   ON nyc.street_light_conditions (status);
        CREATE INDEX IF NOT EXISTS idx_streetlight_borough  ON nyc.street_light_conditions (borough);
        CREATE INDEX IF NOT EXISTS idx_streetlight_zip      ON nyc.street_light_conditions (incident_zip);
        CREATE INDEX IF NOT EXISTS idx_streetlight_latlon   ON nyc.street_light_conditions (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
    "
}

# ── 5. PLUTO ─────────────────────────────────────────────────────────────────
create_table_pluto() {
  run_sql "
        DROP TABLE IF EXISTS nyc.pluto CASCADE;
        CREATE TABLE nyc.pluto (
            borough                 TEXT,
            block                   TEXT,
            lot                     TEXT,
            cd                      TEXT,
            bct2020                 TEXT,
            bctcb2020               TEXT,
            ct2010                  TEXT,
            cb2010                  TEXT,
            schooldist              TEXT,
            council                 TEXT,
            zipcode                 TEXT,
            firecomp                TEXT,
            policeprct              TEXT,
            healtharea              TEXT,
            sanitboro               TEXT,
            sanitsub                TEXT,
            address                 TEXT,
            zonedist1               TEXT,
            zonedist2               TEXT,
            zonedist3               TEXT,
            zonedist4               TEXT,
            overlay1                TEXT,
            overlay2                TEXT,
            spdist1                 TEXT,
            spdist2                 TEXT,
            spdist3                 TEXT,
            ltdheight               TEXT,
            splitzone               TEXT,
            bldgclass               TEXT,
            landuse                 TEXT,
            easession               TEXT,
            ownertype               TEXT,
            ownername               TEXT,
            lotarea                 DOUBLE PRECISION,
            bldgarea                DOUBLE PRECISION,
            comarea                 DOUBLE PRECISION,
            resarea                 DOUBLE PRECISION,
            officearea              DOUBLE PRECISION,
            retailarea              DOUBLE PRECISION,
            garagearea              DOUBLE PRECISION,
            strgearea               DOUBLE PRECISION,
            factryarea              DOUBLE PRECISION,
            otherarea               DOUBLE PRECISION,
            areasource              TEXT,
            numbldgs                INTEGER,
            numfloors               DOUBLE PRECISION,
            unitsres                INTEGER,
            unitstotal              INTEGER,
            lotfront                DOUBLE PRECISION,
            lotdepth                DOUBLE PRECISION,
            bldgfront               DOUBLE PRECISION,
            bldgdepth               DOUBLE PRECISION,
            ext                     TEXT,
            proxcode                TEXT,
            irrlotcode              TEXT,
            lottype                 TEXT,
            bsmtcode                TEXT,
            assessland              DOUBLE PRECISION,
            assesstot               DOUBLE PRECISION,
            exempttot               DOUBLE PRECISION,
            yearbuilt               INTEGER,
            yearalter1              INTEGER,
            yearalter2              INTEGER,
            histdist                TEXT,
            landmark                TEXT,
            builtfar                DOUBLE PRECISION,
            residfar                DOUBLE PRECISION,
            commfar                 DOUBLE PRECISION,
            facilfar                DOUBLE PRECISION,
            borocode                TEXT,
            bbl                     TEXT,
            condono                 TEXT,
            tract2010               TEXT,
            xcoord                  DOUBLE PRECISION,
            ycoord                  DOUBLE PRECISION,
            zonemap                 TEXT,
            zmcode                  TEXT,
            sanborn                 TEXT,
            taxmap                  TEXT,
            edesignum               TEXT,
            appbbl                  TEXT,
            appdate                 TEXT,
            mappluto_f              TEXT,
            plutomapid              TEXT,
            firm07_flag             TEXT,
            pfirm15_flag            TEXT,
            version                 TEXT,
            dcpedited               TEXT,
            latitude                DOUBLE PRECISION,
            longitude               DOUBLE PRECISION,
            notes                   TEXT
        );
    "
}

index_pluto() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_pluto_bbl       ON nyc.pluto (bbl);
        CREATE INDEX IF NOT EXISTS idx_pluto_zipcode   ON nyc.pluto (zipcode);
        CREATE INDEX IF NOT EXISTS idx_pluto_landuse   ON nyc.pluto (landuse);
        CREATE INDEX IF NOT EXISTS idx_pluto_borough   ON nyc.pluto (borocode);
        CREATE INDEX IF NOT EXISTS idx_pluto_bldgclass ON nyc.pluto (bldgclass);
        CREATE INDEX IF NOT EXISTS idx_pluto_latlon    ON nyc.pluto (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
    "
}

# ── 6. DOB Job Filings ──────────────────────────────────────────────────────
create_table_dob() {
  run_sql "
        DROP TABLE IF EXISTS nyc.dob_job_filings CASCADE;
        CREATE TABLE nyc.dob_job_filings (
            job_filing_number       TEXT,
            job_number              TEXT,
            filing_date             TEXT,
            job_status              TEXT,
            job_status_descrp       TEXT,
            job_type                TEXT,
            filing_status           TEXT,
            community_board         TEXT,
            borough                 TEXT,
            house_number            TEXT,
            street_name             TEXT,
            city                    TEXT,
            state                   TEXT,
            zip                     TEXT,
            block                   TEXT,
            lot                     TEXT,
            bin_number              TEXT,
            bbl                     TEXT,
            building_type           TEXT,
            applicant_first_name    TEXT,
            applicant_last_name     TEXT,
            applicant_business_name TEXT,
            applicant_license_number TEXT,
            owner_first_name        TEXT,
            owner_last_name         TEXT,
            owner_business_name     TEXT,
            estimated_job_costs     TEXT,
            work_type               TEXT,
            permit_type             TEXT,
            gis_latitude            DOUBLE PRECISION,
            gis_longitude           DOUBLE PRECISION
        );
    "
}

index_dob() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_dob_filing_date  ON nyc.dob_job_filings (filing_date);
        CREATE INDEX IF NOT EXISTS idx_dob_borough      ON nyc.dob_job_filings (borough);
        CREATE INDEX IF NOT EXISTS idx_dob_bbl          ON nyc.dob_job_filings (bbl);
        CREATE INDEX IF NOT EXISTS idx_dob_job_type     ON nyc.dob_job_filings (job_type);
        CREATE INDEX IF NOT EXISTS idx_dob_status       ON nyc.dob_job_filings (job_status);
        CREATE INDEX IF NOT EXISTS idx_dob_latlon       ON nyc.dob_job_filings (gis_latitude, gis_longitude)
            WHERE gis_latitude IS NOT NULL AND gis_longitude IS NOT NULL;
    "
}

# ── 7. Tree Census 2015 ─────────────────────────────────────────────────────
create_table_trees() {
  run_sql "
        DROP TABLE IF EXISTS nyc.tree_census_2015 CASCADE;
        CREATE TABLE nyc.tree_census_2015 (
            tree_id                 INTEGER PRIMARY KEY,
            block_id                INTEGER,
            created_at              TEXT,
            tree_dbh                INTEGER,
            stump_diam              INTEGER,
            curb_loc                TEXT,
            status                  TEXT,
            health                  TEXT,
            spc_latin               TEXT,
            spc_common              TEXT,
            steward                 TEXT,
            guards                  TEXT,
            sidewalk                TEXT,
            user_type               TEXT,
            problems                TEXT,
            root_stone              TEXT,
            root_grate              TEXT,
            root_other              TEXT,
            trunk_wire              TEXT,
            trnk_light              TEXT,
            trnk_other              TEXT,
            brch_light              TEXT,
            brch_shoe               TEXT,
            brch_other              TEXT,
            address                 TEXT,
            postcode                TEXT,
            zip_city                TEXT,
            community_board         INTEGER,
            borocode                INTEGER,
            borough                 TEXT,
            cncldist                INTEGER,
            st_assem                INTEGER,
            st_senate               INTEGER,
            nta                     TEXT,
            nta_name                TEXT,
            boro_ct                 TEXT,
            state                   TEXT,
            latitude                DOUBLE PRECISION,
            longitude               DOUBLE PRECISION,
            x_sp                    DOUBLE PRECISION,
            y_sp                    DOUBLE PRECISION,
            council_district        INTEGER,
            census_tract            TEXT,
            bin                     INTEGER,
            bbl                     BIGINT
        );
    "
}

index_trees() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_trees_health    ON nyc.tree_census_2015 (health);
        CREATE INDEX IF NOT EXISTS idx_trees_species   ON nyc.tree_census_2015 (spc_common);
        CREATE INDEX IF NOT EXISTS idx_trees_borough   ON nyc.tree_census_2015 (borough);
        CREATE INDEX IF NOT EXISTS idx_trees_postcode  ON nyc.tree_census_2015 (postcode);
        CREATE INDEX IF NOT EXISTS idx_trees_sidewalk  ON nyc.tree_census_2015 (sidewalk);
        CREATE INDEX IF NOT EXISTS idx_trees_latlon    ON nyc.tree_census_2015 (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_trees_species_trgm ON nyc.tree_census_2015
            USING gin (spc_common gin_trgm_ops);
    "
}

# ── 8. MTA Subway Entrances ─────────────────────────────────────────────────
create_table_subway() {
  run_sql "
        DROP TABLE IF EXISTS nyc.mta_subway_entrances CASCADE;
        CREATE TABLE nyc.mta_subway_entrances (
            division                TEXT,
            line                    TEXT,
            station_name            TEXT,
            station_latitude        DOUBLE PRECISION,
            station_longitude       DOUBLE PRECISION,
            route_1                 TEXT,
            route_2                 TEXT,
            route_3                 TEXT,
            route_4                 TEXT,
            route_5                 TEXT,
            route_6                 TEXT,
            route_7                 TEXT,
            route_8                 TEXT,
            route_9                 TEXT,
            route_10                TEXT,
            route_11                TEXT,
            entrance_type           TEXT,
            entry                   TEXT,
            exit_only               TEXT,
            vending                 TEXT,
            staffing                TEXT,
            staff_hours             TEXT,
            ada                     TEXT,
            ada_notes               TEXT,
            free_crossover          TEXT,
            north_south_street      TEXT,
            east_west_street        TEXT,
            corner                  TEXT,
            entrance_latitude       DOUBLE PRECISION,
            entrance_longitude      DOUBLE PRECISION,
            station_location        TEXT,
            entrance_location       TEXT
        );
    "
}

index_subway() {
  run_sql "
        CREATE INDEX IF NOT EXISTS idx_subway_station   ON nyc.mta_subway_entrances (station_name);
        CREATE INDEX IF NOT EXISTS idx_subway_ada       ON nyc.mta_subway_entrances (ada);
        CREATE INDEX IF NOT EXISTS idx_subway_line      ON nyc.mta_subway_entrances (line);
        CREATE INDEX IF NOT EXISTS idx_subway_latlon    ON nyc.mta_subway_entrances (entrance_latitude, entrance_longitude)
            WHERE entrance_latitude IS NOT NULL AND entrance_longitude IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_subway_station_trgm ON nyc.mta_subway_entrances
            USING gin (station_name gin_trgm_ops);
    "
}

# =============================================================================
# POST-LOAD TRANSFORMS
# =============================================================================

post_transform() {
  log "Running post-load transforms..."

  # Convert text dates to proper timestamps where stored as text
  run_sql "
        -- Add geometry columns using PostGIS for spatial queries
        -- 311
        ALTER TABLE nyc.service_requests_311
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.service_requests_311
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_311_geom
            ON nyc.service_requests_311 USING gist(geom);

        -- Collisions
        ALTER TABLE nyc.motor_vehicle_collisions
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.motor_vehicle_collisions
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_collisions_geom
            ON nyc.motor_vehicle_collisions USING gist(geom);

        -- Trees
        ALTER TABLE nyc.tree_census_2015
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.tree_census_2015
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_trees_geom
            ON nyc.tree_census_2015 USING gist(geom);

        -- Street lights
        ALTER TABLE nyc.street_light_conditions
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.street_light_conditions
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_streetlight_geom
            ON nyc.street_light_conditions USING gist(geom);

        -- PLUTO
        ALTER TABLE nyc.pluto
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.pluto
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_pluto_geom
            ON nyc.pluto USING gist(geom);

        -- Subway entrances
        ALTER TABLE nyc.mta_subway_entrances
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.mta_subway_entrances
            SET geom = ST_SetSRID(ST_MakePoint(entrance_longitude, entrance_latitude), 4326)
            WHERE entrance_latitude IS NOT NULL AND entrance_longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_subway_geom
            ON nyc.mta_subway_entrances USING gist(geom);

        -- DOB filings
        ALTER TABLE nyc.dob_job_filings
            ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
        UPDATE nyc.dob_job_filings
            SET geom = ST_SetSRID(ST_MakePoint(gis_longitude, gis_latitude), 4326)
            WHERE gis_latitude IS NOT NULL AND gis_longitude IS NOT NULL AND geom IS NULL;
        CREATE INDEX IF NOT EXISTS idx_dob_geom
            ON nyc.dob_job_filings USING gist(geom);
    "

  # Crash date parsing (text → date)
  run_sql "
        ALTER TABLE nyc.motor_vehicle_collisions
            ADD COLUMN IF NOT EXISTS crash_timestamp TIMESTAMP;
        UPDATE nyc.motor_vehicle_collisions
            SET crash_timestamp = TO_TIMESTAMP(crash_date || ' ' || crash_time, 'MM/DD/YYYY HH24:MI')
            WHERE crash_date IS NOT NULL AND crash_time IS NOT NULL AND crash_timestamp IS NULL;
        CREATE INDEX IF NOT EXISTS idx_collisions_timestamp
            ON nyc.motor_vehicle_collisions (crash_timestamp);
    "

  # Analyze all tables for query planner
  run_sql "
        ANALYZE nyc.service_requests_311;
        ANALYZE nyc.motor_vehicle_collisions;
        ANALYZE nyc.sidewalk_inspections;
        ANALYZE nyc.street_light_conditions;
        ANALYZE nyc.pluto;
        ANALYZE nyc.dob_job_filings;
        ANALYZE nyc.tree_census_2015;
        ANALYZE nyc.mta_subway_entrances;
    "

  log "Post-load transforms complete."
}

# =============================================================================
# DATASET REGISTRY
# =============================================================================

# name|soda_code|download_method|create_fn|index_fn
DATASETS=(
  "311|erm2-nwe9|paged|create_table_311|index_311"
  "collisions|h9gi-nx95|full|create_table_collisions|index_collisions"
  "sidewalks|p4u2-3jgx|full|create_table_sidewalks|index_sidewalks"
  "streetlights|rwz8-g5zv|full|create_table_streetlights|index_streetlights"
  "pluto|64uk-42ks|full|create_table_pluto|index_pluto"
  "dob|w9ak-ipjd|full|create_table_dob|index_dob"
  "trees|uvpi-gqnh|full|create_table_trees|index_trees"
  "subway|drua-migt|full|create_table_subway|index_subway"
)

# =============================================================================
# MAIN
# =============================================================================

main() {
  log "========================================"
  log "NYC Open Data → Postgres Ingest Pipeline"
  log "Target: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
  log "========================================"

  # Test connection
  log "Testing database connection..."
  if ! run_sql "SELECT 1;" >/dev/null 2>&1; then
    die "Cannot connect to PostgreSQL at ${DB_HOST}:${DB_PORT}/${DB_NAME}"
  fi
  log "Connection OK."

  # Create schema & extensions
  create_schemas

  # Determine which datasets to process
  local selected=()
  if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
      for entry in "${DATASETS[@]}"; do
        local name="${entry%%|*}"
        if [[ "$name" == "$arg" ]]; then
          selected+=("$entry")
        fi
      done
    done
    if [[ ${#selected[@]} -eq 0 ]]; then
      die "No matching datasets found. Available: 311, collisions, sidewalks, streetlights, pluto, dob, trees, subway"
    fi
  else
    selected=("${DATASETS[@]}")
  fi

  # Process each dataset
  for entry in "${selected[@]}"; do
    IFS='|' read -r name code method create_fn index_fn <<<"$entry"

    log ""
    log "━━━ Processing: ${name} ━━━"

    # 1. Download
    if [[ "$method" == "paged" ]]; then
      download_csv_paged "$name" "$code"
    else
      download_csv "$name" "$code"
    fi

    # 2. Sanitize
    sanitize_csv "${DATA_DIR}/${name}.csv"

    # 3. Create table
    log "  ↳ Creating table..."
    $create_fn

    # 4. Bulk load
    bulk_load "nyc.$(
      case $name in
      311) echo "service_requests_311" ;;
      collisions) echo "motor_vehicle_collisions" ;;
      sidewalks) echo "sidewalk_inspections" ;;
      streetlights) echo "street_light_conditions" ;;
      pluto) echo "pluto" ;;
      dob) echo "dob_job_filings" ;;
      trees) echo "tree_census_2015" ;;
      subway) echo "mta_subway_entrances" ;;
      esac
    )" "${DATA_DIR}/${name}.csv"

    # 5. Create indexes
    log "  ↳ Creating indexes..."
    $index_fn

    log "  ✓ ${name} complete"
  done

  # 6. Post-load transforms (PostGIS geometry, date parsing, ANALYZE)
  post_transform

  # Summary
  log ""
  log "========================================"
  log "INGEST COMPLETE — Summary"
  log "========================================"
  run_sql "
        SELECT
            schemaname || '.' || tablename AS table_name,
            pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
            n_live_tup AS row_count
        FROM pg_stat_user_tables
        WHERE schemaname = 'nyc'
        ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
    "

  log "Log file: ${LOG_FILE}"
  log "Done!"
}

main "$@"
