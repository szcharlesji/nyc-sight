#!/usr/bin/env bash
# =============================================================================
# healthcheck.sh — Point-in-time NIM Stack Health Probe
#
# Checks all four NIM services once (no polling/retry).
# Designed for cron, monitoring scripts, or quick manual checks.
#
# Exit codes:
#   0 — all services healthy
#   1 — at least one service unhealthy or unreachable
#
# Usage:
#   ./healthcheck.sh           # color output to terminal
#   ./healthcheck.sh --json    # machine-readable JSON output
# =============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_FILE="${SCRIPT_DIR}/.env"
readonly CURL_TIMEOUT=5

# Terminal colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Parse flags
JSON_OUTPUT=false
for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUTPUT=true ;;
        --help|-h)
            echo "Usage: $0 [--json] [--help]"
            exit 0
            ;;
    esac
done

# Source .env for port values
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=.env
    source "$ENV_FILE"
    set +a
fi

# Defaults (match .env)
NEMOTRON_PORT="${NEMOTRON_PORT:-8005}"
COSMOS_PORT="${COSMOS_PORT:-8000}"
MAGPIE_HTTP_PORT="${MAGPIE_HTTP_PORT:-9000}"
PARAKEET_HTTP_PORT="${PARAKEET_HTTP_PORT:-9001}"

# Service definitions
declare -A SERVICES=(
    [nemotron]="http://0.0.0.0:${NEMOTRON_PORT}/v1/models"
    [cosmos]="http://0.0.0.0:${COSMOS_PORT}/v1/models"
    [magpie-tts]="http://0.0.0.0:${MAGPIE_HTTP_PORT}/v1/health/ready"
    [parakeet-asr]="http://0.0.0.0:${PARAKEET_HTTP_PORT}/v1/health/ready"
)

# Ordered list (for consistent output)
readonly SERVICE_ORDER=(nemotron cosmos magpie-tts parakeet-asr)
readonly SERVICE_LABELS=(
    "Nemotron-3-Nano-30B (LLM)"
    "Cosmos-Reason2-8B   (VLM)"
    "Magpie TTS          (TTS)"
    "Parakeet ASR        (ASR)"
)

all_healthy=true
json_results=()

check_service() {
    local name="$1"
    local label="$2"
    local url="${SERVICES[$name]}"
    local status="unhealthy"
    local detail=""

    if response=$(curl -sf --max-time "$CURL_TIMEOUT" "$url" 2>&1); then
        status="healthy"
        # For vLLM services, extract model name from response
        if [[ "$url" == *"/v1/models"* ]]; then
            detail=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    models = [m['id'] for m in d.get('data', [])]
    print(', '.join(models))
except: print('(parse error)')
" 2>/dev/null || echo "")
        else
            # Riva health endpoint returns {"ready": true/false}
            ready=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('true' if d.get('ready') else 'false')
except: print('unknown')
" 2>/dev/null || echo "unknown")
            if [[ "$ready" != "true" ]]; then
                status="unhealthy"
                detail="ready=$ready"
            fi
        fi
    else
        detail="connection refused or timeout"
    fi

    if [[ "$JSON_OUTPUT" == true ]]; then
        json_results+=("{\"service\":\"${name}\",\"status\":\"${status}\",\"url\":\"${url}\",\"detail\":\"${detail}\"}")
    else
        if [[ "$status" == "healthy" ]]; then
            echo -e "    ${GREEN}●${NC} ${label}  ${GREEN}healthy${NC}  ${detail:+— $detail}"
        else
            echo -e "    ${RED}●${NC} ${label}  ${RED}unhealthy${NC}  ${detail:+— $detail}"
            all_healthy=false
        fi
    fi

    [[ "$status" == "healthy" ]]
}

# ─── Run Checks ──────────────────────────────────────────────────────────────

if [[ "$JSON_OUTPUT" == false ]]; then
    echo -e "${BOLD}==> NIM Stack Health Check${NC}"
    echo ""
fi

for i in "${!SERVICE_ORDER[@]}"; do
    check_service "${SERVICE_ORDER[$i]}" "${SERVICE_LABELS[$i]}" || all_healthy=false
done

# ─── Output ───────────────────────────────────────────────────────────────────

if [[ "$JSON_OUTPUT" == true ]]; then
    # Join JSON array
    printf '{"services":[%s],"all_healthy":%s}\n' \
        "$(IFS=,; echo "${json_results[*]}")" \
        "$all_healthy"
else
    echo ""
    if [[ "$all_healthy" == true ]]; then
        echo -e "    ${GREEN}All services healthy.${NC}"
    else
        echo -e "    ${RED}One or more services unhealthy.${NC}"
    fi

    # GPU summary if available
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo -e "${BOLD}==> GPU Memory${NC}"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
            --format=csv,noheader,nounits \
            | awk '{printf "    Used: %s MB  |  Free: %s MB  |  Total: %s MB\n", $1, $2, $3}'
    fi
fi

if [[ "$all_healthy" == true ]]; then
    exit 0
else
    exit 1
fi
