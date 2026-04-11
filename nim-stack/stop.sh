#!/usr/bin/env bash
# =============================================================================
# stop.sh — Clean NIM Stack Teardown
#
# Stops all containers and removes the compose network.
# Safe to run when nothing is running (idempotent).
#
# Usage:
#   ./stop.sh              # stop all containers
#   ./stop.sh --prune      # stop + print cache cleanup instructions
# =============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
readonly ENV_FILE="${SCRIPT_DIR}/.env"

# Terminal colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

echo -e "${BOLD}==> Stopping NIM stack...${NC}"
docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
echo -e "${GREEN}    All containers stopped.${NC}"

# GPU memory after shutdown
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo -e "${BOLD}==> GPU Memory After Shutdown${NC}"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits \
        | awk '{printf "    Used: %s MB  |  Free: %s MB  |  Total: %s MB\n", $1, $2, $3}'
fi

# Optional cache pruning instructions
if [[ "${1:-}" == "--prune" ]]; then
    # Source .env for LOCAL_NIM_CACHE path
    local_cache="${LOCAL_NIM_CACHE:-~/.cache/nim}"
    if [[ -f "$ENV_FILE" ]]; then
        local_cache=$(grep -E '^LOCAL_NIM_CACHE=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 || echo "$local_cache")
    fi
    local_cache="${local_cache/#\~/$HOME}"

    echo ""
    echo -e "${YELLOW}==> Cache Pruning Instructions${NC}"
    echo "    The NIM cache may contain 50GB+ of downloaded models and compiled engines."
    echo "    Deleting it will force re-download and TRT recompilation on next start."
    echo ""
    echo "    To clear the cache:"
    echo "        rm -rf ${local_cache}"
    echo ""
    echo "    To clear Docker's build cache:"
    echo "        docker system prune --volumes"
fi

echo ""
echo "Done."
