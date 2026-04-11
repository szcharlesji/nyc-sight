#!/usr/bin/env bash
# =============================================================================
# start.sh — Sequential NIM Stack Orchestrator for DGX Spark
#
# Launches four NVIDIA NIM containers one at a time, waiting for each to become
# healthy before starting the next. This prevents vLLM's GPU memory profiling
# race condition (AssertionError: free memory increased during profiling).
#
# Launch order (lightest GPU consumers first):
#   1. Parakeet ASR         (ASR)  ~2-4GB
#   2. Magpie TTS           (TTS)  ~11GB
#   3. Cosmos-Reason2-8B    (VLM)  ~44GB
#   4. Nemotron-Nano-12B    (LLM)  ~17-22GB
#
# Usage:
#   export NGC_API_KEY=<your_key>
#   ./start.sh
#   ./start.sh --skip-smoke   # skip smoke tests
# =============================================================================

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
readonly ENV_FILE="${SCRIPT_DIR}/.env"

# Terminal colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'  # No Color

# Parse CLI flags
SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --skip-smoke) SKIP_SMOKE=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-smoke] [--help]"
            echo "  --skip-smoke  Skip smoke tests after each service starts"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--skip-smoke] [--help]"
            exit 1
            ;;
    esac
done

# ─── Source .env ──────────────────────────────────────────────────────────────
# Docker Compose .env files don't expand ${HOME} or ~.
# We source it here to get port values for health URLs, then resolve paths.

if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}[ERROR] .env file not found at ${ENV_FILE}${NC}"
    echo "        Copy .env.example or create .env with NGC_API_KEY set."
    exit 1
fi

set -a
# shellcheck source=.env
source "$ENV_FILE"
set +a

# Resolve ~ and $HOME in paths (Docker Compose can't do this)
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE/#\~/$HOME}"
NEMOTRON_MODEL_PATH="${NEMOTRON_MODEL_PATH/#\~/$HOME}"
export LOCAL_NIM_CACHE NEMOTRON_MODEL_PATH

# Export host UID for Cosmos container (avoids readonly UID builtin)
export HOST_UID="$(id -u)"

# Resolved port values (with defaults matching .env)
readonly NEMOTRON_PORT="${NEMOTRON_PORT:-8005}"
readonly COSMOS_PORT="${COSMOS_PORT:-8000}"
readonly MAGPIE_HTTP_PORT="${MAGPIE_HTTP_PORT:-9000}"
readonly PARAKEET_HTTP_PORT="${PARAKEET_HTTP_PORT:-9001}"
readonly MAGPIE_GRPC_PORT="${MAGPIE_GRPC_PORT:-50051}"
readonly PARAKEET_GRPC_PORT="${PARAKEET_GRPC_PORT:-50052}"

# Service definitions — parallel arrays for the sequential launch loop
readonly SERVICES=(parakeet-asr magpie-tts cosmos nemotron)
readonly CONTAINERS=(parakeet-asr magpie-tts cosmos-reason2-8b nemotron-nano)
readonly HEALTH_URLS=(
    "http://0.0.0.0:${PARAKEET_HTTP_PORT}/v1/health/ready"
    "http://0.0.0.0:${MAGPIE_HTTP_PORT}/v1/health/ready"
    "http://0.0.0.0:${COSMOS_PORT}/v1/models"
    "http://0.0.0.0:${NEMOTRON_PORT}/v1/models"
)
# Riva services: 1800s (up to 30 min for first-run TRT engine compilation)
# vLLM services: 600s (5-8 min for bf16 torch.compile)
readonly TIMEOUTS=(1800 1800 600 600)
readonly SMOKE_TYPES=(none tts vlm llm)

# ─── Helper Functions ─────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo -e "${RED}==> Interrupted — tearing down partial stack...${NC}"
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    echo -e "${RED}    Stack stopped.${NC}"
    exit 1
}
trap cleanup SIGINT SIGTERM

print_banner() {
    local service="$1"
    local container="$2"
    local index="$3"
    local total="${#SERVICES[@]}"
    echo ""
    echo -e "${BOLD}${CYAN}==> [${index}/${total}] Starting ${service}${NC}"
    echo -e "    Container: ${container}"
}

check_port() {
    local port="$1"
    local name="$2"
    if lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "${RED}[ERROR] Port ${port} is already in use (needed by ${name}).${NC}"
        echo "        Run: lsof -i :${port} to see what's using it."
        return 1
    fi
    return 0
}

wait_for_health() {
    local container="$1"
    local url="$2"
    local max_wait="${3:-600}"
    local interval=10
    local elapsed=0

    echo -e "    ${YELLOW}Waiting for ${container} to become ready (timeout: ${max_wait}s)...${NC}"
    echo -n "    "

    while true; do
        # Check if the health endpoint responds
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}ready!${NC}"
            return 0
        fi

        # Check the container is still alive (not crashed/exited)
        if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo ""
            echo -e "    ${RED}[ERROR] Container ${container} exited unexpectedly.${NC}"
            echo "    Last 30 lines of logs:"
            docker compose -f "$COMPOSE_FILE" logs --tail=30 "${container}" 2>/dev/null || \
                docker logs --tail=30 "${container}" 2>/dev/null || true
            return 1
        fi

        # Check timeout
        if [[ $elapsed -ge $max_wait ]]; then
            echo ""
            echo -e "    ${RED}[ERROR] ${container} did not become ready within ${max_wait}s.${NC}"
            echo "    Last 30 lines of logs:"
            docker compose -f "$COMPOSE_FILE" logs --tail=30 "${container}" 2>/dev/null || \
                docker logs --tail=30 "${container}" 2>/dev/null || true
            return 1
        fi

        echo -n "."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
}

smoke_test_llm() {
    local port="$1"
    local model="$2"
    echo -n "    Smoke test (chat completion)... "
    local response
    response=$(curl -sf -X POST "http://0.0.0.0:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
            \"max_tokens\": 10
        }" 2>&1) || {
        echo -e "${RED}FAILED${NC}"
        echo "    Response: ${response}"
        return 1
    }
    local content
    content=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>&1) || {
        echo -e "${RED}FAILED (JSON parse)${NC}"
        echo "    Raw response: ${response}"
        return 1
    }
    echo -e "${GREEN}OK${NC} — \"${content}\""
    return 0
}

smoke_test_tts() {
    local port="$1"
    echo -n "    Smoke test (audio synthesis)... "
    local http_code
    http_code=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "http://0.0.0.0:${port}/v1/audio/synthesize" \
        -F "language=en-US" \
        -F "text=Hello world" \
        -F "voice=Magpie-Multilingual.EN-US.Aria" 2>&1) || {
        echo -e "${RED}FAILED${NC}"
        return 1
    }
    if [[ "$http_code" == "200" ]]; then
        echo -e "${GREEN}OK${NC} — HTTP 200"
    else
        echo -e "${RED}FAILED${NC} — HTTP ${http_code}"
        return 1
    fi
    return 0
}

gpu_summary() {
    echo ""
    echo -e "${BOLD}==> GPU Memory Usage${NC}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
            --format=csv,noheader,nounits \
            | awk '{printf "    Used: %s MB  |  Free: %s MB  |  Total: %s MB\n", $1, $2, $3}'
    else
        echo "    nvidia-smi not available on this host."
    fi
}

# ─── Preflight Checks ────────────────────────────────────────────────────────

echo -e "${BOLD}==> NIM Stack — Preflight Checks${NC}"

# NGC API Key
if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo -e "${RED}[ERROR] NGC_API_KEY is not set.${NC}"
    echo "        Export it: export NGC_API_KEY=<your_key>"
    exit 1
fi
echo -e "    ${GREEN}✓${NC} NGC_API_KEY is set"

# Nemotron model path
readonly NEMOTRON_MODEL_SUBDIR="${NEMOTRON_MODEL_SUBDIR:-nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
readonly FULL_MODEL_PATH="${NEMOTRON_MODEL_PATH}/${NEMOTRON_MODEL_SUBDIR}"
if [[ ! -d "$FULL_MODEL_PATH" ]]; then
    echo -e "${RED}[ERROR] Nemotron model not found: ${FULL_MODEL_PATH}${NC}"
    echo "        Set NEMOTRON_MODEL_PATH and NEMOTRON_MODEL_SUBDIR in .env"
    exit 1
fi
echo -e "    ${GREEN}✓${NC} Nemotron model found: ${FULL_MODEL_PATH}"

# Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Docker daemon is not running.${NC}"
    exit 1
fi
echo -e "    ${GREEN}✓${NC} Docker daemon is running"

# nvidia-smi (warning only — DGX Spark should always have this)
if command -v nvidia-smi &>/dev/null; then
    echo -e "    ${GREEN}✓${NC} nvidia-smi available"
else
    echo -e "    ${YELLOW}⚠${NC} nvidia-smi not found — GPU checks will be skipped"
fi

# NIM cache directory
mkdir -p "$LOCAL_NIM_CACHE"
# chmod 777 so both root (Nemotron) and host-UID (Cosmos) can write
chmod 777 "$LOCAL_NIM_CACHE"
echo -e "    ${GREEN}✓${NC} NIM cache: ${LOCAL_NIM_CACHE}"

# Port conflict check
readonly ALL_PORTS=(
    "$NEMOTRON_PORT:Nemotron"
    "$COSMOS_PORT:Cosmos"
    "$MAGPIE_HTTP_PORT:Magpie-HTTP"
    "$MAGPIE_GRPC_PORT:Magpie-gRPC"
    "$PARAKEET_HTTP_PORT:Parakeet-HTTP"
    "$PARAKEET_GRPC_PORT:Parakeet-gRPC"
)
port_conflict=false
for entry in "${ALL_PORTS[@]}"; do
    local_port="${entry%%:*}"
    local_name="${entry##*:}"
    if ! check_port "$local_port" "$local_name"; then
        port_conflict=true
    fi
done
if [[ "$port_conflict" == true ]]; then
    echo -e "${RED}[ERROR] Port conflicts detected. Resolve them before starting.${NC}"
    exit 1
fi
echo -e "    ${GREEN}✓${NC} All ports available (${NEMOTRON_PORT}, ${COSMOS_PORT}, ${MAGPIE_HTTP_PORT}, ${MAGPIE_GRPC_PORT}, ${PARAKEET_HTTP_PORT}, ${PARAKEET_GRPC_PORT})"

# ─── Cleanup Stale Stack ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}==> Cleaning up any existing stack...${NC}"
docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# ─── Sequential Launch ────────────────────────────────────────────────────────

total=${#SERVICES[@]}
for i in $(seq 0 $((total - 1))); do
    service="${SERVICES[$i]}"
    container="${CONTAINERS[$i]}"
    health_url="${HEALTH_URLS[$i]}"
    timeout="${TIMEOUTS[$i]}"
    smoke_type="${SMOKE_TYPES[$i]}"

    print_banner "$service" "$container" "$((i + 1))"

    # Launch the single service
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "$service"

    # Wait for health
    if ! wait_for_health "$container" "$health_url" "$timeout"; then
        echo ""
        echo -e "${RED}[FATAL] ${service} failed to start. Tearing down stack.${NC}"
        docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
        exit 1
    fi

    # Smoke test
    if [[ "$SKIP_SMOKE" == false ]]; then
        case "$smoke_type" in
            llm)
                smoke_test_llm "$NEMOTRON_PORT" "${NEMOTRON_MODEL_NAME:-nemotron-nano}" || true
                ;;
            vlm)
                smoke_test_llm "$COSMOS_PORT" "nvidia/cosmos-reason2-8b" || true
                ;;
            tts)
                smoke_test_tts "$MAGPIE_HTTP_PORT" || true
                ;;
            none)
                echo "    Smoke test: skipped (ASR requires audio input; health check sufficient)"
                ;;
        esac
    fi

    echo -e "    ${GREEN}✓ ${service} is ready${NC}"
done

# ─── GPU Summary ──────────────────────────────────────────────────────────────

gpu_summary

# ─── Final Summary ────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}==> All four NIMs are running!${NC}"
echo ""
echo "    ┌─────────────────────────────────────────────────────────────────┐"
echo "    │  Service            │  Endpoint                                │"
echo "    ├─────────────────────┼──────────────────────────────────────────┤"
echo "    │  Nemotron (LLM)     │  http://0.0.0.0:${NEMOTRON_PORT}/v1/chat/completions  │"
echo "    │  Cosmos (VLM)       │  http://0.0.0.0:${COSMOS_PORT}/v1/chat/completions  │"
echo "    │  Magpie TTS (HTTP)  │  http://0.0.0.0:${MAGPIE_HTTP_PORT}/v1/audio/synthesize  │"
echo "    │  Magpie TTS (gRPC)  │  0.0.0.0:${MAGPIE_GRPC_PORT}                         │"
echo "    │  Parakeet ASR (HTTP)│  http://0.0.0.0:${PARAKEET_HTTP_PORT}/v1/health/ready     │"
echo "    │  Parakeet ASR (gRPC)│  0.0.0.0:${PARAKEET_GRPC_PORT}                         │"
echo "    └─────────────────────┴──────────────────────────────────────────┘"
echo ""
echo "    To stop the stack:       ./stop.sh"
echo "    To check health:         ./healthcheck.sh"
echo "    To monitor GPU memory:   watch -n 5 nvidia-smi"
echo "    To view logs:            docker compose -f ${COMPOSE_FILE} logs -f [service]"
echo ""
