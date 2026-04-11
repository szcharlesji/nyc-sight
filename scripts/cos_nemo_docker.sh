#!/usr/bin/env bash
# =============================================================================
# start_nims.sh
# Launches Nemotron-3-Nano-30B (LLM) then Cosmos-Reason2-8B (VLM) NIM containers
# on a DGX Spark (GB10, SM121) with shared GPU memory budget under 90GB.
#
# Launch order: Nemotron first, Cosmos second.
# Each container waits for the previous to be fully ready before starting,
# avoiding the vLLM memory profiling race condition (AssertionError: free memory
# increased during profiling).
#
# Tested parameters (debugged on GB10, 128GB unified memory):
#   Nemotron: NIM_KVCACHE_PERCENT=0.35, NIM_MAX_MODEL_LEN=32768
#   Cosmos:   NIM_KVCACHE_PERCENT=0.35, NIM_MAX_MODEL_LEN=16384
#   Estimated total GPU usage: ~75-85GB
#
# Usage:
#   export NGC_API_KEY=<your_key>
#   export NIM_LLM_MODEL_FREE_IMAGE=<your_model_free_image>
#   bash start_nims.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Shared cache directory for downloaded model artifacts
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"

# Nemotron model path (local weights on disk)
NEMOTRON_MODEL_PATH="${NEMOTRON_MODEL_PATH:-$HOME/Downloads/models/llm/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"

# NIM for Nemotron
NIM_LLM_MODEL_FREE_IMAGE=nvcr.io/nim/nvidia/model-free-nim:2.0.2

# --- Nemotron settings -------------------------------------------------------
NEMOTRON_CONTAINER="nemotron-nano"
NEMOTRON_PORT=8005

# KV cache fraction for Nemotron.
# 128GB * 0.35 = 44.8GB total budget; after ~27GB weights+framework = ~17GB KV cache.
# Sufficient for multi-turn chat with max_model_len=32768.
NEMOTRON_KVCACHE_PERCENT=0.35

# Max context length for Nemotron.
# Multi-turn chat use case; 32768 tokens covers long conversations comfortably.
NEMOTRON_MAX_MODEL_LEN=32768

# --- Cosmos settings ---------------------------------------------------------
COSMOS_IMAGE="nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0"
COSMOS_CONTAINER="cosmos-reason2-8b"
COSMOS_PORT=8000

# bf16 profile ID for Cosmos on GB10 — avoids FP8 kernel bugs on SM121.
# Source: profile list printed in NIM startup log (vllm-bf16-tp1).
# To refresh: docker run --rm --runtime=nvidia --gpus all -e NGC_API_KEY \
#             nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0 list-model-profiles
COSMOS_BF16_PROFILE="3266ed3ec2297386d2e4a94e6c84a5b0ba92244f787c538f33577c4c78c5aef2"

# KV cache fraction for Cosmos.
# 128GB * 0.35 = 44.8GB total budget; after ~36GB weights+framework = ~8GB KV cache.
# Sufficient for short video use case: <=10s @ 4fps => ~40 frames => ~5k tokens.
# NIM_KVCACHE_PERCENT must be > 36/128 = 0.28 or vLLM reports negative KV cache.
COSMOS_KVCACHE_PERCENT=0.35

# Max context length for Cosmos.
# 10s @ 4fps ~= 40 frames ~= 5k multimodal tokens. 16384 is a safe ceiling.
# Default is 262144 which requires 36GB KV cache alone — not viable when sharing GPU.
COSMOS_MAX_MODEL_LEN=16384

# -----------------------------------------------------------------------------
# Helper: wait for a NIM container's HTTP endpoint to become available.
#
# Why sequential launch matters:
#   vLLM snapshots free GPU memory at the start of profiling. If another
#   container is initializing concurrently (allocating then freeing scratch
#   memory), free memory can appear to INCREASE mid-profile, triggering:
#     AssertionError: Initial free memory X GiB, current free memory Y GiB
#   Launching containers one at a time (and waiting for each to be fully
#   ready) eliminates this race condition entirely.
# -----------------------------------------------------------------------------

wait_for_health() {
    local NAME=$1
    local PORT=$2
    local MAX_WAIT=600  # seconds — bf16 torch.compile can take 5-8 min
    local ELAPSED=0
    local INTERVAL=10

    echo ""
    echo "==> Waiting for $NAME to become ready (this may take 5-8 minutes)..."
    echo -n "    "
    while true; do
        if curl -sf "http://0.0.0.0:${PORT}/v1/models" > /dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
            echo ""
            echo "[ERROR] Container $NAME exited unexpectedly."
            echo "        docker logs $NAME"
            exit 1
        fi
        if [[ $ELAPSED -ge $MAX_WAIT ]]; then
            echo ""
            echo "[ERROR] $NAME did not become ready within ${MAX_WAIT}s."
            echo "        docker logs $NAME"
            exit 1
        fi
        echo -n "."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done
}

# -----------------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------------

echo "==> Checking required environment variables..."

if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "[ERROR] NGC_API_KEY is not set. Export it before running this script."
    exit 1
fi

if [[ -z "${NIM_LLM_MODEL_FREE_IMAGE:-}" ]]; then
    echo "[ERROR] NIM_LLM_MODEL_FREE_IMAGE is not set. Export it before running this script."
    exit 1
fi

if [[ ! -d "$NEMOTRON_MODEL_PATH" ]]; then
    echo "[ERROR] Nemotron model path does not exist: $NEMOTRON_MODEL_PATH"
    echo "        Override with: export NEMOTRON_MODEL_PATH=<path>"
    exit 1
fi

echo "==> Creating NIM cache directory at $LOCAL_NIM_CACHE..."
mkdir -p "$LOCAL_NIM_CACHE"

# -----------------------------------------------------------------------------
# Stop and remove any existing containers with the same name
# -----------------------------------------------------------------------------

for CONTAINER in "$NEMOTRON_CONTAINER" "$COSMOS_CONTAINER"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo "==> Removing existing container: $CONTAINER"
        docker rm -f "$CONTAINER"
    fi
done

# -----------------------------------------------------------------------------
# Step 1: Launch Nemotron-3-Nano-30B-A3B-NVFP4 (LLM, port 8005)
#
# Nemotron starts first so its memory profiling runs on a clean GPU.
#
# Key settings:
#   NIM_KVCACHE_PERCENT=0.35       Sufficient for chat; leaves room for Cosmos
#   NIM_MAX_MODEL_LEN=32768        Multi-turn chat context window
#   VLLM_USE_FLASHINFER_MOE_FP4=0  Workaround: NVFP4 MoE CUDA graph crash on GB10
#                                  (flashinfer issue #2776)
# -----------------------------------------------------------------------------

echo ""
echo "==> [1/2] Starting Nemotron-3-Nano-30B on port $NEMOTRON_PORT..."

docker run -d \
    --gpus all \
    --shm-size=16GB \
    --name "$NEMOTRON_CONTAINER" \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -v "$HOME/Downloads/models/llm:/mnt/models" \
    -e NIM_MODEL_PATH="/mnt/models/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
    -e NIM_SERVED_MODEL_NAME="nemotron-nano" \
    -e NIM_KVCACHE_PERCENT="$NEMOTRON_KVCACHE_PERCENT" \
    -e NIM_MAX_MODEL_LEN="$NEMOTRON_MAX_MODEL_LEN" \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -p "${NEMOTRON_PORT}:8000" \
    "$NIM_LLM_MODEL_FREE_IMAGE" \
    /mnt/models/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code

echo "    Container: $NEMOTRON_CONTAINER"
echo "    Endpoint:  http://0.0.0.0:${NEMOTRON_PORT}/v1/chat/completions"
echo "    Model:     nemotron-nano"

wait_for_health "$NEMOTRON_CONTAINER" "$NEMOTRON_PORT"

# -----------------------------------------------------------------------------
# Step 2: Launch Cosmos-Reason2-8B (VLM, port 8000)
#
# Cosmos starts only after Nemotron is fully ready and its GPU memory is stable.
#
# Key settings:
#   --runtime=nvidia              Per official NIM VLM 1.6.0 docs
#   NIM_MODEL_PROFILE             Force bf16 — FP8 produces garbage output on GB10
#                                 (FP8 kernel numerical errors on SM121)
#   NIM_KVCACHE_PERCENT=0.35      Must be > 0.28 (36GB floor / 128GB total)
#   NIM_MAX_MODEL_LEN=16384       10s @ 4fps ~= 5k tokens; 16384 is a safe ceiling
# -----------------------------------------------------------------------------

echo ""
echo "==> [2/2] Starting Cosmos-Reason2-8B on port $COSMOS_PORT..."

docker run -d \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16GB \
    --name "$COSMOS_CONTAINER" \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_MODEL_PROFILE="$COSMOS_BF16_PROFILE" \
    -e NIM_KVCACHE_PERCENT="$COSMOS_KVCACHE_PERCENT" \
    -e NIM_MAX_MODEL_LEN="$COSMOS_MAX_MODEL_LEN" \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u "$(id -u)" \
    -p "${COSMOS_PORT}:8000" \
    "$COSMOS_IMAGE"

echo "    Container: $COSMOS_CONTAINER"
echo "    Endpoint:  http://0.0.0.0:${COSMOS_PORT}/v1/chat/completions"
echo "    Model:     nvidia/cosmos-reason2-8b"

wait_for_health "$COSMOS_CONTAINER" "$COSMOS_PORT"

# -----------------------------------------------------------------------------
# GPU memory summary
# -----------------------------------------------------------------------------

echo ""
echo "==> GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits \
    | awk '{printf "    Used: %s MB  |  Free: %s MB  |  Total: %s MB\n", $1, $2, $3}'

# -----------------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------------

echo ""
echo "==> Running smoke tests..."

echo -n "    [Nemotron] "
curl -sf -X POST "http://0.0.0.0:${NEMOTRON_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nemotron-nano",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" \
    || echo "FAILED — check: docker logs $NEMOTRON_CONTAINER"

echo -n "    [Cosmos]   "
curl -sf -X POST "http://0.0.0.0:${COSMOS_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/cosmos-reason2-8b",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" \
    || echo "FAILED — check: docker logs $COSMOS_CONTAINER"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "==> Both NIMs are running."
echo ""
echo "    Nemotron-Nano-30B  : http://0.0.0.0:${NEMOTRON_PORT}/v1/chat/completions"
echo "    Cosmos-Reason2-8B  : http://0.0.0.0:${COSMOS_PORT}/v1/chat/completions"
echo ""
echo "    To stop both containers:"
echo "        docker rm -f $NEMOTRON_CONTAINER $COSMOS_CONTAINER"
echo ""
echo "    To monitor GPU memory live:"
echo "        watch -n 5 nvidia-smi"
