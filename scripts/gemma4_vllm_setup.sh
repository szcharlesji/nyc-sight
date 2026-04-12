# =============================================================================
# Gemma-4-26B-A4B-IT (LLM, port 8006)
#
# Model: ~/Downloads/models/vlm/gemma-4-26B-A4B-it-AWQ-4bit (local AWQ 4bit)
# Architecture: Gemma 4, MoE, AWQ 4bit quantized
#
# Memory estimate on GB10 (128GB unified):
#   Weights (AWQ 4bit, MoE sparse):  ~13GB on disk, ~6-8GB active params in GPU
#   Framework overhead:          ~3-4GB
#   KV cache @ 0.25 * 128GB:    ~32GB  → 32768 token context
#   Estimated peak:              ~40-45GB
#
# If co-running with Nemotron + Cosmos (~75-85GB), this will exceed 128GB budget.
# Run standalone, or reduce GEMMA_KVCACHE_PERCENT to 0.15 when co-running.
#
# Key flags:
#   VLLM_WORKER_MULTIPROC_METHOD=spawn  Required for HF-sourced MoE models
#   --trust-remote-code                 Required for Gemma 4 tokenizer/config
#   --dtype bfloat16                    Force bf16 to avoid FP8 issues on SM121
# =============================================================================
set -euo pipefail

# Shared cache directory for downloaded model artifacts
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"

# NIM for model free
NIM_LLM_MODEL_FREE_IMAGE=nvcr.io/nim/nvidia/model-free-nim:2.0.2

# --- Gemma settings -----------------------------------------------------------
GEMMA_CONTAINER="gemma4-26b"
GEMMA_PORT=8006

# Local model path — mounted into container at /mnt/models
GEMMA_LOCAL_MODEL_DIR="$HOME/Downloads/models/vlm"
GEMMA_MODEL_NAME="gemma-4-26B-A4B-it-AWQ-4bit"
GEMMA_MODEL_PATH="/mnt/models/$GEMMA_MODEL_NAME"

# KV cache fraction.
# Standalone: 0.25 gives ~32GB for KV cache (comfortable for 32k context)
# Co-running: drop to 0.15 to stay within 128GB shared budget
GEMMA_KVCACHE_PERCENT=0.25

# Max context length.
# Gemma 4 supports up to 128k, but 32768 is a safe ceiling when sharing GPU.
# Increase to 65536 if running standalone with GEMMA_KVCACHE_PERCENT=0.35.
GEMMA_MAX_MODEL_LEN=32768

# Served model name alias (used in API calls)
GEMMA_SERVED_MODEL_NAME="gemma4-26b"

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

# Verify local model directory exists
if [[ ! -d "$GEMMA_LOCAL_MODEL_DIR/$GEMMA_MODEL_NAME" ]]; then
    echo "[ERROR] Local model not found: $GEMMA_LOCAL_MODEL_DIR/$GEMMA_MODEL_NAME"
    exit 1
fi

# --- Stop existing container if present ---------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${GEMMA_CONTAINER}$"; then
    echo "==> Removing existing container: $GEMMA_CONTAINER"
    docker rm -f "$GEMMA_CONTAINER"
fi



# --- Launch -------------------------------------------------------------------
echo ""
echo "==> [Gemma] Starting gemma-4-26B-A4B-it-AWQ-4bit on port $GEMMA_PORT..."
echo "    Using local model: $GEMMA_LOCAL_MODEL_DIR/$GEMMA_MODEL_NAME"

docker run -d \
    --gpus all \
    --shm-size=16GB \
    --name "$GEMMA_CONTAINER" \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -v "$GEMMA_LOCAL_MODEL_DIR:/mnt/models" \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_MODEL_PATH="$GEMMA_MODEL_PATH" \
    -e NIM_SERVED_MODEL_NAME="$GEMMA_SERVED_MODEL_NAME" \
    -e NIM_KVCACHE_PERCENT="$GEMMA_KVCACHE_PERCENT" \
    -e NIM_MAX_MODEL_LEN="$GEMMA_MAX_MODEL_LEN" \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -p "${GEMMA_PORT}:8000" \
    "$NIM_LLM_MODEL_FREE_IMAGE" \
    "$GEMMA_MODEL_PATH" \
    --trust-remote-code \
    --dtype bfloat16

echo "    Container: $GEMMA_CONTAINER"
echo "    Endpoint:  http://0.0.0.0:${GEMMA_PORT}/v1/chat/completions"
echo "    Model:     $GEMMA_SERVED_MODEL_NAME"

# wait_for_health "$GEMMA_CONTAINER" "$GEMMA_PORT"

# # --- Smoke test ---------------------------------------------------------------
 echo -n "    [Gemma]    "
 curl -sf -X POST "http://0.0.0.0:${GEMMA_PORT}/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d "{
         \"model\": \"${GEMMA_SERVED_MODEL_NAME}\",
         \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
         \"max_tokens\": 10
     }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" \
     || echo "FAILED — check: docker logs $GEMMA_CONTAINER"
