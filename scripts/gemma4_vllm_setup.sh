set -euo pipefail

LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"

GEMMA_IMAGE="vllm/vllm-openai:gemma4"

GEMMA_CONTAINER="gemma4-26b"
GEMMA_PORT=8006
GEMMA_LOCAL_MODEL_DIR="$HOME/Downloads/models/vlm"
GEMMA_MODEL_NAME="gemma-4-26B-A4B-it-AWQ-4bit"
GEMMA_MODEL_PATH="/mnt/models/$GEMMA_MODEL_NAME"
GEMMA_KVCACHE_PERCENT=0.25
GEMMA_MAX_MODEL_LEN=32768
GEMMA_SERVED_MODEL_NAME="gemma4-26b"

# --- Preflight ----------------------------------------------------------------
if [[ ! -d "$GEMMA_LOCAL_MODEL_DIR/$GEMMA_MODEL_NAME" ]]; then
    echo "[ERROR] Local model not found: $GEMMA_LOCAL_MODEL_DIR/$GEMMA_MODEL_NAME"
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${GEMMA_CONTAINER}$"; then
    echo "==> Removing existing container: $GEMMA_CONTAINER"
    docker rm -f "$GEMMA_CONTAINER"
fi

# --- Launch -------------------------------------------------------------------
echo "==> [Gemma] Starting $GEMMA_MODEL_NAME on port $GEMMA_PORT..."

docker run -d \
    --gpus all \
    --ipc=host \
    --shm-size=16GB \
    --name "$GEMMA_CONTAINER" \
    -v "$GEMMA_LOCAL_MODEL_DIR:/mnt/models" \
    -p "${GEMMA_PORT}:8000" \
    "$GEMMA_IMAGE" \
    --model "$GEMMA_MODEL_PATH" \
    --served-model-name "$GEMMA_SERVED_MODEL_NAME" \
    --gpu-memory-utilization "$GEMMA_KVCACHE_PERCENT" \
    --max-model-len "$GEMMA_MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000

echo "    Container: $GEMMA_CONTAINER"
echo "    Endpoint:  http://0.0.0.0:${GEMMA_PORT}/v1/chat/completions"
echo "    Model:     $GEMMA_SERVED_MODEL_NAME"