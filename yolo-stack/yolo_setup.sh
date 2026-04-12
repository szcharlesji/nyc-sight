#!/bin/bash
# start_server.sh — launch Spark Sight YOLO11 server in NGC PyTorch container

set -e

IMAGE="sparksight:yolo"
CONTAINER_NAME="yolo-server"
WORKSPACE="$HOME/docker_workspace"   # remap to your actual project root if different
PORT=8081

# ── stop existing container if running ──────────────
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "[*] stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
fi
if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME"
fi

# ── launch ──────────────────────────────────────────
echo "[*] starting $CONTAINER_NAME on port $PORT"
echo "[*] workspace: $WORKSPACE → /workspace/Spark-Sight"

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$PORT:8080" \
    -v "$WORKSPACE:/workspace" \
    -w /workspace/Spark-Sight/yolo-stack \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    python server.py

echo "[*] container started, tailing logs (Ctrl+C to detach)..."
echo "[*] server will be available at http://localhost:$PORT"
echo "[*] health check: curl http://localhost:$PORT/health"
echo ""
docker logs -f "$CONTAINER_NAME"