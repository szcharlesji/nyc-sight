#!/usr/bin/env bash
# Set up and build the parakeet-eou-server on DGX Spark (aarch64).
# Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$REPO_ROOT/parakeet-eou-server"

# 1. Ensure Rust is available
if ! command -v cargo >/dev/null 2>&1; then
    echo "[parakeet-setup] installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
echo "[parakeet-setup] cargo $(cargo --version)"

# 2. Download the ONNX model files
echo "[parakeet-setup] fetching model..."
bash "$SERVER_DIR/scripts/download_model.sh" "$SERVER_DIR/models/eou"

# 3. Build release binary (CPU)
echo "[parakeet-setup] building release binary (CPU)..."
( cd "$SERVER_DIR" && cargo build --release )

BIN="$SERVER_DIR/target/release/parakeet-eou-server"
echo
echo "[parakeet-setup] done. Binary: $BIN"

if command -v tailscale >/dev/null 2>&1; then
    TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
    if [[ -n "$TS_IP" ]]; then
        echo "[parakeet-setup] Tailscale IP: $TS_IP"
        echo "[parakeet-setup] Phone connects to: ws://$TS_IP:3030/asr"
    fi
fi

echo
echo "Run with:"
echo "  RUST_LOG=info $BIN --model-dir $SERVER_DIR/models/eou --host 0.0.0.0 --port 3030"
