#!/usr/bin/env bash
# Set up and run the kokoro-tts-server on DGX Spark (aarch64 + GB10).
# Idempotent: safe to re-run.
#
# Usage:
#   scripts/kokoro_tts_setup.sh                 # server deps only
#   scripts/kokoro_tts_setup.sh --with-client   # also install pyaudio for client demo
set -euo pipefail

WITH_CLIENT=0
for arg in "$@"; do
    case "$arg" in
        --with-client) WITH_CLIENT=1 ;;
        *) echo "[kokoro-setup] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$REPO_ROOT/kokoro-tts-server"
VENV_DIR="$SERVER_DIR/.venv"

# 1. System deps
need_apt=()
command -v espeak-ng >/dev/null 2>&1 || need_apt+=(espeak-ng)
command -v ffmpeg >/dev/null 2>&1 || need_apt+=(ffmpeg)
# python3.12-dev is required so packages like pyaudio can build C extensions.
dpkg -s python3.12-dev >/dev/null 2>&1 || need_apt+=(python3.12-dev)
dpkg -s python3.12-venv >/dev/null 2>&1 || need_apt+=(python3.12-venv)

if (( WITH_CLIENT )); then
    dpkg -s libportaudio2 >/dev/null 2>&1 || need_apt+=(libportaudio2)
    dpkg -s portaudio19-dev >/dev/null 2>&1 || need_apt+=(portaudio19-dev)
fi

if (( ${#need_apt[@]} )); then
    echo "[kokoro-setup] installing apt deps: ${need_apt[*]}"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${need_apt[@]}"
else
    echo "[kokoro-setup] apt deps already present"
fi

# 2. Locate a compatible Python. kokoro 0.9.4 requires >=3.10,<3.13 — prefer 3.12.
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "[kokoro-setup] ERROR: need python3.10, 3.11, or 3.12 for kokoro>=0.9.4." >&2
    echo "[kokoro-setup]        Install with: sudo apt-get install python3.12 python3.12-venv" >&2
    exit 1
fi
echo "[kokoro-setup] using $PYTHON_BIN ($($PYTHON_BIN --version))"

# 3. Pick an installer. uv is massively faster; fall back to pip.
USE_UV=0
if command -v uv >/dev/null 2>&1; then
    USE_UV=1
    echo "[kokoro-setup] using uv $(uv --version | awk '{print $2}')"
else
    echo "[kokoro-setup] uv not found; using pip (install uv for a 5-10x speedup: https://docs.astral.sh/uv/)"
fi

# 4. Venv (rebuild if the existing one was made with the wrong Python)
if [[ -d "$VENV_DIR" ]]; then
    existing_ver="$("$VENV_DIR/bin/python" --version 2>&1 || echo unknown)"
    wanted_ver="$("$PYTHON_BIN" --version)"
    if [[ "$existing_ver" != "$wanted_ver" ]]; then
        echo "[kokoro-setup] existing venv is $existing_ver; rebuilding with $wanted_ver"
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[kokoro-setup] creating venv at $VENV_DIR"
    if (( USE_UV )); then
        uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi

VENV_PY="$VENV_DIR/bin/python"

# 5. Install deps
echo "[kokoro-setup] installing server deps"
if (( USE_UV )); then
    uv pip install --python "$VENV_PY" -r "$SERVER_DIR/requirements.txt"
else
    "$VENV_PY" -m pip install --upgrade pip wheel
    "$VENV_PY" -m pip install -r "$SERVER_DIR/requirements.txt"
fi

if (( WITH_CLIENT )); then
    echo "[kokoro-setup] installing client deps (pyaudio)"
    if (( USE_UV )); then
        uv pip install --python "$VENV_PY" -r "$SERVER_DIR/requirements-client.txt"
    else
        "$VENV_PY" -m pip install -r "$SERVER_DIR/requirements-client.txt"
    fi
fi

# 6. Download model weights + voices
echo "[kokoro-setup] fetching model..."
bash "$SERVER_DIR/scripts/download_model.sh" "$SERVER_DIR/models/kokoro-82m"

echo
echo "[kokoro-setup] done."
echo
echo "Run with:"
echo "  cd $SERVER_DIR && source .venv/bin/activate"
echo "  python -m uvicorn server:app --host 0.0.0.0 --port 8880"
echo "  # use 'python -m uvicorn' to ensure the venv's uvicorn is used,"
echo "  # not a system install on a different Python."
echo

if command -v tailscale >/dev/null 2>&1; then
    TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
    if [[ -n "$TS_IP" ]]; then
        echo "[kokoro-setup] Tailscale IP: $TS_IP"
        echo "[kokoro-setup] OpenAI base_url: http://$TS_IP:8880/v1"
    fi
fi
