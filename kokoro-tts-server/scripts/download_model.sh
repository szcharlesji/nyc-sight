#!/usr/bin/env bash
# Download Kokoro-82M weights + all voice packs from Hugging Face.
# Idempotent: skips files that already exist.
#
# Usage:
#   scripts/download_model.sh [DEST_DIR]
#
# Default DEST_DIR: ./models/kokoro-82m
set -euo pipefail

DEST="${1:-./models/kokoro-82m}"
REPO="hexgrad/Kokoro-82M"

mkdir -p "$DEST"

echo "[kokoro-download] target: $DEST"
echo "[kokoro-download] repo:   $REPO"

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
    HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN="huggingface-cli"
fi

if [[ -n "$HF_BIN" ]]; then
    echo "[kokoro-download] using $HF_BIN (supports resume + parallel)"
    # Pull model weights, config, and the entire voices/ directory.
    "$HF_BIN" download "$REPO" \
        --local-dir "$DEST" \
        --include "*.pth" "*.json" "voices/*.pt"
else
    echo "[kokoro-download] huggingface-cli not found; falling back to curl"
    BASE="https://huggingface.co/${REPO}/resolve/main"

    FILES=(
        "kokoro-v1_0.pth"
        "config.json"
    )
    VOICES=(
        af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky
        am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
        bf_alice bf_emma bf_isabella bf_lily
        bm_daniel bm_fable bm_george bm_lewis
        ef_dora em_alex em_santa
        ff_siwis
        hf_alpha hf_beta hm_omega hm_psi
        if_sara im_nicola
        jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo
        pf_dora pm_alex pm_santa
        zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang
    )

    download() {
        local src="$1" dst="$2"
        if [[ -s "$dst" ]]; then
            echo "[kokoro-download] skip (exists): $dst"
            return 0
        fi
        mkdir -p "$(dirname "$dst")"
        echo "[kokoro-download] fetch: $src"
        curl -fL --retry 3 --retry-delay 2 -o "$dst.part" "$src"
        mv "$dst.part" "$dst"
    }

    for f in "${FILES[@]}"; do
        download "$BASE/$f" "$DEST/$f"
    done
    for v in "${VOICES[@]}"; do
        download "$BASE/voices/$v.pt" "$DEST/voices/$v.pt"
    done
fi

echo
echo "[kokoro-download] done. Contents:"
ls -lh "$DEST" || true
echo
echo "Voice pack count: $(ls -1 "$DEST/voices" 2>/dev/null | wc -l)"
