#!/usr/bin/env bash
# Download the Parakeet EOU 120M ONNX model files from HuggingFace.
# Idempotent: re-running skips files that already exist and are non-empty.
set -euo pipefail

DEST="${1:-./models/eou}"
BASE="https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx"

FILES=(
    "encoder.onnx"
    "decoder_joint.onnx"
    "tokenizer.json"
)

mkdir -p "$DEST"
cd "$DEST"

for f in "${FILES[@]}"; do
    if [[ -s "$f" ]]; then
        echo "[skip] $f already present ($(du -h "$f" | cut -f1))"
    else
        echo "[get ] $f"
        wget -c --show-progress -O "$f.part" "$BASE/$f"
        mv "$f.part" "$f"
    fi
done

echo "Model files ready in $DEST"
ls -lh
