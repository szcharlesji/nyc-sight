# parakeet-eou-server

Rust-based streaming ASR WebSocket server for the Spark Sight blind navigation
assistant. Runs the NVIDIA Parakeet EOU 120M model via `parakeet-rs` on DGX
Spark (aarch64 + GB10). CPU-first: no prebuilt ONNX Runtime GPU binary exists
for aarch64 + CUDA 13 + sm_121, so the default build targets CPU. See
`../parakeet_eou_server.md` for the full design spec.

## Build

```bash
# From this directory
./scripts/download_model.sh            # pulls ~481MB of ONNX model files
cargo build --release
```

## Run

```bash
RUST_LOG=info ./target/release/parakeet-eou-server \
  --model-dir ./models/eou \
  --host 0.0.0.0 \
  --port 3030

# Get Tailscale IP for the phone client
tailscale ip -4
```

CLI flags:

| Flag             | Default          | Description                          |
| ---------------- | ---------------- | ------------------------------------ |
| `--model-dir`    | `./models/eou`   | Dir with the 3 ONNX/tokenizer files  |
| `--host`         | `0.0.0.0`        | Bind host (Tailscale-friendly)       |
| `--port`         | `3030`           | Bind port                            |
| `--max-sessions` | `2`              | Concurrent WebSocket sessions        |
| `--log-level`    | `info`           | Overridden by `RUST_LOG` if set      |

## Endpoints

- `GET /asr` — WebSocket. Accept binary frames of little-endian `f32` PCM at
  16 kHz mono, or JSON text frames `{"type": "audio", "data": "<b64>",
  "sample_rate": 16000}`. Send `{"type": "eos"}` to flush at end of stream.
  Responses are JSON text frames: `{"type": "partial", "text": "..."}`,
  `{"type": "eou", "text": "..."}`, or `{"type": "error", "message": "..."}`.
- `GET /health` — Returns status JSON with session count and rolling average
  per-chunk inference latency.

## Testing

```bash
pip install websockets
python scripts/test_client.py test_16k.wav
# or over Tailscale
python scripts/test_client.py test_16k.wav --server ws://100.64.0.5:3030/asr
```

## GPU (experimental)

Requires a custom-built `libonnxruntime.so` for aarch64 + CUDA 13 + sm_121.
See `github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell`.

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
cargo build --release --features cuda
```
