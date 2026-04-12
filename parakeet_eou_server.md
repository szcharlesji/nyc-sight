# Claude Code Prompt: Parakeet EOU Streaming ASR Server for DGX Spark (GB10)

## Goal

Build a Rust-based streaming ASR (Automatic Speech Recognition) WebSocket server using `parakeet-rs` and the NVIDIA Parakeet EOU 120M model. This server is one component of a blind navigation assistant running entirely on-device on an NVIDIA DGX Spark. The phone sends raw PCM audio chunks over WebSocket; the server transcribes them in real-time and returns incremental text plus end-of-utterance signals.

---

## System Architecture (Current)

The blind navigation assistant has been simplified from the original dual-agent design. Cosmos Reason and Nemotron 120B have been **removed** — the system now uses:

- **YOLO** — fast object detection for real-time obstacle/hazard identification from the phone camera
- **Gemma 4 26B A4B** (MoE, ~4B active params) — multimodal VLM for conversational understanding, scene description, and NYC Open Data queries. Served via Ollama on the Spark.
- **Parakeet EOU 120M** (this server) — streaming ASR with end-of-utterance detection
- **TTS** — text-to-speech for spoken responses back to the user

This is a leaner, faster pipeline. YOLO handles the always-on hazard detection that Cosmos Reason previously did (at much lower latency and memory cost), while Gemma 4 26B handles the conversational/reasoning role that Nemotron 120B previously filled (at ~18GB quantized vs ~60GB+ for Nemotron).

```
┌──────────────────────────────────────────────────────────┐
│                   PHONE (chest-mounted)                   │
│                                                           │
│  Camera frames ──► WebSocket ──────────┐                  │
│  Microphone PCM ──► WebSocket ─────┐   │                  │
│  GPS coords ──► WebSocket ─────┐   │   │                  │
│  Speaker ◄── WebSocket ◄──┐    │   │   │                  │
└───────────────────────────┼────┼───┼───┼──────────────────┘
        Tailscale VPN       │    │   │   │
        ┌───────────────────┘    │   │   │
        ▼                        ▼   │   ▼
┌──────────────────────────────────────────────────────────┐
│                    DGX SPARK (GB10)                       │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Parakeet EOU │  │   YOLO       │  │  Gemma 4     │   │
│  │ (this server)│  │  (TensorRT)  │  │  26B A4B     │   │
│  │  Rust/ONNX   │  │  Detection   │  │  (Ollama)    │   │
│  │  :3030/asr   │  │  :8080       │  │  :11434      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │           │
│         └──────────┬───────┘──────────────────┘           │
│                    ▼                                      │
│            Orchestrator (Python)                          │
│              + NYC Open Data (SQLite)                     │
│              + TTS output → phone                        │
└──────────────────────────────────────────────────────────┘
```

### Data flow for a user utterance:

1. Phone streams 16kHz mono PCM audio over WebSocket to `ws://<tailscale-ip>:3030/asr`
2. Parakeet EOU transcribes incrementally, emits partial text events
3. When `<EOU>` token detected → emit `eou` event with full transcript
4. Orchestrator receives the completed utterance, passes it to Gemma 4 along with YOLO detections and latest camera frame for context
5. Gemma 4 responds → TTS → audio sent back to phone

---

## Hardware & Platform Constraints

- **Device**: NVIDIA DGX Spark (Acer Veriton GN100)
- **CPU**: ARM64 Grace (Cortex-X925 + Cortex-A725, 20 cores)
- **GPU**: NVIDIA GB10 Blackwell, compute capability 12.1, CUDA 13.0
- **Memory**: 128GB unified
- **OS**: DGX OS (Ubuntu 24.04-based, aarch64)
- **Network**: Tailscale VPN for phone ↔ Spark communication. The server binds to `0.0.0.0` but is only reachable via Tailscale IP (100.x.x.x).

### Critical: ONNX Runtime on GB10

**No prebuilt ONNX Runtime GPU binary exists for aarch64 + CUDA 13 + sm_121 as of April 2026.** The `ort` Rust crate cannot auto-download a working CUDA binary for this platform.

**Strategy**: Build for CPU-first. The EOU model is only 120M parameters (~481MB total). CPU inference on Grace's 20 ARM cores should achieve real-time latency. The Rust decode loop eliminates Python interpreter overhead — the primary bottleneck in streaming RNNT decoding.

**If GPU support is needed later**, use the `load-dynamic` feature of the `ort` crate and point `ORT_DYLIB_PATH` to a custom-built `libonnxruntime.so`. Reference build: `github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell` (ORT v1.24.4, sm_121). Make GPU an optional feature flag that falls back gracefully.

### Memory Budget Context

The GB10 has 128GB unified memory shared between CPU and GPU. Current allocation:
- Gemma 4 26B A4B (Q4 via Ollama): ~18GB
- YOLO (TensorRT): ~2-4GB
- NYC Open Data SQLite: negligible
- TTS: TBD (small model)
- **Parakeet EOU (this server): ~2GB** — 481MB model files + ONNX runtime overhead

There is plenty of headroom. The EOU model's small footprint is one of the main reasons the architecture switch works.

---

## Model Files

Download from: `https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx`

Three files, all must be in the same directory:
- `encoder.onnx` (459 MB) — FastConformer streaming encoder
- `decoder_joint.onnx` (21.3 MB) — RNNT joint/decoder network
- `tokenizer.json` (20.1 KB) — HuggingFace tokenizer vocabulary

---

## Networking: Tailscale

The phone connects to the DGX Spark over Tailscale VPN. This means:

1. **The server binds to `0.0.0.0`** — not `127.0.0.1`. Tailscale traffic arrives on the `tailscale0` interface.
2. **The phone connects to `ws://100.x.x.x:3030/asr`** where `100.x.x.x` is the Spark's Tailscale IP.
3. **No TLS required** — Tailscale provides WireGuard encryption at the tunnel level. Adding TLS on top would add unnecessary latency and setup complexity.
4. **Firewall**: Tailscale ACLs handle access control. The server does NOT need to implement authentication.
5. **Latency**: Tailscale adds ~1-5ms overhead on a local network. Over cellular/remote, expect 20-80ms. This is additive to inference latency, so keeping per-chunk inference fast is critical.
6. **The health endpoint should be accessible** for monitoring: `http://100.x.x.x:3030/health`

### CLI should support Tailscale-friendly defaults:

```bash
# Default: bind all interfaces, no TLS, Tailscale-friendly
parakeet-eou-server --model-dir ./models/eou --port 3030

# Explicit Tailscale IP binding if desired
parakeet-eou-server --model-dir ./models/eou --host 100.64.0.5 --port 3030
```

---

## Functional Requirements

### 1. WebSocket Streaming Endpoint

- Path: `GET /asr` (WebSocket upgrade)
- Accept binary frames containing raw 16kHz mono f32 PCM audio (preferred, lowest overhead)
- Also accept text frames with JSON: `{"type": "audio", "data": "<base64-encoded PCM>", "sample_rate": 16000}` for backward compatibility with the existing phone client
- Process audio in 160ms chunks (2560 samples at 16kHz)
- Return JSON text frames with transcription events

### 2. Transcription Events

```jsonc
// Partial transcript (emitted every chunk that produces new text)
{"type": "partial", "text": "what is the nearest"}

// EOU detected — user has finished speaking
// This triggers the orchestrator to dispatch to Gemma 4
{"type": "eou", "text": "what is the nearest subway entrance"}

// Error
{"type": "error", "message": "Model inference failed: ..."}
```

The EOU model emits an `<EOU>` token when it detects end-of-utterance. When this token appears, emit the `eou` event with the full accumulated transcript, then reset internal state for the next utterance.

### 3. Health / Status Endpoint

- `GET /health` → `{"status": "ok", "model": "parakeet-eou-120m", "device": "cpu", "sessions": 1, "avg_latency_ms": 45.2}`
- Include: model load time, current active session count, rolling average per-chunk inference latency
- This endpoint is used for monitoring over Tailscale — keep it lightweight

### 4. Session Management

- Each WebSocket connection gets its own `ParakeetEOU` instance
- Model instances maintain internal RNNT decoder state across chunks within a session
- On disconnect, clean up the model instance and log session duration + total audio processed
- Max concurrent sessions: configurable, default 2 (single user device, but phone may reconnect)

### 5. Audio Preprocessing

- Accept: 16kHz mono f32 PCM (preferred), or 16-bit PCM (convert to f32 by dividing by 32768.0)
- If sample rate != 16kHz, reject with error (resampling adds latency; the phone should send 16kHz)
- Buffer incoming audio and dispatch to the model in 160ms chunks (2560 samples)
- Handle partial chunks at stream end (pass `is_final=true` on the last chunk)

### 6. Graceful Shutdown

- Handle SIGTERM/SIGINT — drain active sessions, log final stats
- This matters because the server runs alongside Ollama, YOLO, and the orchestrator on the same machine. Clean shutdown prevents port conflicts on restart.

---

## Technical Implementation Details

### Cargo.toml

```toml
[package]
name = "parakeet-eou-server"
version = "0.1.0"
edition = "2021"

[dependencies]
parakeet-rs = "0.3"
tokio = { version = "1", features = ["full"] }
axum = { version = "0.8", features = ["ws"] }
axum-extra = "0.10"
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1"
base64 = "0.22"
clap = { version = "4", features = ["derive"] }

[features]
default = []
cuda = ["parakeet-rs/cuda"]
```

### Key parakeet-rs API Usage

```rust
use parakeet_rs::ParakeetEOU;

// Load model once at startup
let mut model = ParakeetEOU::from_pretrained("./models/eou", None)?;

// For CUDA (optional, requires custom-built ORT for GB10):
// use parakeet_rs::{ExecutionConfig, ExecutionProvider};
// let config = ExecutionConfig::new()
//     .with_execution_provider(ExecutionProvider::Cuda);
// let mut model = ParakeetEOU::from_pretrained("./models/eou", Some(config))?;

// Streaming transcription — call repeatedly with sequential 160ms chunks
const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
let text = model.transcribe(chunk, is_final)?;
// Check for <EOU> token in output
```

**Important**: `ParakeetEOU` maintains internal decoder cache state. Each WebSocket session needs its own instance. The model files load from disk once but each `from_pretrained` creates a new ONNX Runtime session.

### CLI Arguments

```
parakeet-eou-server \
  --model-dir ./models/eou \
  --host 0.0.0.0 \
  --port 3030 \
  --max-sessions 2 \
  --log-level info
```

### Logging

- Use `tracing` + `tracing-subscriber` — **mandatory**. Without it, ONNX Runtime EP registration is invisible and debugging is impossible.
- Log per-chunk inference latency at DEBUG level
- Log session lifecycle (connect/disconnect/EOU) at INFO level
- Startup banner: model path, device (CPU/CUDA), model load time, bound address

### Error Handling

- Model loading failure at startup → exit with clear error message listing model directory contents
- Inference failure on a chunk → send error event to client, keep session alive
- Invalid audio format → send error event explaining expected format
- Connection drop mid-stream → clean up session, log audio duration processed

---

## File Structure

```
parakeet-eou-server/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI parsing, tracing init, model loading, server startup
│   ├── server.rs             # axum router, WebSocket handler, health endpoint
│   ├── transcriber.rs        # ParakeetEOU wrapper, chunk buffering, EOU detection, transcript accumulation
│   └── audio.rs              # PCM format conversion (i16→f32), base64 decoding
├── scripts/
│   ├── download_model.sh     # Downloads the 3 model files from HuggingFace
│   └── test_client.py        # Python WebSocket test client
├── README.md
└── .cargo/
    └── config.toml           # aarch64-specific linker settings if needed
```

---

## Testing

### Integration Test Script

Create `scripts/test_client.py` — a Python WebSocket client that replays a WAV file with realistic timing. This is the primary way to verify the server works end-to-end over Tailscale.

```python
#!/usr/bin/env python3
"""
Test client for the Parakeet EOU streaming ASR server.

Usage:
    # Local testing
    python scripts/test_client.py test.wav

    # Over Tailscale
    python scripts/test_client.py test.wav --server ws://100.64.0.5:3030/asr

    # With timing simulation disabled (fast mode)
    python scripts/test_client.py test.wav --fast
"""
import argparse
import asyncio
import json
import struct
import time
import wave

import websockets


async def test_streaming(wav_path: str, server_url: str, fast: bool = False):
    with wave.open(wav_path, "rb") as wf:
        assert wf.getsampwidth() == 2 and wf.getnchannels() == 1, \
            f"Expected 16-bit mono WAV, got {wf.getsampwidth()*8}-bit {wf.getnchannels()}ch"
        sr = wf.getframerate()
        if sr != 16000:
            print(f"WARNING: WAV sample rate is {sr}Hz, server expects 16000Hz")
        frames = wf.readframes(wf.getnframes())

    # Convert i16 to f32
    samples = struct.unpack(f"<{len(frames)//2}h", frames)
    f32_bytes = struct.pack(f"<{len(samples)}f", *[s / 32768.0 for s in samples])

    chunk_bytes = 2560 * 4  # 160ms of f32 samples
    total_chunks = (len(f32_bytes) + chunk_bytes - 1) // chunk_bytes
    audio_duration = len(samples) / 16000

    print(f"Audio: {audio_duration:.1f}s, {total_chunks} chunks")
    print(f"Server: {server_url}")
    print(f"Mode: {'fast (no timing)' if fast else 'realtime (160ms spacing)'}")
    print("---")

    t0 = time.monotonic()

    async with websockets.connect(server_url) as ws:
        for i in range(0, len(f32_bytes), chunk_bytes):
            chunk = f32_bytes[i : i + chunk_bytes]
            await ws.send(chunk)

            if not fast:
                await asyncio.sleep(0.16)

            # Non-blocking receive
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    event = json.loads(msg)
                    elapsed = time.monotonic() - t0
                    etype = event["type"]
                    text = event.get("text", event.get("message", ""))
                    print(f"[{elapsed:6.2f}s] [{etype:7s}] {text}")
                    if etype == "eou":
                        print("--- EOU detected ---")
            except (asyncio.TimeoutError, Exception):
                pass

        # Signal end of audio stream
        await ws.send(json.dumps({"type": "eos"}))

        # Drain remaining events
        try:
            async for msg in ws:
                event = json.loads(msg)
                elapsed = time.monotonic() - t0
                etype = event["type"]
                text = event.get("text", event.get("message", ""))
                print(f"[{elapsed:6.2f}s] [{etype:7s}] {text}")
                if etype == "eou":
                    print("--- EOU detected ---")
                    break
        except websockets.exceptions.ConnectionClosed:
            pass

    total = time.monotonic() - t0
    print(f"\nDone in {total:.2f}s (audio was {audio_duration:.1f}s, RTF={total/audio_duration:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parakeet EOU ASR server")
    parser.add_argument("wav", help="Path to 16kHz mono WAV file")
    parser.add_argument("--server", default="ws://localhost:3030/asr", help="WebSocket URL")
    parser.add_argument("--fast", action="store_true", help="Send chunks without timing delay")
    args = parser.parse_args()
    asyncio.run(test_streaming(args.wav, args.server, args.fast))
```

### Download a Test WAV

```bash
# LibriSpeech sample
wget -O test.wav https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav
# Resample to 16kHz if needed:
# ffmpeg -i test.wav -ar 16000 -ac 1 test_16k.wav
```

---

## Build & Run on GB10

```bash
# 1. Install Rust (if not present on the Spark)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. Create project (Claude Code will scaffold this)
cd ~/projects

# 3. Download model
mkdir -p models/eou && cd models/eou
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/encoder.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/decoder_joint.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/tokenizer.json
cd ../..

# 4. Build (CPU-only — safe default for GB10)
cargo build --release

# 5. Run (binds to all interfaces for Tailscale access)
RUST_LOG=info ./target/release/parakeet-eou-server \
  --model-dir ./models/eou \
  --host 0.0.0.0 \
  --port 3030

# 6. Get Tailscale IP
tailscale ip -4
# e.g., 100.64.0.5

# 7. Test from your laptop (over Tailscale)
pip install websockets
python scripts/test_client.py test_16k.wav --server ws://100.64.0.5:3030/asr
```

### Optional: CUDA Build

```bash
# Only if you have a custom-built libonnxruntime.so for aarch64 + CUDA 13 + sm_121
# See: github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
cargo build --release --features cuda
```

### Running Alongside Other Services

The ASR server coexists with:
- Ollama (Gemma 4, port 11434)
- YOLO inference server (port 8080)
- Orchestrator (Python, manages the pipeline)
- TTS service

Use `tmux` or `systemd` to manage all processes. Example tmux layout:

```bash
tmux new-session -d -s asr './target/release/parakeet-eou-server --model-dir ./models/eou --port 3030'
tmux new-session -d -s ollama 'ollama serve'
tmux new-session -d -s yolo './yolo-server --port 8080'
tmux new-session -d -s orchestrator 'python app/main.py'
```

---

## Performance Targets

- **Chunk latency**: <200ms per 160ms chunk on CPU (real-time factor < 1.25)
- **EOU detection latency**: <300ms after user stops speaking
- **Model load time**: <5s
- **Memory**: <2GB RSS
- **Concurrent sessions**: 2 simultaneous streams on CPU

If CPU latency exceeds 200ms per chunk, investigate:
1. ONNX Runtime threading: try `ORT_NUM_THREADS=10` (half the Grace cores — leave headroom for Ollama/YOLO)
2. Profile encoder vs decode loop — add per-stage timing
3. The Nemotron 0.6B streaming model is an alternative (larger but cache-aware), though it would need its own ORT build

---

## Notes for Claude Code

- This is aarch64 Linux (ARM64). Watch for platform-specific assumptions.
- The `ort` crate version must match what `parakeet-rs` v0.3 pins. Check its Cargo.toml before adding `ort` as a direct dependency. Do NOT add separate `onnxruntime` or `onnxruntime-sys` deps.
- The EOU model outputs text **without punctuation or capitalization**. This is expected — Gemma 4 handles text normalization downstream.
- Audio arrives from a phone over Tailscale (potentially over cellular). Handle partial messages, dropped connections, and reconnections gracefully.
- The server binds to `0.0.0.0` by default. There is no authentication — Tailscale ACLs provide access control.
- Do NOT use TLS. Tailscale provides WireGuard encryption. Adding TLS adds latency and certificate management overhead for zero security benefit.
- The `download_model.sh` script should be idempotent — skip files that already exist (the encoder is 459MB; re-downloading it during a hackathon is painful).
- CORS is not needed since the phone client uses native WebSocket, not browser fetch. But include permissive CORS headers on the health endpoint for convenience during debugging.
