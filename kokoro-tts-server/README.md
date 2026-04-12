# kokoro-tts-server

High-throughput, low-latency Kokoro-82M TTS server for the Spark Sight voice
agent on DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13, sm_121). Exposes an
OpenAI-compatible `/v1/audio/speech` endpoint with streaming PCM / MP3 / Opus /
FLAC output (WAV is buffered — its header needs a length). See
`../kokoro-tts-spark-prompt.md` for the full design spec.

## Build

Use the top-level helper script (recommended — uses `uv` if available):

```bash
../scripts/kokoro_tts_setup.sh                 # server deps only
../scripts/kokoro_tts_setup.sh --with-client   # also install pyaudio for client demo
```

Manual (from this directory):

```bash
# Kokoro 0.9.4 requires Python >=3.10,<3.13 — use 3.12.
./scripts/download_model.sh          # pulls Kokoro-82M weights + 54 voices
uv venv --python python3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Or with plain pip: python3.12 -m venv .venv && pip install -r requirements.txt

# System deps (Ubuntu):
sudo apt-get install python3.12 python3.12-venv python3.12-dev espeak-ng ffmpeg

# Optional (client.py speaker playback):
sudo apt-get install libportaudio2 portaudio19-dev
uv pip install -r requirements-client.txt
```

Or use the top-level helper:

```bash
../scripts/kokoro_tts_setup.sh
```

## Run

```bash
# Use `python -m uvicorn` so PATH can't pick up a system uvicorn on the wrong Python.
python -m uvicorn server:app --host 0.0.0.0 --port 8880
# or, equivalently, with CLI flags:
python server.py --host 0.0.0.0 --port 8880 --dtype fp16 --batch-size 4

# Phone / orchestrator connects to:
tailscale ip -4   # e.g. 100.64.0.5
# base_url = http://100.64.0.5:8880/v1
```

## CLI flags

| Flag                 | Default              | Description                                                  |
| -------------------- | -------------------- | ------------------------------------------------------------ |
| `--host`             | `0.0.0.0`            | Bind host (Tailscale-friendly).                              |
| `--port`             | `8880`               | Bind port. OpenAI-compat clients default here.               |
| `--model-dir`        | `./models/kokoro-82m`| Directory holding `kokoro-v1_0.pth`, `config.json`, `voices/*.pt`. |
| `--lang-code`        | `a`                  | Kokoro lang code (`a` = American English).                   |
| `--dtype`            | `fp16`               | `fp32` / `fp16` / `bf16`. Voice style vectors stay FP32.     |
| `--batch-size`       | `4`                  | Max chunks per batched forward.                              |
| `--first-batch-size` | `1`                  | TTFA optimization: first group uses this batch size.         |
| `--torch-compile`    | off                  | Try `torch.compile(mode="reduce-overhead")` on the forward.  |
| `--max-concurrent`   | `4`                  | HTTP semaphore; excess requests get 429.                     |
| `--verify-patch`     | off                  | On warmup, compare batched vs. sequential outputs.           |
| `--device`           | `cuda`               | Falls back to `cpu` if CUDA isn't available.                 |
| `--log-level`        | `info`               | uvicorn / backend log level.                                 |

## Endpoints

- **`POST /v1/audio/speech`** — OpenAI-compatible. Body:
  ```json
  {
    "model": "kokoro",
    "input": "Hello world.",
    "voice": "af_bella",
    "response_format": "pcm",
    "speed": 1.0,
    "stream": true
  }
  ```
  `response_format` ∈ `pcm | wav | mp3 | opus | flac`. PCM / MP3 / Opus / FLAC
  stream chunked; WAV is buffered and returned as a single response. PCM is raw
  16-bit little-endian mono at 24 kHz (sample rate echoed back in
  `X-Sample-Rate`).

- **`GET /v1/audio/voices`** — returns `{"voices": [...]}` (OpenAI doesn't
  define this, but it's useful for clients that want to enumerate).

- **`GET /health`** — returns device, dtype, voice count, VRAM stats, active
  request count.

### Voice blending

The `voice` field accepts blends:

```
af_bella                      # single voice
af_bella+af_sky               # equal-weight 50/50 blend
af_bella(2)+af_sky(1)         # 2:1 blend (weights are normalized)
```

Components are `.pt` voice packs under `models/kokoro-82m/voices/`. The blend
is a weighted sum of each pack's full `(511, 1, 256)` style tensor, so the
downstream per-token-count style selection still works correctly.

### Example requests

```bash
# PCM streaming to stdout
curl -sS -X POST http://localhost:8880/v1/audio/speech \
     -H 'content-type: application/json' \
     -d '{"input":"hello","voice":"af_bella","response_format":"pcm"}' \
     > hello.pcm

# Blended voices, MP3
curl -sS -X POST http://localhost:8880/v1/audio/speech \
     -H 'content-type: application/json' \
     -d '{"input":"hello","voice":"af_bella+af_sky","response_format":"mp3"}' \
     > hello.mp3
```

## Testing

```bash
# Smoke: posts once, writes WAV, prints TTFB
python scripts/test_client.py --voice af_bella --text "hello there" --format wav --out out.wav

# Live speaker playback (requires portaudio)
python client.py --server http://localhost:8880 --text "Hello from Kokoro."

# LLM interleave demo
python client.py --demo-llm-interleave --server http://localhost:8880
```

## Benchmarks

```bash
# Direct backend (no HTTP overhead)
python benchmark.py --full --output benchmark_results.json

# Through the HTTP stack
python benchmark.py --client-mode http://localhost:8880 --output client_results.json
```

Measures time-to-first-audio (TTFA), total generation time, RTF (generation
seconds per audio second — lower is better), peak VRAM, and throughput. Sweeps
input size (5/10/25/50/100/200 words), precision (FP32/FP16/BF16), batch size
(1/2/4/8), and `torch.compile` on/off.

Performance targets from the spec:

| Metric                | Target | Stretch |
| --------------------- | ------ | ------- |
| TTFA (one sentence)   | <100ms | <50ms   |
| RTF (GPU, FP16)       | <0.05  | <0.02   |
| Peak VRAM             | <3GB   | <2GB    |

## Docker

```bash
docker build -t kokoro-tts-server:latest .
docker run --rm --gpus all -p 8880:8880 kokoro-tts-server:latest
```

The image is built from `nvcr.io/nvidia/pytorch:25.04-py3` (first Blackwell-aware
NGC tag). Bump to `25.07-py3` or newer if your host driver requires CUDA 13
userspace. Model weights and all 54 voice packs are baked into the image at
build time — remove the `RUN ./scripts/download_model.sh` line from the
`Dockerfile` if you prefer runtime download.

## Integration with Spark Sight

Add to `.env` (see `../.env.example`):

```
KOKORO_TTS_URL=http://<GB10_IP>:8880/v1
KOKORO_TTS_VOICE=af_bella
KOKORO_TTS_FORMAT=pcm
```

Any OpenAI SDK works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<GB10_IP>:8880/v1", api_key="not-used")
resp = client.audio.speech.create(
    model="kokoro",
    voice="af_bella",
    input="Hello from Spark Sight.",
    response_format="pcm",
)
```

## Architecture notes

- **Inference path**: loaded via `kokoro>=0.9.4` (`KPipeline` owns the Misaki
  G2P and vocab); the bare `KModel` is used for our own batched forward.
- **Batched alignment**: the stock Kokoro forward builds `pred_aln_trg` with
  `torch.repeat_interleave` which doesn't vectorize across samples. When
  `batch_size > 1`, the backend wraps the forward in a
  `_patched_alignment` context that routes replication through the NimbleEdge
  mask-based construction (see `backend.py`). If any version-sensitive call
  site breaks the batched path, the server logs once and falls back to
  per-chunk sequential inference for the rest of its lifetime.
- **TTFA**: `first_batch_size=1` means the first chunk is emitted alone (low
  latency); subsequent chunks pack into full batches (high throughput).
- **Encoders**: PCM is a passthrough; WAV is buffered; FLAC uses
  `soundfile.SoundFile` incrementally; MP3/Opus use PyAV with an internal
  `_GrowBuffer` sink that drains container bytes between calls. Encoders run
  in a thread executor so MP3/Opus CPU work never blocks the event loop.
- **Concurrency**: an `asyncio.Lock` inside the backend serializes GPU
  inference; an `asyncio.Semaphore(max_concurrent_requests)` at the HTTP
  layer 429s when the queue is saturated.

## Non-goals

Voice cloning, multi-language-in-one-request, WebSocket transport, TensorRT,
ONNX GPU build, training / fine-tuning. CPU ONNX fallback is a potential
follow-up (see spec).
