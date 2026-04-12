# Container API Reference

Endpoints exposed by the two services in `docker-compose.yml`.

| Service        | Container      | Host port | Protocol(s)        |
| -------------- | -------------- | --------- | ------------------ |
| `kokoro-tts`   | `kokoro-tts`   | `8880`    | HTTP (REST)        |
| `parakeet-eou` | `parakeet-eou` | `3030`    | HTTP (REST) + WS   |

Start the stack with `docker compose up -d`; stop with `docker compose down`.
All host-bound; reach them on `http://localhost:<port>` from the Docker host,
or on the host's Tailscale/LAN IP from other devices.

---

## kokoro-tts (port 8880)

Kokoro-82M text-to-speech. FastAPI, OpenAI-compatible `/v1/audio/speech` with
streaming formats. CUDA by default (`--gpus all` via compose's nvidia
reservation).

### `GET /health`

Liveness + runtime stats.

**200 OK**
```json
{
  "status": "ok",
  "device": "cuda",
  "dtype": "fp16",
  "voices": 54,
  "sample_rate": 24000,
  "batched_enabled": true,
  "gpu_memory_allocated_mb": 397.6,
  "gpu_memory_reserved_mb": 746.0,
  "active_requests": 0,
  "max_concurrent": 4
}
```

### `GET /v1/audio/voices`

Lists every voice pack loaded from `models/kokoro-82m/voices/`.

**200 OK**
```json
{ "voices": ["af_alloy", "af_aoede", "af_bella", ..., "zm_yunyang"] }
```

54 voices total — American/British English (`af_*`, `am_*`, `bf_*`, `bm_*`),
plus Spanish, French, Hindi, Italian, Japanese, Portuguese, and Mandarin
variants. Blend two voices with `+` and an optional weight, e.g.
`af_bella+af_sky` or `af_bella*0.7+af_sky*0.3`.

### `POST /v1/audio/speech`

OpenAI-compatible. Streams audio by default (WAV is always buffered).

**Request body**
```json
{
  "model": "kokoro",
  "input": "hello from docker",
  "voice": "af_bella",
  "response_format": "wav",
  "speed": 1.0,
  "stream": true
}
```

| Field             | Type    | Default   | Notes                                                    |
| ----------------- | ------- | --------- | -------------------------------------------------------- |
| `model`           | string  | `kokoro`  | Accepted but ignored — always Kokoro-82M.                |
| `input`           | string  | required  | Text to speak. ≥1 char.                                  |
| `voice`           | string  | `af_bella`| Voice id or blend expression.                            |
| `response_format` | enum    | `pcm`     | One of `pcm`, `wav`, `mp3`, `opus`, `flac`.              |
| `speed`           | float   | `1.0`     | `(0.25, 4.0)` exclusive.                                 |
| `stream`          | bool    | `true`    | Ignored for `wav` (always buffered).                     |

**Response**
- `wav` → `audio/wav`, buffered, RIFF header with length.
- `pcm` → `audio/pcm`, 24 kHz mono f32le, streamed.
- `mp3` → `audio/mpeg`, streamed (default bitrate 128 kbps).
- `opus` → `audio/ogg; codecs=opus`, streamed.
- `flac` → `audio/flac`, streamed.

Streaming responses include headers `X-Sample-Rate: 24000` and
`X-Audio-Format: <fmt>`.

**Error codes**
- `400` — unsupported `response_format`, unknown voice id, malformed blend syntax.
- `429` — `max_concurrent` (default 4) in-flight requests exceeded. Back off and retry.

**Example**
```bash
curl -N -X POST http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello from docker","voice":"af_bella","response_format":"wav"}' \
  --output /tmp/out.wav
```

---

## parakeet-eou (port 3030)

Parakeet-EOU-120M streaming ASR with end-of-utterance detection, over
WebSocket. axum + `parakeet-rs` on ONNX Runtime. CPU-only in this image
(prebuilt CUDA ORT for aarch64 + sm_121 isn't available).

### `GET /health`

**200 OK**
```json
{
  "status": "ok",
  "model": "parakeet-eou-120m",
  "device": "cpu",
  "sessions": 0,
  "max_sessions": 2,
  "avg_latency_ms": 0.0,
  "model_load_time_ms": 1361.66
}
```

`avg_latency_ms` is a rolling mean of per-chunk inference time. `sessions` is
the current live WebSocket count; the server rejects new upgrades with
`503 Service Unavailable` once `max_sessions` is reached (default `2`).

### `GET /asr` (WebSocket)

Streaming ASR. Open a WebSocket and push audio frames; receive JSON text
events as utterances form.

**Accepted sample rate:** 16 kHz mono. The server does **not** resample —
payloads at other rates are rejected with an `error` event.

**Client → server frame types**

1. **Binary frame** — little-endian `f32` PCM samples at 16 kHz mono,
   no header. Each 4 bytes is one sample. Length must be a multiple of 4.

2. **JSON text frame** — base64-wrapped audio:
   ```json
   {
     "type": "audio",
     "data": "<base64 payload>",
     "sample_rate": 16000,
     "encoding": "f32le"
   }
   ```
   `encoding` accepts `f32le` (default) or `i16le`. `sample_rate` must be
   `16000`.

3. **End-of-stream** — tells the server to flush the decoder:
   ```json
   {"type": "eos"}
   ```

**Server → client events** — JSON text frames, discriminated by `type`:

| `type`    | Payload               | Meaning                                                |
| --------- | --------------------- | ------------------------------------------------------ |
| `partial` | `{"text": "..."}`     | Incremental transcript of the current utterance.       |
| `eou`     | `{"text": "..."}`     | End-of-utterance detected; finalized transcript.       |
| `error`   | `{"message": "..."}`  | Decoder/input error. Stream may continue or be closed. |

**Upgrade failures**
- `503 Service Unavailable` — `max_sessions` reached. Close and retry later.

**Example (Python, using the repo's test client)**
```bash
pip install websockets
python parakeet-eou-server/scripts/test_client.py some_16k.wav \
  --server ws://localhost:3030/asr
```

---

## Operational notes

**Compose lifecycle**
```bash
docker compose up -d                       # start both
docker compose ps                          # state
docker compose logs -f kokoro-tts          # tail one service
docker compose restart parakeet-eou        # bounce one service
docker compose down                        # stop + remove
```

**Resource model**
- `kokoro-tts` reserves all NVIDIA GPUs (`deploy.resources.reservations.devices`).
  Requires `nvidia-container-toolkit` on the host.
- `parakeet-eou` is CPU-only; no GPU request.

**Model weights**
- Both images are self-contained — weights are downloaded at `docker build`
  time, not at container start. Rebuild to refresh. First build of kokoro
  pulls ~3 GB (model + 54 voice packs); parakeet pulls ~481 MB of ONNX files.

**Concurrency limits**
- Kokoro: `max_concurrent_requests = 4` (returns `429` when full).
- Parakeet: `max_sessions = 2` (returns `503` on WS upgrade when full).
