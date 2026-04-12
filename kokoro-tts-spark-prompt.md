# Claude Code Prompt: Kokoro-82M TTS Server Optimized for DGX Spark (GB10)

## Objective

Build a high-performance, low-latency TTS inference server using Kokoro-82M on an NVIDIA DGX Spark (GB10 Grace Blackwell Superchip). The server must expose an OpenAI-compatible `/v1/audio/speech` endpoint with streaming PCM output, optimized for sub-100ms time-to-first-audio on single sentences. This will be used as the TTS component in a real-time voice agent pipeline co-located with an LLM on the same device.

## Hardware Context

- **Device**: NVIDIA DGX Spark
- **GPU**: GB10 Grace Blackwell, compute capability sm_121, CUDA 13.0/13.1
- **CPU**: 72-core Grace ARM (aarch64)
- **Memory**: 128GB unified (CPU+GPU coherent)
- **OS**: Ubuntu 24.04 aarch64
- **Key constraint**: No prebuilt ONNX Runtime GPU wheels for aarch64 + CUDA 13 exist. PyTorch CUDA is the primary inference path. If ONNX GPU is desired, ORT 1.24.4+ must be built from source targeting sm_121 (see https://github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell for reference).

## Model Details

- **Model**: Kokoro-82M (v1.0) — https://huggingface.co/hexgrad/Kokoro-82M
- **Architecture**: StyleTTS2 + ISTFTNet decoder-only (no encoder, no diffusion)
- **Parameters**: 82M (~350MB FP32, <1GB FP16)
- **GPU VRAM at inference**: ~2-3GB including CUDA kernels/buffers
- **Sample rate**: 24000 Hz, mono, 16-bit PCM
- **Context length**: 512 tokens max per chunk
- **Phonemizer**: Misaki (G2P) — requires `espeak-ng` system dependency
- **Inference library**: `pip install kokoro>=0.9.4` (official), or raw PyTorch via `hexgrad/Kokoro-82M` weights
- **Voices**: 54 voices across 8 languages, stored as `.pt` style vectors in `voices/` directory
- **License**: Apache 2.0

## Architecture Requirements

### 1. Inference Backend (`backend.py`)

Build the core inference engine with these optimizations:

**Model Loading & Warmup:**
- Load Kokoro-82M weights to GPU on startup via PyTorch
- Pre-load all voice style vectors to GPU (they're small tensors, ~256-dim each)
- Run a warmup inference pass on a dummy sentence to JIT-compile CUDA kernels
- Use `torch.inference_mode()` context for all generation
- Investigate `torch.compile()` with the `inductor` backend for the forward pass — profile whether it helps on sm_121 (Blackwell may have different compilation characteristics than Ampere/Hopper)

**Sentence-Level Chunking & Streaming:**
- Kokoro's context window is 512 tokens. Long text must be split at sentence boundaries.
- Implement a chunking pipeline:
  1. Text → phonemes via Misaki G2P (`from kokoro import KPipeline`)
  2. Phonemes → token IDs using the vocab from `config.json`
  3. Greedily merge sentences into chunks of ≤510 tokens (leaving room for pad tokens at start/end)
  4. For each chunk: `[0] + tokens + [0]` padding, select style vector based on `len(tokens)`, run inference
- Stream audio chunk-by-chunk: yield the first chunk's audio as soon as it's generated, don't wait for all chunks

**Batched Inference (Critical Optimization):**
- The standard Kokoro model only supports batch_size=1 because the alignment matrix construction uses `torch.interleave` which doesn't vectorize across batches.
- Implement NimbleEdge's mask-based batched alignment fix (see https://github.com/NimbleEdge/kokoro):
  ```python
  # Replace torch.interleave-based alignment with:
  frame_indices = torch.arange(max_frames, device=self.device).view(1, 1, -1)
  duration_cumsum = duration.cumsum(dim=1).unsqueeze(-1)
  mask1 = duration_cumsum > frame_indices
  mask2 = frame_indices >= torch.cat(
      [torch.zeros(duration.shape[0], 1, 1), duration_cumsum[:, :-1, :]], dim=1
  )
  pred_aln_trg = (mask1 & mask2).float().transpose(1, 2)
  ```
- This enables parallel generation of multiple sentences in a single forward pass — crucial for reducing total latency on multi-sentence inputs
- Requires padding input_ids to the same length within the batch and masking appropriately

**Half Precision:**
- Run model in FP16 by default (`model.half()`) — Kokoro is stable in FP16 and the GB10 has excellent FP16 throughput
- Keep voice style vectors in FP32 (they're tiny and precision matters for prosody)

### 2. Server (`server.py`)

**FastAPI with Streaming Response:**
```python
@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    # Returns StreamingResponse with chunked PCM/WAV/MP3
    ...
```

**OpenAI-Compatible API Contract:**
- Accept: `model` (ignored, always kokoro), `input` (text), `voice` (voice ID like `af_bella`), `response_format` (pcm/wav/mp3/opus/flac), `speed` (float, default 1.0)
- Support voice blending syntax: `af_bella+af_sky` for 50/50 mix, `af_bella(2)+af_sky(1)` for weighted blend
- Return: chunked transfer-encoded audio stream

**Streaming Protocol:**
- For `response_format=pcm`: stream raw 16-bit signed LE PCM at 24kHz as each sentence-chunk completes
- For `response_format=wav`: buffer complete audio, return with WAV header (non-streaming)
- For `response_format=mp3`/`opus`/`flac`: encode on the fly per chunk, stream

**Concurrency:**
- Use an asyncio lock around GPU inference to prevent OOM from concurrent requests (single-user device)
- Queue incoming requests if the GPU is busy with LLM inference (the voice agent pipeline will manage this externally, but the server should be robust)

### 3. Benchmarking Suite (`benchmark.py`)

Build a comprehensive profiling script that measures:
- **Time-to-first-audio (TTFA)**: from request to first audio byte
- **Total generation time**: for inputs of 5, 10, 25, 50, 100, 200 words
- **Real-time factor (RTF)**: generation_time / audio_duration (lower is better, <0.1 means >10x realtime)
- **GPU memory usage**: peak VRAM during inference
- **Throughput**: characters per second, tokens per second
- Profile with `torch.cuda.Event` timers (not wall clock) for accurate GPU timing
- Compare: single inference vs batched inference, FP32 vs FP16, with/without `torch.compile()`
- Output results as a formatted table and JSON

### 4. Integration Interface (`client.py`)

Provide a client that demonstrates:
- Streaming PCM to speakers in real-time via PyAudio
- Integration with an LLM token stream: accept text from an async generator (simulating LLM output), buffer until sentence boundary, dispatch TTS, play audio — demonstrating the interleaved LLM+TTS pattern from the Daily.co/Pipecat voice agent architecture
- Measure end-to-end voice-to-voice latency simulation

## Project Structure

```
kokoro-spark-tts/
├── server.py              # FastAPI server with OpenAI-compatible endpoint
├── backend.py             # Core Kokoro inference engine with optimizations
├── chunker.py             # Text → phoneme → token chunking pipeline
├── benchmark.py           # Profiling & latency measurement suite
├── client.py              # Streaming client + LLM interleave demo
├── voice_manager.py       # Voice loading, blending, caching
├── audio_encoder.py       # PCM → WAV/MP3/Opus encoding
├── Dockerfile             # For DGX Spark (aarch64, CUDA 13.1, PyTorch)
├── requirements.txt
├── config.py              # All tunables (chunk size, batch size, precision, etc.)
└── README.md              # Setup, benchmarks, architecture docs
```

## Key Technical Decisions to Make (Profile and Decide)

1. **PyTorch eager vs `torch.compile()`**: The Blackwell sm_121 architecture is new — `torch.compile(inductor)` may or may not have good kernel support. Profile both paths and default to whichever is faster.

2. **FP16 vs FP32 vs BF16**: GB10 supports BF16 natively. Test if BF16 gives better throughput than FP16 without quality loss.

3. **Batch size sweet spot**: With 128GB unified memory and a tiny model, batch sizes of 4-8 sentences should be feasible. Find the batch size where latency stops improving due to memory bandwidth saturation.

4. **CUDA graphs**: For fixed-shape inputs (padded to max chunk length), CUDA graph capture could eliminate kernel launch overhead. Investigate whether Kokoro's forward pass is graph-capturable.

5. **Grace CPU fallback path**: Optionally implement a CPU inference path using ONNX Runtime on the Grace's 72 ARM cores. This allows TTS to run on CPU while the GPU handles LLM inference exclusively — the "split compute" strategy. Compare latency: GPU ~50ms vs CPU ~200-400ms per sentence.

## Dockerfile Skeleton

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3
# This NGC container includes PyTorch with CUDA for aarch64 Blackwell

RUN apt-get update && apt-get install -y espeak-ng && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights and voices on build
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

EXPOSE 8880
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
```

## Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| TTFA (single sentence) | <100ms | <50ms |
| RTF (GPU, FP16) | <0.05 | <0.02 |
| RTF (CPU, ONNX) | <0.5 | <0.3 |
| Peak VRAM | <3GB | <2GB |
| Concurrent with 8B LLM | Yes | Yes, with interleaved scheduling |

## Reference Implementations to Study

- **Kokoro-FastAPI** (remsky): https://github.com/remsky/Kokoro-FastAPI — production-ready FastAPI wrapper, ARM multi-arch Docker images, smart chunking with sentence boundary detection. Study their `audio_service.py` and `pytorch_backend.py`.
- **NimbleEdge batched Kokoro**: https://github.com/NimbleEdge/kokoro — mask-based batched alignment matrix, ONNX export with batch support. Critical for the batched inference optimization.
- **Official Kokoro library**: https://github.com/hexgrad/kokoro — `KPipeline` API, phonemizer, voice pack loading.
- **Daily.co voice agent**: https://github.com/pipecat-ai/nemotron-january-2026/ — interleaved LLM+TTS inference pattern on DGX Spark, custom WebSocket TTS server, chunked streaming.
- **ONNX Runtime on Spark**: https://github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell — if pursuing ONNX GPU path, prebuilt ORT 1.24.4 binaries for sm_121.

## Non-Goals (For Now)

- Voice cloning / zero-shot speaker adaptation
- Multi-language in a single request
- WebSocket transport (HTTP streaming is sufficient)
- TensorRT conversion (Kokoro's dynamic shapes make this complex; revisit later)
- Training or fine-tuning

## Success Criteria

1. Server starts, loads model, and serves `/v1/audio/speech` with streaming PCM output
2. Benchmark suite runs and produces latency/RTF numbers for the GB10
3. Single-sentence TTFA is under 100ms on GPU
4. Batched inference demonstrably faster than sequential for multi-sentence inputs
5. Client demo shows real-time audio playback from streaming TTS
6. Docker container builds and runs on DGX Spark without modification
