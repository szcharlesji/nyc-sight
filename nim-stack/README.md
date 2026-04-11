# NIM Stack — DGX Spark (GB10, SM121, 128GB)

Four NVIDIA NIM containers running the Spark Sight AI pipeline on a single DGX Spark.

## Prerequisites

- **NVIDIA DGX Spark** (GB10, SM121, 128GB unified memory)
- **NGC Account** + API key ([sign up](https://org.ngc.nvidia.com/setup))
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Nemotron model weights** downloaded to `~/Downloads/models/llm/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
- `curl`, `python3`, `lsof` (for preflight checks and smoke tests)

## Quick Start

```bash
# 1. Set your NGC API key
export NGC_API_KEY=<your_key>

# 2. Launch all four NIMs sequentially
cd nim-stack
./start.sh

# 3. Verify
./healthcheck.sh
```

## Memory Budget

| Service | Type | Weights | KV Cache | Total |
|---------|------|---------|----------|-------|
| Nemotron-3-Nano-30B | LLM | ~27 GB | ~17 GB | **~44 GB** |
| Cosmos-Reason2-8B | VLM | ~36 GB | ~8 GB | **~44 GB** |
| Magpie TTS Multilingual | TTS | ~11 GB | — | **~11 GB** |
| Parakeet 1.1B RNNT | ASR | ~2-4 GB | — | **~4 GB** |
| **Total** | | | | **~103 GB / 128 GB** |

~25 GB headroom for CUDA context, scratch memory, and OS.

## Port Map

| Service | HTTP | gRPC | Health Endpoint |
|---------|------|------|-----------------|
| Nemotron (LLM) | 8005 | — | `GET /v1/models` |
| Cosmos (VLM) | 8000 | — | `GET /v1/models` |
| Magpie TTS | 9000 | 50051 | `GET /v1/health/ready` |
| Parakeet ASR | 9001 | 50052 | `GET /v1/health/ready` |

## Architecture

```
                    Sequential Launch Order
                    =======================

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Nemotron   │────▶│   Cosmos    │────▶│ Magpie TTS  │────▶│Parakeet ASR │
    │  (LLM)      │     │   (VLM)     │     │  (TTS)      │     │  (ASR)      │
    │  ~44GB GPU  │     │  ~44GB GPU  │     │  ~11GB GPU  │     │  ~4GB GPU   │
    │  port 8005  │     │  port 8000  │     │  port 9000  │     │  port 9001  │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │                    │
          └───────────────────┴───────────────────┴────────────────────┘
                                    │
                              nim-net (bridge)

    start.sh launches each container, waits for health, runs smoke test,
    then moves to the next. This prevents vLLM's GPU memory profiling
    race condition.
```

## GB10 / SM121 Workarounds

These hardware-specific settings are critical for correct operation:

| Setting | Service | Why |
|---------|---------|-----|
| `VLLM_USE_FLASHINFER_MOE_FP4=0` | Nemotron | flashinfer issue #2776 — MoE FP4 CUDA graph crash on GB10 |
| `NIM_MODEL_PROFILE=bf16` | Cosmos | FP8 quantization produces numerically incorrect output on SM121 |
| `NIM_MAX_MODEL_LEN=32768` | Nemotron | Default (131072) exceeds KV cache budget on shared 128GB GPU |
| `NIM_MAX_MODEL_LEN=16384` | Cosmos | Default (262144) needs 36GB KV cache alone — fatal OOM |
| `NIM_KVCACHE_PERCENT=0.35` | Both vLLM | Caps KV cache to ~35% of remaining memory; prevents one model from starving the other |
| Sequential startup | All | Prevents vLLM `AssertionError: free memory increased during profiling` |

## Troubleshooting

### `AssertionError: free memory increased during profiling`
**Cause:** Two vLLM containers started concurrently. One's scratch allocations interfere with the other's memory profiling baseline.
**Fix:** Always use `start.sh` for sequential startup. Never run `docker compose up` (all services at once).

### OOM / Killed by cgroup
**Cause:** Combined GPU memory exceeds 128GB.
**Fix:** Reduce `NIM_KVCACHE_PERCENT` or `NIM_MAX_MODEL_LEN` in `.env`. Check current usage with `nvidia-smi`.

### Cosmos outputs garbage text
**Cause:** Running FP8 quantized profile on SM121. FP8 is numerically broken on GB10.
**Fix:** Ensure `COSMOS_BF16_PROFILE` is set correctly in `.env`. The hash `3266ed3e...` forces bf16.

### Riva container takes 30+ minutes to start
**Cause:** First run downloads models and compiles TensorRT engines. This is expected behavior.
**Fix:** Be patient. Subsequent starts use cached engines and take <2 minutes. Do not delete `LOCAL_NIM_CACHE` unless necessary.

### Port already in use
**Cause:** Previous containers didn't clean up, or another service uses the port.
**Fix:** Run `./stop.sh` first, or check with `lsof -i :<port>`.

### CUDA graph crash on Nemotron
**Cause:** Missing `VLLM_USE_FLASHINFER_MOE_FP4=0` environment variable.
**Fix:** Verify the variable is set in `docker-compose.yml` (it is by default). See flashinfer issue #2776.

## Customization

### Adjusting KV Cache / Context Length

Edit `.env` and restart. The key trade-offs:

- **Higher `NIM_KVCACHE_PERCENT`** = longer context support, but less GPU memory for other services
- **Higher `NIM_MAX_MODEL_LEN`** = more tokens per request, but proportionally more KV cache needed
- **Rule of thumb**: `KV cache GB ≈ NIM_KVCACHE_PERCENT × (128GB - total_weights_GB)`

### Adding / Removing Services

1. Comment out or remove the service in `docker-compose.yml`
2. Remove it from the parallel arrays in `start.sh` (`SERVICES`, `CONTAINERS`, `HEALTH_URLS`, `TIMEOUTS`, `SMOKE_TYPES`)
3. Update port checks in `start.sh` preflight

### Using Different Models

For Nemotron: change `NEMOTRON_MODEL_SUBDIR` in `.env` to point to a different model directory under `NEMOTRON_MODEL_PATH`. Ensure the model is compatible with `model-free-nim`.

For Cosmos: change `COSMOS_IMAGE` and `COSMOS_BF16_PROFILE` in `.env`. The profile hash is model-specific.

## File Reference

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service definitions, volumes, network, logging |
| `.env` | All tunable configuration (ports, images, GPU budgets) |
| `start.sh` | Sequential orchestrator with health polling + smoke tests |
| `stop.sh` | Clean teardown (`--prune` for cache cleanup instructions) |
| `healthcheck.sh` | Point-in-time health probe (`--json` for machine output) |
