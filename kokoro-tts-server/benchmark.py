"""Benchmark suite for the Kokoro-82M TTS backend.

Measures time-to-first-audio (TTFA), total generation time, real-time factor
(RTF), peak VRAM, and throughput. Supports sweeping across input sizes,
precisions (FP32 / FP16 / BF16), batched vs. sequential, and with/without
``torch.compile``.

Two modes:

    # Direct backend (imports KokoroBackend in-process)
    python benchmark.py --full --output benchmark_results.json

    # Through the HTTP stack (measures the full server)
    python benchmark.py --client-mode http://localhost:8880 --output client_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from config import SAMPLE_RATE, ServerConfig


log = logging.getLogger("benchmark")


SAMPLE_TEXTS = {
    5: "The quick brown fox jumps today.",
    10: "The quick brown fox jumps over the lazy dog every morning.",
    25: (
        "The quick brown fox jumps over the lazy dog every morning at sunrise "
        "while the farmer tends his field and the birds sing in the trees above."
    ),
    50: (
        "The quick brown fox jumps over the lazy dog every morning at sunrise. "
        "The farmer tends his field and the birds sing in the trees above. "
        "Mist hangs over the river as children walk to school, laughing and "
        "trading stories about the weekend just past."
    ),
    100: (
        "The quick brown fox jumps over the lazy dog every morning at sunrise. "
        "The farmer tends his field and the birds sing in the trees above. "
        "Mist hangs over the river as children walk to school, laughing and "
        "trading stories about the weekend just past. In the market square a "
        "vendor arranges baskets of apples and pears while a cat naps on a "
        "sunny stone. The baker slides loaves from the oven, the scent of "
        "bread drifting down the cobbled street, and shopkeepers raise their "
        "shutters as the day begins."
    ),
    200: None,  # built below
}
SAMPLE_TEXTS[200] = SAMPLE_TEXTS[100] + " " + SAMPLE_TEXTS[100]


@dataclass
class BenchResult:
    label: str
    words: int
    dtype: str
    batch_size: int
    compiled: bool
    ttfa_s: float
    total_s: float
    audio_s: float
    rtf: float
    peak_vram_mb: float
    iters: int = 1
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Direct-backend benchmarking
# ---------------------------------------------------------------------------


async def _run_direct(backend, text: str, voice: str) -> tuple[float, float, float]:
    """Returns (ttfa_s, total_s, audio_samples)."""
    t0 = time.perf_counter()
    ttfa: Optional[float] = None
    total_samples = 0
    async for pcm in backend.generate(text, voice, speed=1.0):
        if ttfa is None:
            ttfa = time.perf_counter() - t0
        total_samples += int(pcm.shape[0])
    t_end = time.perf_counter()
    if ttfa is None:
        ttfa = t_end - t0
    return ttfa, t_end - t0, total_samples


def _bench_direct(
    backend,
    text: str,
    voice: str,
    label: str,
    iters: int,
    warmups: int,
) -> BenchResult:
    import torch

    words = len(text.split())
    loop = asyncio.new_event_loop()
    try:
        for _ in range(warmups):
            loop.run_until_complete(_run_direct(backend, text, voice))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ttfas, totals, samples_list = [], [], []
        for _ in range(iters):
            ttfa, total, samples = loop.run_until_complete(_run_direct(backend, text, voice))
            ttfas.append(ttfa)
            totals.append(total)
            samples_list.append(samples)
    finally:
        loop.close()

    ttfa = statistics.median(ttfas)
    total = statistics.median(totals)
    samples = int(statistics.median(samples_list))
    audio_s = samples / SAMPLE_RATE
    rtf = total / audio_s if audio_s > 0 else float("nan")
    vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    return BenchResult(
        label=label,
        words=words,
        dtype=backend.cfg.dtype,
        batch_size=backend.cfg.batch_size,
        compiled=backend.cfg.enable_torch_compile,
        ttfa_s=ttfa,
        total_s=total,
        audio_s=audio_s,
        rtf=rtf,
        peak_vram_mb=vram,
        iters=iters,
    )


def sweep_sizes(backend, voice: str, iters: int, warmups: int) -> List[BenchResult]:
    out: List[BenchResult] = []
    for n_words in sorted(SAMPLE_TEXTS.keys()):
        text = SAMPLE_TEXTS[n_words]
        out.append(_bench_direct(backend, text, voice, f"size_{n_words}w", iters, warmups))
    return out


def compare_precision(base_cfg: ServerConfig, voice: str, iters: int, warmups: int) -> List[BenchResult]:
    from backend import KokoroBackend

    out: List[BenchResult] = []
    for dtype in ("fp32", "fp16", "bf16"):
        cfg = base_cfg.with_overrides(dtype=dtype)
        log.info("benchmark precision: %s", dtype)
        backend = KokoroBackend(cfg)
        out.append(_bench_direct(backend, SAMPLE_TEXTS[25], voice, f"dtype_{dtype}", iters, warmups))
    return out


def compare_batched(base_cfg: ServerConfig, voice: str, iters: int, warmups: int) -> List[BenchResult]:
    from backend import KokoroBackend

    text = SAMPLE_TEXTS[100]
    out: List[BenchResult] = []
    for bs in (1, 2, 4, 8):
        cfg = base_cfg.with_overrides(batch_size=bs, first_batch_size=min(1, bs))
        log.info("benchmark batch_size: %d", bs)
        backend = KokoroBackend(cfg)
        out.append(_bench_direct(backend, text, voice, f"batch_{bs}", iters, warmups))
    return out


def compare_compile(base_cfg: ServerConfig, voice: str, iters: int, warmups: int) -> List[BenchResult]:
    from backend import KokoroBackend

    out: List[BenchResult] = []
    for enabled in (False, True):
        cfg = base_cfg.with_overrides(enable_torch_compile=enabled)
        log.info("benchmark compile=%s", enabled)
        backend = KokoroBackend(cfg)
        out.append(_bench_direct(backend, SAMPLE_TEXTS[25], voice, f"compile_{enabled}", iters, warmups))
    return out


# ---------------------------------------------------------------------------
# Client-mode benchmarking (HTTP)
# ---------------------------------------------------------------------------


async def _bench_client(url: str, voice: str, iters: int) -> List[BenchResult]:
    import httpx

    out: List[BenchResult] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for n_words, text in SAMPLE_TEXTS.items():
            ttfas, totals, sizes = [], [], []
            for _ in range(iters):
                t0 = time.perf_counter()
                ttfa: Optional[float] = None
                size = 0
                async with client.stream(
                    "POST",
                    f"{url.rstrip('/')}/v1/audio/speech",
                    json={
                        "input": text,
                        "voice": voice,
                        "response_format": "pcm",
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if ttfa is None and chunk:
                            ttfa = time.perf_counter() - t0
                        size += len(chunk)
                total = time.perf_counter() - t0
                ttfas.append(ttfa or total)
                totals.append(total)
                sizes.append(size)
            ttfa_m = statistics.median(ttfas)
            total_m = statistics.median(totals)
            size_m = int(statistics.median(sizes))
            audio_s = (size_m // 2) / SAMPLE_RATE  # s16 mono
            out.append(
                BenchResult(
                    label=f"client_{n_words}w",
                    words=n_words,
                    dtype="n/a",
                    batch_size=-1,
                    compiled=False,
                    ttfa_s=ttfa_m,
                    total_s=total_m,
                    audio_s=audio_s,
                    rtf=total_m / audio_s if audio_s > 0 else float("nan"),
                    peak_vram_mb=0.0,
                    iters=iters,
                    extra={"bytes": size_m},
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: Iterable[BenchResult]) -> None:
    rows = list(results)
    if not rows:
        print("(no results)")
        return
    headers = ["label", "words", "dtype", "bs", "cmp", "ttfa_ms", "total_ms", "audio_s", "rtf", "vram_mb"]
    widths = [max(len(h), 10) for h in headers]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [
            r.label,
            str(r.words),
            r.dtype,
            str(r.batch_size),
            "Y" if r.compiled else "N",
            f"{r.ttfa_s*1000:.1f}",
            f"{r.total_s*1000:.1f}",
            f"{r.audio_s:.2f}",
            f"{r.rtf:.3f}",
            f"{r.peak_vram_mb:.0f}",
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))


def write_json(results: Iterable[BenchResult], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser("kokoro-tts-benchmark")
    parser.add_argument("--model-dir", type=Path, default=Path("./models/kokoro-82m"))
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))

    parser.add_argument("--sizes", action="store_true", help="Sweep word counts.")
    parser.add_argument("--precision", action="store_true", help="FP32 vs FP16 vs BF16.")
    parser.add_argument("--batch", action="store_true", help="batch_size in {1,2,4,8}.")
    parser.add_argument("--compile", action="store_true", help="torch.compile on/off.")
    parser.add_argument("--full", action="store_true", help="Run all suites.")
    parser.add_argument("--client-mode", dest="client_mode", default=None,
                        help="HTTP URL; runs through the server instead of importing backend.")

    ns = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")

    results: List[BenchResult] = []

    if ns.client_mode:
        results = asyncio.run(_bench_client(ns.client_mode, ns.voice, ns.iters))
    else:
        from backend import KokoroBackend

        base_cfg = ServerConfig(model_dir=ns.model_dir)

        if ns.full or ns.sizes:
            backend = KokoroBackend(base_cfg)
            results.extend(sweep_sizes(backend, ns.voice, ns.iters, ns.warmups))
        if ns.full or ns.precision:
            results.extend(compare_precision(base_cfg, ns.voice, ns.iters, ns.warmups))
        if ns.full or ns.batch:
            results.extend(compare_batched(base_cfg, ns.voice, ns.iters, ns.warmups))
        if ns.full or ns.compile:
            results.extend(compare_compile(base_cfg, ns.voice, ns.iters, ns.warmups))
        if not results:
            backend = KokoroBackend(base_cfg)
            results = sweep_sizes(backend, ns.voice, ns.iters, ns.warmups)

    print_table(results)
    write_json(results, ns.output)


if __name__ == "__main__":
    main()
