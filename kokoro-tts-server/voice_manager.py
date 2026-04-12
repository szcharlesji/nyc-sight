"""Voice pack loading, blending, and caching.

Each Kokoro voice pack is a torch tensor of shape (511, 1, 256): element [n] is
the style vector for an input of (n + 1) tokens. Downstream inference selects
``pack[token_count - 1]`` (or similar) so blends must preserve the full first
dimension.

Blend spec syntax:

    af_bella                    single voice
    af_bella+af_sky             two voices, equal weight
    af_bella(2)+af_sky(1)       weighted blend (normalized to sum = 1)
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import torch


log = logging.getLogger(__name__)

_COMPONENT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)(?:\(([0-9]*\.?[0-9]+)\))?$")

# Voice pack filenames shipped with hexgrad/Kokoro-82M. Mirrors the list used
# in scripts/download_model.sh; kept here so list_voices() works even when
# the local voices/ directory hasn't been populated yet.
_KNOWN_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
)


class VoiceNotFoundError(ValueError):
    pass


class VoiceBlendSyntaxError(ValueError):
    pass


class VoiceManager:
    def __init__(
        self,
        voices_dir: Path,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        hf_repo: Optional[str] = "hexgrad/Kokoro-82M",
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.hf_repo = hf_repo
        self._raw_cache: dict[str, torch.Tensor] = {}
        self._blend_cache: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

        # Create the directory so lazy HF downloads have somewhere to land.
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # -------- public API ----------------------------------------------------

    def list_voices(self) -> List[str]:
        local = sorted(p.stem for p in self.voices_dir.glob("*.pt"))
        if local:
            return local
        # Fall back to the known catalog so clients can enumerate before any
        # voice has been materialized locally.
        return list(_KNOWN_VOICES)

    def get(self, spec: str) -> torch.Tensor:
        """Resolve a blend spec to a (511, 1, 256) style tensor on ``self.device``."""
        spec = (spec or "").strip()
        if not spec:
            raise VoiceBlendSyntaxError("voice spec is empty")

        components = self._parse_blend_spec(spec)
        key = self._canonicalize(components)
        with self._lock:
            if key in self._blend_cache:
                return self._blend_cache[key]
            blended = self._blend(components)
            self._blend_cache[key] = blended
            return blended

    # -------- internals -----------------------------------------------------

    def _parse_blend_spec(self, spec: str) -> List[Tuple[str, float]]:
        parts = [p.strip() for p in spec.split("+") if p.strip()]
        if not parts:
            raise VoiceBlendSyntaxError(f"empty blend spec: {spec!r}")
        out: List[Tuple[str, float]] = []
        for part in parts:
            m = _COMPONENT_RE.match(part)
            if not m:
                raise VoiceBlendSyntaxError(
                    f"cannot parse voice component {part!r} in spec {spec!r}"
                )
            name = m.group(1)
            w = float(m.group(2)) if m.group(2) is not None else 1.0
            if w < 0:
                raise VoiceBlendSyntaxError(f"negative weight for {name!r}")
            out.append((name, w))
        total = sum(w for _, w in out)
        if total <= 0:
            raise VoiceBlendSyntaxError(f"blend weights sum to zero: {spec!r}")
        return [(n, w / total) for n, w in out]

    @staticmethod
    def _canonicalize(components: List[Tuple[str, float]]) -> str:
        items = sorted(components, key=lambda x: x[0])
        return "+".join(f"{n}({w:.4f})" for n, w in items)

    def _load_raw(self, name: str) -> torch.Tensor:
        if name in self._raw_cache:
            return self._raw_cache[name]
        path = self.voices_dir / f"{name}.pt"
        if not path.exists():
            path = self._fetch_from_hf(name)
        t = torch.load(path, map_location=self.device, weights_only=True)
        if not torch.is_tensor(t):
            raise ValueError(f"voice pack {name!r} is not a tensor (got {type(t)})")
        t = t.to(device=self.device, dtype=self.dtype)
        self._raw_cache[name] = t
        return t

    def _fetch_from_hf(self, name: str) -> Path:
        """Download a single voice pack from the configured HF repo into voices_dir."""
        if self.hf_repo is None:
            raise VoiceNotFoundError(
                f"voice not found locally: {name} (expected {self.voices_dir / f'{name}.pt'}); "
                "HF fallback disabled"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise VoiceNotFoundError(
                f"voice {name!r} not present locally and huggingface_hub is unavailable"
            ) from exc

        log.info("fetching voice %r from %s", name, self.hf_repo)
        try:
            downloaded = hf_hub_download(
                repo_id=self.hf_repo,
                filename=f"voices/{name}.pt",
                local_dir=str(self.voices_dir.parent),
            )
        except Exception as exc:
            raise VoiceNotFoundError(
                f"voice {name!r} not found locally ({self.voices_dir / f'{name}.pt'}) "
                f"and HF fetch failed: {exc}"
            ) from exc
        return Path(downloaded)

    def _blend(self, components: List[Tuple[str, float]]) -> torch.Tensor:
        if len(components) == 1 and components[0][1] == 1.0:
            return self._load_raw(components[0][0]).clone()
        stacked = None
        for name, weight in components:
            t = self._load_raw(name)
            contrib = t * weight
            stacked = contrib if stacked is None else stacked + contrib
        assert stacked is not None
        return stacked
