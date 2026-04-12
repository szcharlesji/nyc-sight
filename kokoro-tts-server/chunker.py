"""Text -> phoneme -> token chunking for Kokoro.

Splits input text at sentence boundaries, greedily merges sentences into
chunks whose token count (post-phonemization, pre-pad) stays within
``max_tokens`` (default 510 = 512 context minus 2 pad tokens).

Each emitted chunk carries:
    - ``token_ids``: [0] + merged_ids + [0] (bos/eos pads)
    - ``style_index``: the token-count index used to select the per-length
      style vector. Indexing matches KPipeline: ``style_index = len(merged_ids)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional


# Common abbreviations whose trailing '.' should not end a sentence.
_ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Jr.", "Sr.", "Prof.",
    "vs.", "etc.", "e.g.", "i.e.", "cf.", "Inc.", "Ltd.", "Co.", "Corp.",
    "No.", "Fig.", "Eq.", "Ref.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.",
    "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'\)\]]*)\s+(?=[A-Z0-9\"'\(\[])")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter respecting common abbreviations."""
    text = text.strip()
    if not text:
        return []

    # Protect abbreviations by replacing their '.' with a sentinel.
    sentinel = "\x00"
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", sentinel))

    parts = _SENTENCE_BOUNDARY.split(protected)
    parts = [p.replace(sentinel, ".").strip() for p in parts if p.strip()]
    return parts


@dataclass
class ChunkedInput:
    token_ids: List[int]      # already padded: [0] + merged + [0]
    style_index: int          # len(merged) — matches KPipeline indexing
    source_text: str          # concatenated source sentences for this chunk


class TextChunker:
    """Wraps a ``kokoro.KPipeline`` to expose phoneme-token chunking.

    The pipeline's G2P and vocab are reused directly; we do NOT run its
    generation loop. ``chunks()`` yields ``ChunkedInput`` objects ready for
    batched inference in ``backend.py``.
    """

    def __init__(self, pipeline, max_tokens: int = 510) -> None:
        self.pipeline = pipeline
        self.max_tokens = int(max_tokens)
        # Kokoro exposes its vocab as ``pipeline.model.vocab`` (dict[str, int]).
        vocab = getattr(getattr(pipeline, "model", None), "vocab", None)
        if not isinstance(vocab, dict):
            # Try common alternate locations in case the kokoro API moves.
            vocab = getattr(pipeline, "vocab", None)
        if not isinstance(vocab, dict):
            raise RuntimeError(
                "Could not locate phoneme->id vocab on KPipeline "
                "(checked pipeline.model.vocab and pipeline.vocab)."
            )
        self._vocab: dict = vocab

    # -------- public API ----------------------------------------------------

    def chunks(self, text: str, lang_code: Optional[str] = None) -> Iterator[ChunkedInput]:
        sentences = split_sentences(text)
        if not sentences:
            return iter(())
        return self._merge(sentences, lang_code)

    def phonemize(self, text: str, lang_code: Optional[str] = None) -> str:
        """Best-effort phonemization using whichever G2P entrypoint kokoro exposes."""
        g2p = getattr(self.pipeline, "g2p", None)
        if callable(g2p):
            out = g2p(text)
            # Misaki's G2P returns (phoneme_str, tokens) or a plain str.
            if isinstance(out, tuple):
                return out[0]
            return out
        # Fallback: kokoro>=0.9 pipelines support __call__ with mode='phonemes'.
        raise RuntimeError("KPipeline has no .g2p(text); cannot phonemize")

    def tokenize(self, phonemes: str) -> List[int]:
        """Map a phoneme string to token IDs, dropping unknown characters."""
        return [self._vocab[ch] for ch in phonemes if ch in self._vocab]

    # -------- internals -----------------------------------------------------

    def _merge(self, sentences: List[str], lang_code: Optional[str]) -> Iterator[ChunkedInput]:
        buf_ids: List[int] = []
        buf_text: List[str] = []

        for sentence in sentences:
            phones = self.phonemize(sentence, lang_code)
            ids = self.tokenize(phones)
            if not ids:
                continue
            if len(ids) > self.max_tokens:
                # Hard-truncate a single oversized sentence so we never emit
                # something the model can't eat. Rare in practice.
                ids = ids[: self.max_tokens]

            if buf_ids and len(buf_ids) + len(ids) > self.max_tokens:
                yield self._flush(buf_ids, buf_text)
                buf_ids, buf_text = [], []

            buf_ids.extend(ids)
            buf_text.append(sentence)

        if buf_ids:
            yield self._flush(buf_ids, buf_text)

    @staticmethod
    def _flush(ids: List[int], texts: List[str]) -> ChunkedInput:
        return ChunkedInput(
            token_ids=[0] + ids + [0],
            style_index=len(ids),
            source_text=" ".join(texts),
        )
