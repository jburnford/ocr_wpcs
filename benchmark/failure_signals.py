#!/usr/bin/env python3
"""Gold-free failure signals for out-of-spec ("impossible") OCR inputs.

A separate kind of measurement from CER/WER. When a model is given an image it
cannot reasonably transcribe — a letter photographed on a pile of other letters,
a blank or non-text page, a heavily occluded scan — there is no meaningful gold
to score against. What matters instead is *how the model fails*: does it stop /
say it cannot read (the helpful failure), or does it emit fluent garbage, loop,
or run away producing many times a page's worth of text (the harmful failures
that silently corrupt a downstream evidentiary chain)?

Every signal here is computed from the OCR output text ALONE — no gold. That has
a useful side effect: these results leak no gold and are safe to publish.

Empirically validated separation (Infinity Parser 2, manuscripts):
    clean pages        words 168-2052   gzip_ratio 1.7-2.9
    Monck "pile"       words 9024       gzip_ratio 8.3      -> runaway

Thresholds are deliberately conservative heuristics and are module-level
constants so they can be tuned / reported in the paper.
"""
from __future__ import annotations
import gzip
import re
from collections import Counter

# --- tunable thresholds (documented in the paper) ---------------------------
EMPTY_WORDS = 5          # below this, treat as empty / refusal
LOOP_TOP_NGRAM = 0.05    # one 3-gram is >5% of all 3-grams -> tight loop
RUNAWAY_GZIP = 4.0       # output compresses >4x -> repetitive / runaway garbage
RUNAWAY_BLOWUP = 3.0     # >3x the expected page word count -> overproduction
NGRAM_N = 3

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def output_words(text: str) -> int:
    """Number of word tokens emitted."""
    return len(_words(text))


def gzip_ratio(text: str) -> float:
    """raw_bytes / gzip_bytes. High = repetitive/low-entropy (loops, garbage)."""
    raw = text.encode("utf-8", "replace")
    if not raw:
        return 0.0
    comp = len(gzip.compress(raw, compresslevel=6))
    return round(len(raw) / comp, 2) if comp else 0.0


def lexical_repetition(text: str) -> float:
    """1 - unique_words/total_words. Note: naturally high (~0.6-0.75) for long
    clean English, so this is a weak discriminator on its own — reported, not
    used alone for labelling."""
    w = _words(text)
    return round(1 - len(set(w)) / len(w), 3) if w else 0.0


def top_ngram_share(text: str, n: int = NGRAM_N) -> float:
    """Share of the single most common word n-gram. Catches tight phrase loops
    (one phrase repeated over and over); near-zero for varied text."""
    w = _words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return round(Counter(grams).most_common(1)[0][1] / len(grams), 3)


def length_blowup(text: str, expected_words: float | None) -> float | None:
    """words / expected_words, when a rough page-capacity prior is available."""
    if not expected_words:
        return None
    return round(output_words(text) / expected_words, 2)


def failure_signals(text: str | None,
                    expected_words: float | None = None) -> dict:
    """All gold-free signals + a `failure_label` taxonomy for one OCR output.

    Labels (most-harmful first in priority):
      no-output       tool produced nothing (None) — crash / timeout
      empty/refusal   <EMPTY_WORDS words — stopped or said it can't read (helpful)
      loop            a single n-gram dominates — tight repetition
      runaway/garbage huge gzip ratio or many times a page's worth of text
      clean           none of the above triggered
    """
    if text is None:
        return {"failure_label": "no-output", "words": 0, "gzip_ratio": 0.0,
                "lexical_repetition": 0.0, "top_ngram_share": 0.0,
                "length_blowup": None}
    w = output_words(text)
    gz = gzip_ratio(text)
    rep = lexical_repetition(text)
    top = top_ngram_share(text)
    blow = length_blowup(text, expected_words)
    if w < EMPTY_WORDS:
        label = "empty/refusal"
    elif top >= LOOP_TOP_NGRAM:
        label = "loop"
    elif gz >= RUNAWAY_GZIP or (blow is not None and blow >= RUNAWAY_BLOWUP):
        label = "runaway/garbage"
    else:
        label = "clean"
    return {"failure_label": label, "words": w, "gzip_ratio": gz,
            "lexical_repetition": rep, "top_ngram_share": top,
            "length_blowup": blow}


GRACEFUL = {"empty/refusal", "no-output"}
HARMFUL = {"loop", "runaway/garbage"}
