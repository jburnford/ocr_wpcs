"""Locate a gold-standard article inside a full newspaper page's OCR text.

For the Sask article set the OCR covers a whole page; the gold standard is one
article on it. We find the best-matching contiguous span of the page OCR using
rapidfuzz partial-ratio alignment. If the best similarity is below a threshold
the article is reported NOT LOCATED (a failure tracked separately from CER/WER).
"""
from __future__ import annotations
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

DEFAULT_THRESHOLD = 0.60


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def locate_article(
    gold_text: str,
    page_ocr_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Find the gold article within page OCR.

    Returns dict: located (bool), score (0..1), located_text (best span or ''),
    method. score is rapidfuzz partial_ratio/100 on whitespace-normalized,
    lowercased text. located_text is the raw (un-normalized) OCR substring so
    downstream CER/WER scoring uses real characters.
    """
    g = _norm(gold_text)
    p_norm = _norm(page_ocr_text)
    if not g or not p_norm:
        return {"located": False, "score": 0.0, "located_text": "", "method": "empty"}

    # alignment on normalized text gives the matching window
    aln = fuzz.partial_ratio_alignment(g, p_norm)
    score = (aln.score if aln else 0.0) / 100.0

    if score < threshold or aln is None:
        return {"located": False, "score": round(score, 4),
                "located_text": "", "method": "partial_ratio"}

    # map the normalized-window back to a raw substring of page_ocr_text.
    # normalization only collapses whitespace + lowercases, so a proportional
    # offset map is approximate; pad generously, then refine by sliding.
    raw = page_ocr_text
    ratio = len(raw) / len(p_norm) if p_norm else 1.0
    glen = len(gold_text)
    pad = max(200, glen // 4)
    lo = max(0, int(aln.dest_start * ratio) - pad)
    hi = min(len(raw), int(aln.dest_end * ratio) + pad)
    window = raw[lo:hi]

    located = _best_window(gold_text, window)
    return {
        "located": True,
        "score": round(score, 4),
        "located_text": located,
        "method": "partial_ratio",
    }


def best_match_span(gold_text: str, ocr_text: str) -> dict:
    """Best-matching contiguous span of `ocr_text` for `gold_text`, with no
    threshold gate (unlike locate_article). Used to align a manuscript gold
    segment to its region of a multi-document OCR transcription so surrounding
    extra content is excluded. Returns {score, located_text}.
    """
    g = _norm(gold_text)
    p_norm = _norm(ocr_text)
    if not g or not p_norm:
        return {"score": 0.0, "located_text": ""}
    aln = fuzz.partial_ratio_alignment(g, p_norm)
    score = (aln.score if aln else 0.0) / 100.0
    if aln is None:
        return {"score": round(score, 4), "located_text": ocr_text}
    ratio = len(ocr_text) / len(p_norm) if p_norm else 1.0
    glen = len(gold_text)
    pad = max(200, glen // 4)
    lo = max(0, int(aln.dest_start * ratio) - pad)
    hi = min(len(ocr_text), int(aln.dest_end * ratio) + pad)
    return {"score": round(score, 4),
            "located_text": _best_window(gold_text, ocr_text[lo:hi])}


def _best_window(gold_text: str, candidate: str) -> str:
    """Return the substring of `candidate` that best matches `gold_text`.

    Tries window lengths around the gold length (to absorb OCR insertions/
    deletions); for each, a coarse slide then a fine step-1 refinement around
    the coarse minimum. Picks the global lowest normalized edit distance.
    """
    glen = len(gold_text)
    if len(candidate) <= glen:
        return candidate

    best_text = candidate[:glen]
    best_norm = Levenshtein.distance(gold_text, best_text) / max(glen, 1)

    for frac in (0.85, 1.0, 1.15):
        wlen = max(1, int(glen * frac))
        if wlen >= len(candidate):
            wlen = len(candidate)
        span = len(candidate) - wlen
        if span <= 0:
            cand = candidate
            nd = Levenshtein.distance(gold_text, cand) / max(glen, 1)
            if nd < best_norm:
                best_norm, best_text = nd, cand
            continue
        coarse = max(1, span // 80)
        best_start, best_d = 0, Levenshtein.distance(gold_text, candidate[:wlen])
        for start in range(0, span + 1, coarse):
            d = Levenshtein.distance(gold_text, candidate[start:start + wlen])
            if d < best_d:
                best_d, best_start = d, start
        for start in range(max(0, best_start - coarse),
                            min(span, best_start + coarse) + 1):
            d = Levenshtein.distance(gold_text, candidate[start:start + wlen])
            if d < best_d:
                best_d, best_start = d, start
        nd = best_d / max(glen, 1)
        if nd < best_norm:
            best_norm, best_text = nd, candidate[best_start:best_start + wlen]
    return best_text
