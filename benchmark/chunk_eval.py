#!/usr/bin/env python3
"""Order-invariant, chunk-aligned OCR scoring for full pages.

Motivation: corpus CER/WER over a linearized page conflates three things —
character recognition, reading-order serialization, and what-counts-as-the-page
(ads in/out). For multi-column newspapers aimed at downstream text mining /
information extraction, what matters is whether each *coherent chunk* (article
or paragraph) comes out accurately and intact, regardless of the order the tool
emitted the chunks in.

Approach: segment only the GOLD into chunks (the tool output stays whole), then
for each gold chunk find its best-matching contiguous span anywhere in the tool
output (rapidfuzz). Score CER/WER within the matched span. A gold chunk whose
best match is below threshold is "not recovered" (dropped/garbled content).
Tool text never claimed by any gold chunk is reported as "extra" (ads,
mastheads, noise) but does NOT count against CER — per the gold's scope, a tool
should not be punished for reading more than the transcriber chose to record.

Two granularities are reported: `paragraph` (blank-line units, reliable on every
gold) and `article` (separator/column/header units — approximate where the gold
lacks layout markers).
"""
from __future__ import annotations
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import gold_loaders as gl
from locate_article import _best_window
from ocr_metrics import bleu_score, cer, normalize_text, wer

LOCATE_THRESHOLD = 0.55           # min partial-ratio similarity to call a chunk recovered

_BRACKET = re.compile(r"\[[^\]]*\]")
_AD = re.compile(r"\[\s*(ad\b|advertisement|masthead)", re.I)
_COL = re.compile(r"\[\s*(left|right|second|centre|center|first|third|fourth|"
                  r"fifth|sixth)[^\]]*column", re.I)
_SEP = re.compile(r"\[\s*line for separation", re.I)
_IMG = re.compile(r"\[[^\]]*(image|cartoon|portrait|photo|illustration|figure)"
                  r"[^\]]*\]", re.I)


# ----------------------------------------------------------------- gold ------
def _gold_body(md_name: str) -> str:
    raw = (gl.FULLPAGE_REVIEW / md_name).read_text(encoding="utf-8",
                                                   errors="replace")
    m = re.search(r"##\s*Page Content\s*\n", raw)
    return raw[m.end():] if m else raw


def _looks_like_header(line: str) -> bool:
    """Heuristic newspaper headline: short, capitalized/all-caps, no end period."""
    s = line.strip().rstrip(".")
    words = [w for w in re.findall(r"[A-Za-z]+", s)]
    if not words or len(words) > 8 or len(s) > 64:
        return False
    if s.isupper():
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words) >= 0.8


def _clean(line: str) -> str:
    return _BRACKET.sub(" ", line).strip()


def segment_gold(md_name: str, level: str) -> tuple[list[str], int]:
    """Segment a full-page gold into scored chunks.

    Returns (chunks, ad_blocks). `chunks` are coherent gold units (article or
    paragraph). `ad_blocks` counts ad/masthead markers seen (context only).
    Bracket-only marker lines drive boundaries but are never scored content.
    """
    blocks = re.split(r"\n\s*\n", _gold_body(md_name))
    paras: list[str] = []          # (text, starts_article) tuples flattened below
    starts: list[bool] = []
    ad_blocks = 0
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        boundary = False
        kept: list[str] = []
        for ln in lines:
            s = ln.strip()
            if s.startswith("#"):
                continue
            if _BRACKET.fullmatch(s):                  # whole line is a marker
                if _AD.search(s):
                    ad_blocks += 1
                if _AD.search(s) or _COL.search(s) or _SEP.search(s) or _IMG.search(s):
                    boundary = True
                continue
            c = _clean(ln)
            if c:
                if not kept and _looks_like_header(c):
                    boundary = True
                kept.append(c)
        if not kept:
            continue
        paras.append(" ".join(kept))
        starts.append(boundary)

    if level == "paragraph":
        return [p for p in paras if p.strip()], ad_blocks

    # article: merge paragraphs until the next boundary
    articles: list[str] = []
    cur: list[str] = []
    for para, starts_art in zip(paras, starts):
        if starts_art and cur:
            articles.append(" ".join(cur))
            cur = []
        cur.append(para)
    if cur:
        articles.append(" ".join(cur))
    return [a for a in articles if a.strip()], ad_blocks


# ------------------------------------------------------------- matching ------
def _best_span(gold_chunk: str, hyp: str) -> tuple[float, str]:
    """Best-matching contiguous span of `hyp` for `gold_chunk`, order-free.

    Similarity is partial_ratio on normalized text; the returned span is the raw
    hyp substring (refined to gold length) for downstream CER/WER.
    """
    g = re.sub(r"\s+", " ", gold_chunk.lower()).strip()
    h = re.sub(r"\s+", " ", hyp.lower()).strip()
    if not g or not h:
        return 0.0, ""
    aln = fuzz.partial_ratio_alignment(g, h)
    score = (aln.score if aln else 0.0) / 100.0
    if aln is None:
        return score, ""
    ratio = len(hyp) / len(h) if h else 1.0
    pad = max(200, len(gold_chunk) // 4)
    lo = max(0, int(aln.dest_start * ratio) - pad)
    hi = min(len(hyp), int(aln.dest_end * ratio) + pad)
    return score, _best_window(gold_chunk, hyp[lo:hi])


def score_chunks(gold_chunks: list[str], hyp: str) -> dict:
    """Order-invariant scoring of one tool's page output against gold chunks."""
    per = []
    tot_gold_chars = tot_edits = 0
    tot_gold_words = tot_wedits = 0
    located_chars = matched_hyp_chars = 0
    for ch in gold_chunks:
        score, span = _best_span(ch, hyp)
        g_sem = normalize_text(ch, semantic=True)
        located = score >= LOCATE_THRESHOLD and bool(span)
        rec = {"gold_preview": ch[:60], "score": round(score, 4),
               "located": located, "gold_chars": len(g_sem)}
        if located:
            o_sem = normalize_text(span, semantic=True)
            ce, ne, n = cer(g_sem, o_sem)
            we, we_e, wn = wer(g_sem, o_sem)
            rec.update(cer=ce, wer=we, coverage=round(min(1.0, len(o_sem) /
                       max(1, len(g_sem))), 3), bleu=round(bleu_score(g_sem, o_sem), 3))
            tot_edits += ne; tot_gold_chars += n
            tot_wedits += we_e; tot_gold_words += wn
            located_chars += len(g_sem); matched_hyp_chars += len(o_sem)
        else:
            tot_gold_chars += len(g_sem)        # unrecovered content = full miss
            tot_gold_words += len(g_sem.split())
        per.append(rec)

    n_chunks = len(gold_chunks)
    n_loc = sum(1 for r in per if r["located"])
    total_gold_sem = sum(r["gold_chars"] for r in per)
    hyp_sem_chars = len(normalize_text(hyp, semantic=True))
    return {
        "chunks": n_chunks,
        "recovered": n_loc,
        "coverage_chunks": round(n_loc / n_chunks, 4) if n_chunks else 0.0,
        "coverage_chars": round(located_chars / total_gold_sem, 4) if total_gold_sem else 0.0,
        # within-recovered recognition quality (the real OCR signal)
        "within_cer": round(tot_edits / tot_gold_chars, 4) if tot_gold_chars else 0.0,
        "within_wer": round(tot_wedits / tot_gold_words, 4) if tot_gold_words else 0.0,
        # how much tool text was never claimed by any gold chunk (ads/extra/noise)
        "extra_ratio": round(max(0, hyp_sem_chars - matched_hyp_chars) /
                             hyp_sem_chars, 4) if hyp_sem_chars else 0.0,
        "per_chunk": per,
    }
