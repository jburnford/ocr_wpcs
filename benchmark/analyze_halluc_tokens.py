#!/usr/bin/env python3
"""Token-level hallucination analysis, head-to-head across tools.

Reuses the harness's exact gold/hyp construction + semantic normalization, then
for each whole-doc corpus extracts the actual hallucinated tokens (real words in
the OCR absent from gold) and classifies each as modernization (within edit-2 of
a gold word — a misread/normalized real page word) or fabrication (nowhere near
the page — the dangerous, NER-extractable kind). Prints, per corpus, a per-tool
fabrication/modernization rate table plus each tool's top fabricated tokens, so
GLM-OCR's behaviour can be read against the others.

Usage: python3 analyze_glmocr_halluc.py [tool ...]   (default: all five)
"""
from __future__ import annotations
import sys
from collections import Counter

import gold_loaders as gl
import ocr_loaders as ol
from rapidfuzz.distance import Levenshtein
from ocr_metrics import normalize_text, _DICT, _strip_boilerplate
from run_eval import _ocr_text, FULLPAGE_SRC, OCR_OUT

ALL_TOOLS = ["olmocr", "chandra", "gemini", "infinity", "glmocr"]
CORPORA = ["bln600", "jacob", "hhtr", "fullpage"]


def classify(gold_text: str, ocr_text: str, max_edit: int = 2):
    """(n_words, modern_tokens, fabricated_tokens) for one pair, using the same
    semantic normalization + edit-distance gate as ocr_metrics.hallucination_split."""
    g_sem = normalize_text(_strip_boilerplate(gold_text), semantic=True)
    o_sem = normalize_text(_strip_boilerplate(ocr_text), semantic=True)
    gold_set = set(g_sem.split())
    by_len: dict[int, list[str]] = {}
    for g in gold_set:
        by_len.setdefault(len(g), []).append(g)
    modern, fab = [], []
    for w in o_sem.split():
        if w in gold_set or w not in _DICT:
            continue
        lw = len(w)
        near = any(
            Levenshtein.distance(w, g, score_cutoff=max_edit) <= max_edit
            for length in range(lw - max_edit, lw + max_edit + 1)
            for g in by_len.get(length, ()))
        (modern if near else fab).append(w)
    return len(o_sem.split()), modern, fab


def gold_for(corpus: str):
    """Yield (stem, gold_text) for a whole-doc corpus, as run_eval builds gold."""
    if corpus == "bln600":
        for p in sorted(gl.BLN600_GT.glob("*.txt")):
            yield p.stem, gl.load_bln600_gt(p.stem)
    elif corpus == "jacob":
        for p in sorted(gl.JACOB_GT.glob("*.xml")):
            yield p.stem, gl.strip_eol_hyphens(gl.load_jacob_gold(p.stem))
    elif corpus == "hhtr":
        for p in sorted(gl.HHTR_GOLD.glob("*.txt")):
            yield p.stem, gl.strip_eol_hyphens(gl.load_hhtr_gold(p.stem))
    elif corpus == "fullpage":
        for p in sorted(FULLPAGE_SRC.glob("*.pdf")):
            yield p.stem, gl.load_fullpage_review(f"{p.stem}_review.md")


def hyp_for(tool: str, corpus: str, stem: str, olm_recs: dict):
    hyp = (_ocr_text(tool, corpus, stem, olm_recs) if tool == "olmocr"
           else _ocr_text(tool, corpus, stem))
    return gl.strip_eol_hyphens(hyp) if hyp else None


def main() -> int:
    tools = [t for t in sys.argv[1:] if t in ALL_TOOLS] or ALL_TOOLS
    for corpus in CORPORA:
        gold = list(gold_for(corpus))
        olm_recs = ol.load_olmocr_jsonl(OCR_OUT / f"olmocr_{corpus}")
        print(f"\n===== {corpus} =====")
        print(f"  {'tool':9} {'n_words':>8} {'modern%':>8} {'fabric%':>8}   top fabricated tokens")
        fabs: dict[str, Counter] = {}
        for tool in tools:
            tot_w = tot_m = tot_f = 0
            fc: Counter = Counter()
            for stem, g in gold:
                hyp = hyp_for(tool, corpus, stem, olm_recs)
                if not hyp:
                    continue
                n, m, f = classify(g, hyp)
                tot_w += n; tot_m += len(m); tot_f += len(f)
                fc.update(f)
            fabs[tool] = fc
            if not tot_w:
                print(f"  {tool:9} {'—':>8}")
                continue
            top = ', '.join(w for w, _ in fc.most_common(12))
            print(f"  {tool:9} {tot_w:8d} {100*tot_m/tot_w:7.2f}% {100*tot_f/tot_w:7.2f}%   {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
