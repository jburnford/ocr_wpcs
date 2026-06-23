#!/usr/bin/env python3
"""Run order-invariant chunk-aligned scoring on the full-page set, all tools.

Reports paragraph-level and article-level coverage + within-chunk CER/WER +
extra-content ratio, per tool. Writes results/fullpage_chunk_eval.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import gold_loaders as gl
import ocr_loaders as ol
from chunk_eval import score_chunks, segment_gold

BENCH = Path("/home/jic823/plato/wpcs-ocr/benchmark")
OCR_OUT = BENCH / "ocr_output"
INF = Path("/home/jic823/plato/wpcs-ocr/infinity_output/fullpage")
FULLPAGE_SRC = Path("/home/jic823/plato/wpcs-ocr/fullpage_pdfs")
TOOLS = ["olmocr", "chandra", "gemini", "infinity", "glmocr"]


def hyp_text(tool: str, stem: str, olmocr_recs: dict) -> str | None:
    if tool == "olmocr":
        rec = olmocr_recs.get(f"{stem}.pdf")
        return gl.strip_markdown(ol.olmocr_full_text(rec)) if rec else None
    if tool == "chandra":
        return gl.strip_markdown(ol.load_chandra_md(OCR_OUT / "chandra_fullpage", stem))
    if tool == "gemini":
        return gl.strip_markdown(ol.load_gemini(OCR_OUT / "gemini_fullpage", stem))
    if tool == "glmocr":
        return gl.strip_markdown(ol.load_gemini(OCR_OUT / "glmocr_fullpage", stem))
    if tool == "infinity":
        return gl.strip_markdown(ol.load_infinity(INF, stem))
    return None


def main() -> int:
    stems = sorted(p.stem for p in FULLPAGE_SRC.glob("*.pdf"))
    # Prefer the newer olmOCR-2 run, matching run_eval.py, so the multi-column
    # score uses the same release as every other corpus.
    _o2 = OCR_OUT / "olmocr2_fullpage"
    olm = ol.load_olmocr_jsonl(_o2 if _o2.is_dir() else OCR_OUT / "olmocr_fullpage")
    out: dict = {"paragraph": {}, "article": {}}
    for level in ("paragraph", "article"):
        # corpus accumulators per tool
        agg = {t: {"chunks": 0, "recovered": 0, "ce": 0, "cd": 0, "we": 0, "wd": 0,
                   "loc_chars": 0, "tot_chars": 0, "hyp_chars": 0,
                   "extra_num": 0, "extra_den": 0, "per_file": []} for t in TOOLS}
        for stem in stems:
            gold_chunks, _ads = segment_gold(f"{stem}_review.md", level)
            gsem = [len(__import__("ocr_metrics").normalize_text(c, semantic=True))
                    for c in gold_chunks]
            for t in TOOLS:
                h = hyp_text(t, stem, olm)
                if not h:
                    continue
                r = score_chunks(gold_chunks, h)
                a = agg[t]
                a["chunks"] += r["chunks"]; a["recovered"] += r["recovered"]
                a["hyp_chars"] += len(__import__("ocr_metrics").normalize_text(
                    h, semantic=True))
                # recompute corpus sums from per_chunk for char-weighting
                for pc in r["per_chunk"]:
                    a["tot_chars"] += pc["gold_chars"]
                    if pc["located"]:
                        a["loc_chars"] += pc["gold_chars"]
                        a["ce"] += pc["cer"] * pc["gold_chars"]
                        a["cd"] += pc["gold_chars"]
                        a["we"] += pc["wer"] * len(pc["gold_preview"].split())  # approx
                a["per_file"].append({"stem": stem, **{k: r[k] for k in
                    ("chunks", "recovered", "coverage_chunks", "coverage_chars",
                     "within_cer", "within_wer", "extra_ratio")}})
        # finalize
        for t in TOOLS:
            a = agg[t]
            correct = max(0.0, a["loc_chars"] - a["ce"])   # correct gold chars recovered
            recall = correct / a["tot_chars"] if a["tot_chars"] else 0.0
            precision = correct / a["hyp_chars"] if a["hyp_chars"] else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            out[level][t] = {
                "chunks": a["chunks"], "recovered": a["recovered"],
                "coverage_chunks": round(a["recovered"] / a["chunks"], 4) if a["chunks"] else 0.0,
                "coverage_chars": round(a["loc_chars"] / a["tot_chars"], 4) if a["tot_chars"] else 0.0,
                "within_cer": round(a["ce"] / a["cd"], 4) if a["cd"] else 0.0,
                # char-level precision/recall/F1 on the order-invariant alignment:
                # recall = correct gold chars / all gold chars (coverage + recognition);
                # precision = correct gold chars / all emitted chars (penalizes
                # over-reading + fabrication); F1 = harmonic mean (single ranking number).
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4),
                "per_file": a["per_file"],
            }
    (BENCH / "results" / "fullpage_chunk_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    for level in ("paragraph", "article"):
        print(f"\n=== {level.upper()}-level (order-invariant) ===")
        print(f"{'tool':9} {'cov(chars)':>11} {'recCER':>8} {'prec':>7} {'recall':>7} {'F1':>7}")
        for t in TOOLS:
            d = out[level][t]
            print(f"{t:9} {d['coverage_chars']*100:10.1f}% {d['within_cer']*100:7.1f}% "
                  f"{d['precision']*100:6.1f}% {d['recall']*100:6.1f}% {d['f1']*100:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
