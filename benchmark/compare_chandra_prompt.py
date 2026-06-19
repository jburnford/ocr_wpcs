#!/usr/bin/env python3
"""Compare default Chandra vs the 'do not modernize' prompt variant on Jacob.

Reads two Chandra output trees:
  ocr_output/chandra_jacob/<stem>/<stem>.md            (default prompt)
  ocr_output/chandra_nomodern_jacob/<stem>/<stem>.md   (diplomatic-transcription prompt)
Scores both against the PAGE-XML gold and reports the hallucination split
(modernization vs fabrication) plus CER/WER, and traces a set of archaic tokens
to see whether the instruction made the model preserve period spelling.

Usage:  python compare_chandra_prompt.py
"""
from __future__ import annotations
import re
from pathlib import Path

import gold_loaders as gl
import ocr_loaders as ol
from ocr_metrics import evaluate_pair, corpus_summary, normalize_text, _DICT

OCR_OUT = Path("ocr_output")
VARIANTS = {
    "default": OCR_OUT / "chandra_jacob",
    "no-modernize": OCR_OUT / "chandra_nomodern_jacob",
}
ARCHAIC = ["bloud", "armes", "goodnesse", "professe", "readinesse",
           "widdow", "summerset", "compleat", "publick", "ferved"]


def _clean(t):
    return gl.strip_eol_hyphens(gl.strip_markdown(t)) if t else t


def main():
    stems = sorted(p.stem for p in gl.JACOB_GT.glob("*.xml"))
    summaries = {}
    for name, d in VARIANTS.items():
        if not d.exists():
            print(f"[skip] {name}: {d} not present yet")
            continue
        results = []
        for stem in stems:
            gold = gl.strip_eol_hyphens(gl.load_jacob_gold(stem))
            hyp = ol.load_chandra_md(d, stem)
            if not hyp:
                continue
            results.append(evaluate_pair(stem, gold, _clean(hyp)))
        summaries[name] = (corpus_summary(results), len(results))

    print(f"\n{'variant':<14}{'n':>4}{'semCER':>8}{'semWER':>8}"
          f"{'halluc%':>9}{'modern%':>9}{'fabric%':>9}")
    for name, (s, n) in summaries.items():
        c = s["chapter"]
        print(f"{name:<14}{n:>4}{s['semantic']['overall_cer']*100:>7.2f}%"
              f"{s['semantic']['overall_wer']*100:>7.2f}%"
              f"{c['overall_hallucination_rate']*100:>8.2f}%"
              f"{c['overall_modernization_rate']*100:>8.2f}%"
              f"{c['overall_fabrication_rate']*100:>8.2f}%")

    # token-level fidelity trace
    print("\narchaic-token preservation (gold has the archaic form):")
    print(f"  {'token':<13}" + "".join(f"{v:>14}" for v in VARIANTS))
    for tok in ARCHAIC:
        cells = {}
        present_in_gold = 0
        for name, d in VARIANTS.items():
            if name not in summaries:
                cells[name] = "-"; continue
            kept = mod = docs = 0
            for stem in stems:
                g = gl.load_jacob_gold(stem).lower()
                if tok not in g:
                    continue
                docs += 1
                hyp = ol.load_chandra_md(d, stem)
                if not hyp:
                    continue
                h = hyp.lower()
                if re.search(r"\b" + re.escape(tok) + r"\b", h):
                    kept += 1
            present_in_gold = docs
            cells[name] = f"{kept}/{docs} kept"
        if present_in_gold:
            print(f"  {tok:<13}" + "".join(f"{cells[v]:>14}" for v in VARIANTS))


if __name__ == "__main__":
    main()
