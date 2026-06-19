#!/usr/bin/env python3
"""Render BENCHMARK_REPORT.md from the per-(dataset,tool) result JSONs."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("/home/jic823/plato/wpcs-ocr/benchmark/results")
REPORT = RESULTS / "BENCHMARK_REPORT.md"

TIERS = [("Excellent", 0.0, 0.05), ("Good", 0.05, 0.10), ("Fair", 0.10, 0.15),
         ("Poor", 0.15, 0.25), ("Very Poor", 0.25, 1e9)]

# (key, label, tools)
DATASETS = [
    ("bln600", "BLN600", ["olmocr", "chandra", "gemini", "infinity"]),
    ("sask", "Sask articles (located)", ["olmocr", "chandra", "infinity"]),
    ("fullpage", "Full pages", ["olmocr", "chandra", "gemini", "infinity"]),
    ("manuscripts", "Handwritten manuscripts",
     ["olmocr", "chandra", "gemini", "infinity"]),
    ("tables", "Tables (page-content cols)",
     ["olmocr", "chandra", "gemini", "infinity"]),
    ("jacob", "Early-modern English 1612-1807",
     ["olmocr", "chandra", "infinity", "baseline"]),
]


def load(dataset: str, tool: str) -> dict | None:
    f = RESULTS / f"{dataset}_{tool}_evaluation_results.json"
    return json.loads(f.read_text()) if f.exists() else None


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def metrics_row(label: str, data: dict | None) -> str:
    if not data or not data.get("summary"):
        return f"| {label} | — | — | — | — | — |"
    s = data["summary"]
    return (f"| {label} | {s['total_files']} "
            f"| {pct(s['strict']['overall_cer'])} / {pct(s['strict']['overall_wer'])} "
            f"| {pct(s['semantic']['overall_cer'])} / {pct(s['semantic']['overall_wer'])} "
            f"| {pct(s['semantic']['average_cer'])} "
            f"| {pct(s['semantic']['average_wer'])} |")


def main() -> int:
    L: list[str] = []
    L.append("# OCR Benchmark — olmOCR vs Chandra 2 vs Gemini 3.5 Flash "
             "vs Infinity Parser 2\n")
    L.append("Gold standards: BLN600 (600 cropped 19th-c. newspaper pages), the "
             "Saskatchewan article set (40 articles inside full issues), 8 full "
             "newspaper pages, 5 handwritten manuscripts, 6 tabular documents, and "
             "100 early-modern English pages (1612-1807, Transkribus PAGE-XML gold) "
             "with a bundled Tesseract baseline. CER/WER use strict (whitespace-"
             "only) and semantic (lowercased, punctuation-stripped) normalization. "
             "Spanning ~1612-1921 and print/handwriting/tables, the benchmark lets "
             "the comparison turn on content type rather than a single leaderboard.\n")

    # ---- Table A: accuracy ----
    L.append("## Table A — Accuracy (corpus CER/WER)\n")
    L.append("| Dataset / Tool | Files | Strict CER/WER | Semantic CER/WER "
             "| Sem. avg CER | Sem. avg WER |")
    L.append("|---|--:|--|--|--:|--:|")
    for ds, label, tools in DATASETS:
        for tool in tools:
            L.append(metrics_row(f"{label} — {tool}", load(ds, tool)))
    L.append("")

    # ---- Table A2: evaluation-chapter metrics ----
    L.append("## Table A2 — Evaluation-chapter metrics\n")
    L.append("Aligned with the co-author's Chapter 1 evaluation: WER, "
             "significant-word accuracy (WER over content words only), BLEU-4, "
             "hallucination rate (real-word errors absent from gold; NLTK "
             "~236k-word dictionary). Computed on semantic-normalized text.\n")
    L.append("| Dataset / Tool | Files | WER | Sig. word acc. | BLEU-4 "
             "| Hallucination rate |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for ds, label, tools in DATASETS:
        for tool in tools:
            d = load(ds, tool)
            if not d or not d.get("summary") or "chapter" not in d["summary"]:
                L.append(f"| {label} — {tool} | — | — | — | — | — |")
                continue
            c = d["summary"]["chapter"]
            L.append(f"| {label} — {tool} | {d['summary']['total_files']} "
                     f"| {pct(c['average_wer'])} "
                     f"| {c['average_sig_word_accuracy']:.3f} "
                     f"| {c['average_bleu']:.3f} "
                     f"| {pct(c['overall_hallucination_rate'])} |")
    L.append("")

    # ---- Table A3: hallucination split ----
    L.append("## Table A3 — Hallucination split: modernization vs fabrication\n")
    L.append("A 'hallucination' (real word absent from gold) is two very "
             "different errors on historical text: **modernization** — the model "
             "silently normalizes a real archaic word that IS on the page "
             "(e.g. 'bloud'→'blood'), a fidelity problem; vs **fabrication** — "
             "text nowhere on the page, the kind a downstream NER wrongly "
             "extracts. modernization + fabrication = the Table A2 count.\n")
    L.append("| Dataset / Tool | Hallucination | Modernization | Fabrication |")
    L.append("|---|--:|--:|--:|")
    for ds, label, tools in DATASETS:
        for tool in tools:
            d = load(ds, tool)
            if not d or "chapter" not in (d.get("summary") or {}):
                continue
            c = d["summary"]["chapter"]
            if "overall_modernization_rate" not in c:
                continue
            L.append(f"| {label} — {tool} | {pct(c['overall_hallucination_rate'])} "
                     f"| {pct(c['overall_modernization_rate'])} "
                     f"| {pct(c['overall_fabrication_rate'])} |")
    L.append("")

    # ---- Table B: Sask location ----
    L.append("## Table B — Sask article location rate\n")
    L.append("OCR runs on the whole issue/page; the gold article must be located "
             "within it (rapidfuzz partial-ratio ≥ 0.60). Articles not located "
             "are excluded from Table A CER/WER.\n")
    L.append("| Tool | Located | Total | Rate | Mean match score |")
    L.append("|---|--:|--:|--:|--:|")
    for tool in ["olmocr", "chandra"]:
        d = load("sask", tool)
        if d and "location" in d:
            loc = d["location"]
            scores = [r["score"] for r in loc["records"] if r["located"]]
            mean = sum(scores) / len(scores) if scores else 0.0
            L.append(f"| {tool} | {loc['located']} | {loc['total']} "
                     f"| {pct(loc['location_rate'])} | {mean:.3f} |")
    L.append("")

    # ---- Table C: Sask by readability ----
    L.append("## Table C — Sask CER/WER by gold-standard readability\n")
    L.append("| Tool | Readability | Articles | Sem. CER | Sem. WER |")
    L.append("|---|---|--:|--:|--:|")
    for tool in ["olmocr", "chandra"]:
        d = load("sask", tool)
        if not d:
            continue
        by_tier: dict[str, list[dict]] = {}
        for r in d["per_file_results"]:
            by_tier.setdefault(r.get("readability", "?"), []).append(r)
        for tier in ["GOOD", "MEDIUM", "POOR"]:
            rs = by_tier.get(tier, [])
            if not rs:
                L.append(f"| {tool} | {tier} | 0 | — | — |")
                continue
            ce = sum(x["cer_edits_semantic"] for x in rs)
            cn = sum(x["gold_chars_semantic"] for x in rs)
            we = sum(x["wer_edits_semantic"] for x in rs)
            wn = sum(x["gold_words_semantic"] for x in rs)
            L.append(f"| {tool} | {tier} | {len(rs)} "
                     f"| {pct(ce / cn if cn else 0)} | {pct(we / wn if wn else 0)} |")
    L.append("")

    # ---- Table D: BLN600 incl. bundled baseline ----
    base = load("bln600", "baseline")
    if base and base.get("summary"):
        L.append("## Table D — BLN600 incl. bundled baseline OCR\n")
        L.append("| Tool | Files | Strict CER/WER | Semantic CER/WER "
                 "| Sem. avg CER | Sem. avg WER |")
        L.append("|---|--:|--|--|--:|--:|")
        for tool, lbl in [("olmocr", "olmOCR"), ("chandra", "Chandra"),
                          ("gemini", "Gemini 3.5 Flash"),
                          ("infinity", "Infinity Parser 2"),
                          ("baseline", "bundled baseline")]:
            L.append(metrics_row(lbl, load("bln600", tool)))
        L.append("")

    # ---- Table E: table cell-value recall ----
    L.append("## Table E — Tables: cell-value recall\n")
    L.append("CER/WER are **not meaningful for tables**: a flattened database "
             "cannot align character-for-character to a printed page (page "
             "headers, table titles, printed column labels, constant columns "
             "repeated down every row, 2-D reading order). The metric that "
             "matters for data extraction is **cell-value recall** — the "
             "fraction of the table's distinct data values the OCR captured. "
             "Table CER/WER in Table A is retained only as a caveated "
             "secondary figure.\n")
    tools_t = ["olmocr", "chandra", "gemini", "infinity"]
    L.append("| Table | data cells | " + " | ".join(tools_t) + " |")
    L.append("|---|--:|" + "--:|" * len(tools_t))
    perfile = {t: {r["filename"]: r for r in (load("tables", t) or {}).get(
        "per_file_results", [])} for t in tools_t}
    tbl_names = sorted({fn for t in tools_t for fn in perfile[t]})
    totals = {t: [0, 0] for t in tools_t}
    for name in tbl_names:
        ncells = next((perfile[t][name]["cells_total"] for t in tools_t
                       if name in perfile[t]), 0)
        cells = []
        for t in tools_t:
            r = perfile[t].get(name)
            if r and "cell_recall" in r:
                cells.append(pct(r["cell_recall"]))
                totals[t][0] += r["cells_found"]
                totals[t][1] += r["cells_total"]
            else:
                cells.append("—")
        L.append(f"| {name} | {ncells} | " + " | ".join(cells) + " |")
    corpus = []
    for t in tools_t:
        f, n = totals[t]
        corpus.append(pct(f / n) if n else "—")
    L.append(f"| **Corpus** | {totals[tools_t[0]][1] or totals[tools_t[2]][1]} "
             f"| " + " | ".join(f"**{c}**" for c in corpus) + " |")
    L.append("")

    # ---- quality tiers ----
    L.append("## Quality tiers (semantic WER distribution)\n")
    for ds, label, tools in DATASETS:
        for tool in tools:
            d = load(ds, tool)
            if not d or not d["per_file_results"]:
                continue
            counts = {name: 0 for name, _, _ in TIERS}
            for r in d["per_file_results"]:
                w = r["wer_semantic"]
                for name, lo, hi in TIERS:
                    if lo <= w < hi:
                        counts[name] += 1
                        break
            dist = "  ".join(f"{k}:{v}" for k, v in counts.items())
            L.append(f"- **{ds}/{tool}** ({len(d['per_file_results'])} files): {dist}")
    L.append("")

    REPORT.write_text("\n".join(L))
    print(f"Report written: {REPORT}")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
