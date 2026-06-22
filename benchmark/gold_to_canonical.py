#!/usr/bin/env python3
"""Convert gold standards into the canonical block JSON (see canonical.py).

Pilot converters: jacob (PAGE-XML) and tables (.xlsx). Each returns a canonical
doc dict; write_* helpers emit <out>/<doc_id>.json into the data repo's
<corpus>/gold_json/.

Continuity requirement: canonical.flatten(jacob_to_canonical(stem)) must equal
gold_loaders.load_jacob_gold(stem) exactly, so every current jacob metric is
reproduced. Verified by tests/test_canonical.py.
"""
from __future__ import annotations
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import openpyxl as _openpyxl

import gold_loaders as gl
import canonical as C

_STRUCT_TYPE = re.compile(r"structure\s*\{[^}]*type:\s*([a-zA-Z_-]+)")
# PAGE structure type -> canonical category
_TYPE_MAP = {
    "page-number": "page_number", "paragraph": "body", "heading": "heading",
    "header": "header", "footer": "footer", "footnote": "footnote",
    "caption": "caption", "table": "table", "marginalia": "marginalia",
    "credit": "caption", "drop-capital": "body", "floating": "body",
}


def jacob_to_canonical(stem: str) -> dict:
    """PAGE-XML -> canonical. Mirrors load_jacob_gold's line collection so the
    flattened text is identical; page-number regions become score=False blocks."""
    root = ET.parse(gl.JACOB_GT / f"{stem}.xml").getroot()
    ns = f"{{{gl.PAGE_NS}}}"
    page = root.find(f"{ns}Page")
    regions = {r.get("id"): r for r in page.findall(f"{ns}TextRegion")}
    ro = page.find(f"{ns}ReadingOrder")
    if ro is not None:
        order = [ri.get("regionRef") for ri in sorted(
            ro.iter(f"{ns}RegionRefIndexed"),
            key=lambda ri: int(ri.get("index")))]
    else:
        order = list(regions)
    blocks = []
    for i, rid in enumerate(order):
        reg = regions.get(rid)
        if reg is None:
            continue
        custom = reg.get("custom") or ""
        m = _STRUCT_TYPE.search(custom)
        category = _TYPE_MAP.get(m.group(1) if m else "", "body")
        lines = []
        for tl in reg.findall(f"{ns}TextLine"):
            u = tl.find(f"{ns}TextEquiv/{ns}Unicode")
            if u is not None and u.text and u.text.strip():
                lines.append(u.text)
        if not lines:
            continue
        text = re.sub(r"\s*<gap/>\s*", " ", "\n".join(lines))
        coords = reg.find(f"{ns}Coords")
        block = {"order": i, "category": category, "text": text}
        # page-number regions are furniture: no OCR tool should reproduce them
        if category == "page_number":
            block["score"] = False
        if coords is not None and coords.get("points"):
            block["bbox"] = _bbox(coords.get("points"))
        blocks.append(block)
    return {"doc_id": stem, "corpus": "jacob",
            "meta": {"source_format": "PAGE-XML", "lang": "en-EME"},
            "pages": [{"page": 1, "blocks": blocks}]}


def _bbox(points: str) -> list[int]:
    xs, ys = [], []
    for pt in points.split():
        x, _, y = pt.partition(",")
        try:
            xs.append(int(float(x))); ys.append(int(float(y)))
        except ValueError:
            pass
    return [min(xs), min(ys), max(xs), max(ys)] if xs else [0, 0, 0, 0]


def tables_to_canonical(xlsx_path: str | Path, doc_id: str) -> dict:
    """.xlsx gold -> a single table block whose `cells` are the sheet rows and
    whose `text` is a deterministic cell-flatten (so the flat-text view is
    formatting-independent; cell-recall uses `cells`)."""
    wb = _openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    cells = []
    for row in wb.active.iter_rows(values_only=True):
        vals = ["" if v is None else str(v).strip() for v in row]
        if any(vals):
            cells.append(vals)
    text = "\n".join(" ".join(c for c in r if c) for r in cells)
    return {"doc_id": doc_id, "corpus": "tables",
            "meta": {"source_format": "xlsx"},
            "pages": [{"page": 1, "blocks": [
                {"order": 0, "category": "table", "cells": cells, "text": text}]}]}


def _one_body(doc_id: str, corpus: str, text: str, fmt: str) -> dict:
    """Flat-text gold (bln600, fullpage, sask): a single body block whose text is
    the existing gold verbatim, so flatten() == the current gold (continuity)."""
    return {"doc_id": doc_id, "corpus": corpus, "meta": {"source_format": fmt},
            "pages": [{"page": 1, "blocks": [
                {"order": 0, "category": "body", "text": text}]}]}


def bln600_to_canonical(stem: str) -> dict:
    return _one_body(stem, "bln600", gl.load_bln600_gt(stem), "txt")


def hhtr_to_canonical(stem: str) -> dict:
    return _one_body(stem, "hhtr", gl.load_hhtr_gold(stem), "txt")


def fullpage_to_canonical(stem: str) -> dict:
    return _one_body(stem, "fullpage",
                     gl.load_fullpage_review(f"{stem}_review.md"), "review-md")


def sask_to_canonical(md_name: str) -> dict:
    return _one_body(Path(md_name).stem, "sask",
                     gl.load_sask_faithful(md_name), "faithful-md")


def manuscripts_to_canonical(stem: str, docx_path: str | Path) -> dict:
    """Each gold segment (a transcribed document within the source) becomes a
    body block; the comparator aligns segment-blocks to the OCR (as eval_manuscripts
    does today)."""
    segs = gl.manuscript_gold_segments(docx_path)
    blocks = [{"order": i, "category": "body", "text": s}
              for i, s in enumerate(segs)]
    return {"doc_id": stem, "corpus": "manuscripts",
            "meta": {"source_format": "docx", "n_segments": len(segs)},
            "pages": [{"page": 1, "blocks": blocks}]}


def write_doc(doc: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{doc['doc_id']}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def build_all(data_root: str | Path) -> dict:
    """Generate <corpus>/gold_json/ for every corpus into the data repo."""
    import csv
    root = Path(data_root)
    bench = Path(__file__).resolve().parent
    counts = {}

    def _emit(corpus, docs):
        out = root / corpus / "gold_json"
        for d in docs:
            write_doc(d, out)
        counts[corpus] = len(docs)

    _emit("jacob", [jacob_to_canonical(p.stem)
                    for p in sorted(gl.JACOB_GT.glob("*.xml"))])
    _emit("bln600", [bln600_to_canonical(p.stem)
                     for p in sorted(gl.BLN600_GT.glob("*.txt"))])
    _emit("hhtr", [hhtr_to_canonical(p.stem)
                   for p in sorted(gl.HHTR_GOLD.glob("*.txt"))])
    from run_eval import FULLPAGE_SRC
    _emit("fullpage", [fullpage_to_canonical(p.stem)
                       for p in sorted(FULLPAGE_SRC.glob("*.pdf"))])
    ms = list(csv.DictReader((bench / "manuscript_manifest.csv").open()))
    _emit("manuscripts", [manuscripts_to_canonical(r["stem"], r["gold_docx"])
                          for r in ms])
    tb = list(csv.DictReader((bench / "table_manifest.csv").open()))
    _emit("tables", [tables_to_canonical(r["gold_xlsx"], r["stem"]) for r in tb])
    sk = list(csv.DictReader((bench / "sask_manifest.csv").open()))
    _emit("sask", [sask_to_canonical(r["md_file"]) for r in sk])
    return counts


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/jic823/plato/ocr_benchmark"
    print("wrote canonical gold:", build_all(root))
