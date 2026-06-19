"""Loaders for olmOCR and Chandra output.

olmOCR writes Dolma-style JSONL: one record per PDF with a concatenated `text`
and `attributes.pdf_page_numbers` = [[char_start, char_end, page_num], ...]
(pages 1-indexed). Records are keyed here by the basename of
`metadata["Source-File"]`.

Chandra writes one directory per input PDF: <out>/<stem>/<stem>.md (full doc).
"""
from __future__ import annotations
import json
from pathlib import Path


def load_olmocr_jsonl(results_dir: str | Path) -> dict[str, dict]:
    """Map PDF basename -> olmOCR record, scanning every output_*.jsonl.

    If a PDF appears in multiple records (multi-line split), the record with
    the longest text wins (olmOCR occasionally re-emits).
    """
    results_dir = Path(results_dir)
    by_pdf: dict[str, dict] = {}
    for jf in sorted(results_dir.rglob("output_*.jsonl")):
        for line in jf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("metadata", {}).get("Source-File", "")
            name = Path(src).name
            if not name:
                continue
            if name not in by_pdf or len(rec.get("text", "")) > len(
                by_pdf[name].get("text", "")
            ):
                by_pdf[name] = rec
    return by_pdf


def olmocr_full_text(record: dict) -> str:
    return record.get("text", "")


def olmocr_page_text(record: dict, page_number: int) -> str | None:
    """Slice the text for one 1-indexed page using pdf_page_numbers.

    Returns None if that page is not present in the record.
    """
    text = record.get("text", "")
    spans = record.get("attributes", {}).get("pdf_page_numbers", []) or []
    for span in spans:
        # span is [char_start, char_end, page_num]
        if len(span) >= 3 and span[2] == page_number:
            return text[span[0]:span[1]]
    return None


def load_chandra_md(output_dir: str | Path, pdf_stem: str) -> str | None:
    """Read Chandra's <output_dir>/<pdf_stem>/<pdf_stem>.md, or None if absent."""
    md = Path(output_dir) / pdf_stem / f"{pdf_stem}.md"
    if md.exists():
        return md.read_text(encoding="utf-8", errors="replace")
    return None


def load_gemini(output_dir: str | Path, stem: str) -> str | None:
    """Read Gemini's <output_dir>/<stem>.json ({"text": ...}), or None if absent."""
    jf = Path(output_dir) / f"{stem}.json"
    if jf.exists():
        return json.loads(jf.read_text(encoding="utf-8", errors="replace")).get("text", "")
    return None


def load_infinity(output_dir: str | Path, stem: str,
                  page: int | None = None) -> str | None:
    """Read Infinity Parser 2 output <output_dir>/<stem>.json and flatten it.

    Infinity emits blocks [{"bbox": [...], "category": "...", "text": "..."}]
    already in reading order. A multi-page PDF is a list of pages, each a list
    of such blocks; a single page may be either a list of pages of length 1 or
    a flat block list. We concatenate block `text` fields (newline-joined).

    `page` (1-indexed) selects one page of a multi-page document; None flattens
    every page. Returns None if the file is absent.
    """
    jf = Path(output_dir) / f"{stem}.json"
    if not jf.exists():
        return None
    data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
    # normalize to a list of pages, each a list of block dicts
    if data and isinstance(data[0], dict):
        pages = [data]                      # flat block list = one page
    else:
        pages = [p for p in data if isinstance(p, list)]
    if page is not None:
        pages = pages[page - 1:page] if 1 <= page <= len(pages) else []
    blocks = [b for pg in pages for b in pg if isinstance(b, dict)]
    return "\n".join(b.get("text", "") for b in blocks if b.get("text"))
