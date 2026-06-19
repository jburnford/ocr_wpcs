#!/usr/bin/env python3
"""Extract each Sask article's target page from its full-issue PDF.

Reads sask_manifest.csv; for each row, pulls the single page given by
pdf_page_number out of the full issue PDF into sask_page_pdfs/<stem>_p<N>.pdf.
These single-page PDFs are what Chandra runs on (Chandra .md has no page
markers, so feeding it one page = a clean per-article output).
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

MANIFEST = Path("/home/jic823/plato/wpcs-ocr/benchmark/sask_manifest.csv")
OUT = Path("/home/jic823/plato/wpcs-ocr/sask_page_pdfs")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(MANIFEST.open()))
    ok = 0
    errs: list[str] = []
    for r in rows:
        pdf_path = Path(r["pdf_path"])
        page_n = int(r["pdf_page_number"])  # 1-indexed
        stem = pdf_path.stem
        dst = OUT / f"{stem}_p{page_n}.pdf"
        try:
            reader = PdfReader(str(pdf_path))
            total = len(reader.pages)
            if not (1 <= page_n <= total):
                errs.append(f"{stem}: page {page_n} out of range (1..{total})")
                continue
            writer = PdfWriter()
            writer.add_page(reader.pages[page_n - 1])
            with dst.open("wb") as f:
                writer.write(f)
            ok += 1
        except Exception as e:
            errs.append(f"{stem}: {e}")
    print(f"Extracted {ok}/{len(rows)} target-page PDFs -> {OUT}", file=sys.stderr)
    if errs:
        print(f"{len(errs)} errors:", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
