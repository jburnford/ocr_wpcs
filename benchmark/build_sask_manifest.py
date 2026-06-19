#!/usr/bin/env python3
"""Build sask_manifest.csv: map each faithful-markup .md to its CSV row + PDF.

Join strategy:
  - Extract an ISO date from each faithful .md filename. Three date forms occur:
    YYYY-MM-DD, YYYY_MM_DD, and YYYYMMDD01 (8 date digits + '01' issue suffix).
  - The CSV `Date` is the primary key. It is unique for 37/40 rows; the 3
    collisions are all SHNO issues dated 1914-01-01 (Cut Knife / Davidson /
    Foam Lake). On a date collision, disambiguate by checking that a token from
    the .md filename prefix (the community/paper name) appears in the CSV
    `Filename`.
  - A rapidfuzz title score (md title vs CSV Article_Title) is recorded as a
    confidence check only; the join itself is date(+paper).
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

SASK = Path("/home/jic823/plato/wpcs-ocr/gold_standard_sask_clone/Article_Gold_Standards")
CSV_IN = SASK / "OCR_Gold_Standard_Complete.csv"
FAITHFUL = SASK / "Transcription_Files" / "Faithful_Markup"
SRC_PDFS = SASK / "Source_Files_Complete"
OUT = Path("/home/jic823/plato/wpcs-ocr/benchmark/sask_manifest.csv")


def iso_date(name: str) -> str | None:
    """Pull an ISO date out of a faithful .md filename."""
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # YYYYMMDD followed by a 2-digit issue suffix
    m = re.search(r"(\d{4})(\d{2})(\d{2})\d{2}", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def md_title(name: str) -> str:
    """Title portion of a faithful .md filename: between the date and _faithful."""
    stem = name.replace("_faithful.md", "")
    stem = re.sub(r"^.*?\d{4}[-_]?\d{2}[-_]?\d{2}(?:\d{2})?[_-]?", "", stem)
    return stem.replace("_", " ").strip()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main() -> int:
    rows = list(csv.DictReader(CSV_IN.open()))
    print(f"CSV rows: {len(rows)}", file=sys.stderr)

    # index PDFs by filename
    pdf_by_name = {p.name: p for p in SRC_PDFS.rglob("*.pdf")}
    print(f"Source PDFs found: {len(pdf_by_name)}", file=sys.stderr)

    # group CSV rows by date
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        by_date.setdefault(r["Date"], []).append(r)

    md_files = sorted(
        p for p in FAITHFUL.iterdir()
        if p.name.endswith("_faithful.md")
    )
    print(f"Faithful .md files: {len(md_files)}", file=sys.stderr)

    out_rows = []
    unmatched = []
    for md in md_files:
        d = iso_date(md.name)
        if d is None or d not in by_date:
            unmatched.append((md.name, f"no date match ({d})"))
            continue
        candidates = by_date[d]
        if len(candidates) == 1:
            row = candidates[0]
        else:
            # date collision: disambiguate by a prefix token in the CSV Filename
            prefix = md.name.split(str(d.split("-")[0]))[0]  # text before the year
            tokens = [t for t in re.split(r"[_\s]+", prefix) if len(t) > 2]
            row = None
            for cand in candidates:
                fn = cand["Filename"].lower()
                if any(t.lower() in fn for t in tokens):
                    row = cand
                    break
            if row is None:
                unmatched.append((md.name, f"collision on {d}, no token match"))
                continue
        score = fuzz.token_sort_ratio(norm(md_title(md.name)), norm(row["Article_Title"]))
        pdf = pdf_by_name.get(row["Filename"])
        out_rows.append({
            "md_file": md.name,
            "csv_filename": row["Filename"],
            "source": row["Source"],
            "newspaper": row["Newspaper"],
            "date": row["Date"],
            "article_title": row["Article_Title"],
            "pdf_path": str(pdf) if pdf else "MISSING",
            "pdf_page_number": row["PDF_Page_Number"],
            "readability": row["Readability"].strip().upper(),
            "title_match_score": round(score / 100, 3),
        })

    fields = ["md_file", "csv_filename", "source", "newspaper", "date",
              "article_title", "pdf_path", "pdf_page_number", "readability",
              "title_match_score"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: r["md_file"]))

    print(f"\nMatched {len(out_rows)}/{len(md_files)}  -> {OUT}", file=sys.stderr)
    low = [r for r in out_rows if r["title_match_score"] < 0.6]
    miss = [r for r in out_rows if r["pdf_path"] == "MISSING"]
    if low:
        print(f"\n{len(low)} rows with title_match_score < 0.6 (review):", file=sys.stderr)
        for r in low:
            print(f"  {r['md_file']}  <->  {r['csv_filename']} "
                  f"[{r['article_title']}]  score={r['title_match_score']}", file=sys.stderr)
    if miss:
        print(f"\n{len(miss)} rows with MISSING pdf:", file=sys.stderr)
        for r in miss:
            print(f"  {r['md_file']} -> {r['csv_filename']}", file=sys.stderr)
    if unmatched:
        print(f"\n{len(unmatched)} UNMATCHED .md files:", file=sys.stderr)
        for n, why in unmatched:
            print(f"  {n}: {why}", file=sys.stderr)
    return 0 if len(out_rows) == len(md_files) and not miss else 1


if __name__ == "__main__":
    raise SystemExit(main())
