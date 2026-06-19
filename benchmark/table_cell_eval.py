#!/usr/bin/env python3
"""Row-aligned cell-level table evaluation.

The table gold .xlsx files are normalized research databases, not faithful grid
transcriptions of the printed page, so a strict cell[i][j] comparison is not
possible. This upgrades the crude "value found anywhere in the OCR" recall
(gold_loaders.table_cell_recall) to "value found in the OCR row aligned to that
value's gold row":

  - each tool's OCR is parsed into rows (HTML <table>, markdown pipe table, or
    plain lines);
  - gold data rows are aligned to OCR rows by a monotonic, non-overlapping DP
    that maximises value hits (a gold row may span up to MAXSPAN OCR rows to
    absorb line wrap);
  - every gold cell is then scored: hit-in-row (captured, correct row),
    wrong-row (captured somewhere else — a misplacement the old metric counted
    as success), or missed.

Numeric cells are matched token-exact on digit runs so "$400 00" credits 400
but "4000" does not. Output: console summary + results/table_cell_report.html.
"""
from __future__ import annotations
import csv
import html
import json
import re
from pathlib import Path

import openpyxl

import gold_loaders as gl

BENCH = Path("/home/jic823/plato/wpcs-ocr/benchmark")
OCR = BENCH / "ocr_output"
MANIFEST = BENCH / "table_manifest.csv"
RESULTS = BENCH / "results"
TOOLS = ("olmocr", "chandra", "gemini")
MAXSPAN = 3  # a gold row may align to up to this many consecutive OCR rows


# --------------------------------------------------------------- normalize ---
def norm(s: object) -> str:
    """Lowercase, collapse dot-leaders ('Weikamp.....teacher') and whitespace."""
    t = str(s if s is not None else "").lower()
    t = re.sub(r"\.{2,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ------------------------------------------------------------------- gold ---
def gold_rows(xlsx: str) -> tuple[list[str], list[list[dict]]]:
    """(column names, rows). Each row is a list of cell dicts for the page-
    content, data-bearing columns: {col, raw, text, numeric}."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if len(rows) < 2:
        return [], []
    headers = [str(c) if c is not None else "" for c in rows[0]]
    keep = gl.table_content_columns(headers)
    data = rows[1:]
    # keep only data-bearing columns (>=2 distinct non-empty values)
    cols = []
    for ci in keep:
        vals = {str(r[ci]).strip() for r in data
                if ci < len(r) and r[ci] is not None and str(r[ci]).strip()}
        if len(vals) >= 2:
            cols.append(ci)
    out_rows = []
    for r in data:
        cells = []
        for ci in cols:
            v = r[ci] if ci < len(r) else None
            if v is None or not str(v).strip():
                cells.append(None)
                continue
            numeric = is_num(v)
            raw = str(int(v)) if numeric and float(v).is_integer() else str(v)
            ntext = norm(raw)
            # a short non-numeric value (middle initial 'B.', a 2-char token)
            # cannot be scored by substring without trivial false matches —
            # exclude it rather than count it as a guaranteed miss.
            if not numeric and len(ntext) < 3:
                cells.append(None)
                continue
            cells.append({"col": headers[ci], "raw": raw.strip(),
                          "text": ntext, "numeric": numeric})
        if any(c for c in cells):
            out_rows.append(cells)
    return [headers[ci] for ci in cols], out_rows


# -------------------------------------------------------------- OCR rows ---
def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


def _plain_rows(chunk: str) -> list[str]:
    """Rows from non-HTML text: a pipe-delimited line is split on '|', any other
    non-rule line is one row. Handles documents that mix a pipe/plain body with
    HTML table fragments."""
    out = []
    for line in chunk.splitlines():
        if re.fullmatch(r"[\s=_+\-|:.]*", line):  # blank or rule line
            continue
        if line.count("|") >= 2:
            cells = [c.strip() for c in line.split("|")]
            row = norm(" ".join(_strip_tags(c) for c in cells))
        else:
            row = norm(_strip_tags(line))
        if row:
            out.append(row)
    return out


def ocr_rows(text: str) -> list[str]:
    """Parse a tool's table OCR into normalized row-text strings, in document
    order. <table> blocks are parsed as <tr> rows; text outside them (and whole
    plain/pipe-table documents) is parsed line-wise — so a partial HTML table
    plus a plain-text remainder is not silently dropped."""
    if not text:
        return []
    rows, pos = [], 0
    for m in re.finditer(r"<table[^>]*>.*?</table>", text, re.S | re.I):
        rows += _plain_rows(text[pos:m.start()])
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.S | re.I):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
            row = norm(" ".join(_strip_tags(c) for c in cells))
            if row:
                rows.append(row)
        pos = m.end()
    rows += _plain_rows(text[pos:])
    return rows


# ------------------------------------------------------------- scoring ---
def _cell_in(cell: dict, row_text: str, row_digits: set[str]) -> bool:
    if cell["numeric"]:
        return cell["raw"] in row_digits
    return len(cell["text"]) >= 3 and cell["text"] in row_text


def _hits(grow: list[dict], otext: str) -> int:
    digits = set(re.findall(r"\d+", otext))
    return sum(1 for c in grow if c and _cell_in(c, otext, digits))


def align(grows: list[list[dict]], orows: list[str]) -> list[str]:
    """Monotonic, non-overlapping alignment maximising total cell hits. Returns,
    for each gold row, the joined OCR text it was aligned to ('' if unmatched)."""
    G, N = len(grows), len(orows)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> int:
        if i == G:
            return 0
        opts = [best(i + 1, j)]                      # gold row i unmatched
        if j < N:
            opts.append(best(i, j + 1))              # skip OCR row j
            for k in range(1, MAXSPAN + 1):
                if j + k > N:
                    break
                txt = " ".join(orows[j:j + k])
                opts.append(_hits(grows[i], txt) + best(i + 1, j + k))
        return max(opts)

    matched = [""] * G
    i = j = 0
    while i < G:
        if best(i, j) == best(i + 1, j):
            i += 1
            continue
        if j < N and best(i, j) == best(i, j + 1):
            j += 1
            continue
        chosen_k, chosen = 0, 0
        for k in range(1, MAXSPAN + 1):
            if j + k > N:
                break
            txt = " ".join(orows[j:j + k])
            val = _hits(grows[i], txt) + best(i + 1, j + k)
            if val >= chosen:
                chosen, chosen_k = val, k
        matched[i] = " ".join(orows[j:j + chosen_k])
        i += 1
        j += chosen_k
    return matched


def evaluate(stem: str, xlsx: str) -> dict:
    cols, grows = gold_rows(xlsx)
    full_ocr, per_tool = {}, {}
    for tool in TOOLS:
        txt = load_ocr(tool, stem)
        orows = ocr_rows(txt)
        matched = align(grows, orows) if grows else []
        all_text = norm(txt)
        all_digits = set(re.findall(r"\d+", all_text))
        cells = []  # flat list of per-cell verdicts
        for ri, grow in enumerate(grows):
            mtext = matched[ri] if ri < len(matched) else ""
            mdig = set(re.findall(r"\d+", mtext))
            row_cells = []
            for c in grow:
                if c is None:
                    row_cells.append(None)
                    continue
                in_row = _cell_in(c, mtext, mdig)
                anywhere = _cell_in(c, all_text, all_digits)
                verdict = "hit" if in_row else ("wrongrow" if anywhere else "miss")
                row_cells.append({**c, "verdict": verdict})
                cells.append((c["numeric"], verdict))
            per_tool.setdefault(tool, {"rows": []})
            per_tool[tool]["rows"].append(row_cells)
        tot = len(cells)
        num = [v for n, v in cells if n]
        per_tool[tool]["summary"] = {
            "cells": tot,
            "hit": sum(1 for _, v in cells if v == "hit"),
            "wrongrow": sum(1 for _, v in cells if v == "wrongrow"),
            "miss": sum(1 for _, v in cells if v == "miss"),
            "row_aligned_recall": (sum(1 for _, v in cells if v == "hit") / tot
                                   if tot else 0.0),
            "anywhere_recall": (sum(1 for _, v in cells
                                    if v != "miss") / tot if tot else 0.0),
            "numeric_cells": len(num),
            "numeric_recall": (sum(1 for v in num if v == "hit") / len(num)
                               if num else 0.0),
        }
    return {"stem": stem, "columns": cols, "gold_rows": grows,
            "tools": per_tool}


# -------------------------------------------------------------- OCR load ---
def load_ocr(tool: str, stem: str) -> str:
    if tool == "olmocr":
        for jf in (OCR / "olmocr_tables").rglob("output_*.jsonl"):
            for line in jf.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if stem in r.get("metadata", {}).get("Source-File", ""):
                    return r.get("text", "")
        return ""
    if tool == "chandra":
        md = OCR / "chandra_tables" / stem / f"{stem}.md"
        return md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
    if tool == "gemini":
        jf = OCR / "gemini_tables" / f"{stem}.json"
        if jf.exists():
            return json.loads(jf.read_text(encoding="utf-8",
                                           errors="replace")).get("text", "")
    return ""


# ---------------------------------------------------------------- report ---
def write_html(results: list[dict]) -> None:
    css = """
    body{font-family:Georgia,serif;max-width:1180px;margin:0 auto;padding:1.5em;}
    h1{font-size:1.6em;} h2{font-size:1.2em;margin-top:1.6em;
      border-bottom:2px solid #e0e0e0;}
    table.g{border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;
      font-size:.74em;margin:.5em 0;}
    table.g th,table.g td{border:1px solid #ddd;padding:2px 5px;text-align:left;}
    table.g th{background:#f0f0f0;}
    .hit{background:#74c476;} .wrongrow{background:#ffe9c7;}
    .miss{background:#ffd6d6;} .num{font-weight:700;}
    .rownum{background:#f0f0f0;color:#999;}
    table.s{border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;
      font-size:.82em;margin:.4em 0;}
    table.s th,table.s td{border:1px solid #ddd;padding:3px 8px;text-align:right;}
    table.s td.l{text-align:left;} table.s th{background:#f0f0f0;}
    .legend span{padding:1px 7px;border-radius:3px;margin-right:6px;
      font-family:Helvetica,Arial,sans-serif;font-size:.8em;}
    """
    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         "<title>Table cell evaluation</title>", f"<style>{css}</style>",
         "</head><body>", "<h1>Row-aligned table cell evaluation</h1>",
         "<p>Every gold page-content cell, scored against the OCR row aligned "
         "to its gold row. <b>Numeric cells</b> (bold) are matched token-exact "
         "on digit runs.</p>",
         "<p class='legend'><span class='hit'>hit — captured, correct row</span>"
         "<span class='wrongrow'>captured but wrong row</span>"
         "<span class='miss'>missed</span></p>",
         "<p>The <b>Σ</b> footer row gives per-column <i>hit/wrong-row/miss</i> "
         "counts. A column that is almost entirely wrong-row (e.g. an agency or "
         "section name) is a derived gold field not printed on each page row — "
         "no OCR can place it, so it depresses recall equally for every tool.</p>"]
    # overall summary
    h.append("<h2>Summary</h2><table class='s'><tr><th class='l'>Table</th>"
             "<th class='l'>Tool</th><th>Cells</th><th>Row-aligned recall</th>"
             "<th>Anywhere recall</th><th>Wrong-row</th>"
             "<th>Numeric recall</th></tr>")
    for res in results:
        for tool in TOOLS:
            s = res["tools"][tool]["summary"]
            h.append(
                f"<tr><td class='l'>{html.escape(res['stem'])}</td>"
                f"<td class='l'>{tool}</td><td>{s['cells']}</td>"
                f"<td>{s['row_aligned_recall']*100:.1f}%</td>"
                f"<td>{s['anywhere_recall']*100:.1f}%</td>"
                f"<td>{s['wrongrow']}</td>"
                f"<td>{s['numeric_recall']*100:.1f}% "
                f"({s['numeric_cells']})</td></tr>")
    h.append("</table>")
    # per-table per-tool grids
    for res in results:
        h.append(f"<h2>{html.escape(res['stem'])}</h2>")
        cols = res["columns"]
        for tool in TOOLS:
            s = res["tools"][tool]["summary"]
            h.append(f"<p><b>{tool}</b> — row-aligned {s['row_aligned_recall']*100:.1f}%"
                     f", numeric {s['numeric_recall']*100:.1f}%</p>")
            h.append("<table class='g'><tr><th class='rownum'>#</th>"
                     + "".join(f"<th>{html.escape(c)}</th>" for c in cols)
                     + "</tr>")
            grid = res["tools"][tool]["rows"]
            for ri, row_cells in enumerate(grid, 1):
                tds = [f"<td class='rownum'>{ri}</td>"]
                for c in row_cells:
                    if c is None:
                        tds.append("<td></td>")
                        continue
                    cls = c["verdict"] + (" num" if c["numeric"] else "")
                    tds.append(f"<td class='{cls}'>{html.escape(c['raw'])}</td>")
                h.append("<tr>" + "".join(tds) + "</tr>")
            # per-column verdict footer — a column that is nearly all wrong-row
            # is a derived/sectional gold column not printed on each page row.
            foot = ["<td class='rownum'>Σ</td>"]
            for ci in range(len(cols)):
                vs = [r[ci]["verdict"] for r in grid
                      if ci < len(r) and r[ci] is not None]
                if not vs:
                    foot.append("<td></td>")
                    continue
                hh = vs.count("hit")
                w = vs.count("wrongrow")
                mm = vs.count("miss")
                foot.append(f"<td class='rownum' style='font-size:.9em'>"
                            f"{hh}/{w}/{mm}</td>")
            h.append("<tr>" + "".join(foot) + "</tr></table>")
    h.append("</body></html>")
    (RESULTS / "table_cell_report.html").write_text("".join(h))


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open()))
    results = [evaluate(r["stem"], r["gold_xlsx"]) for r in rows]
    write_html(results)
    print(f"{'table':26s} {'tool':8s} {'cells':>6s} {'row-aln':>8s} "
          f"{'anywhere':>9s} {'wrong-row':>10s} {'numeric':>9s}")
    for res in results:
        for tool in TOOLS:
            s = res["tools"][tool]["summary"]
            print(f"{res['stem']:26s} {tool:8s} {s['cells']:6d} "
                  f"{s['row_aligned_recall']*100:7.1f}% "
                  f"{s['anywhere_recall']*100:8.1f}% "
                  f"{s['wrongrow']:10d} "
                  f"{s['numeric_recall']*100:7.1f}% ({s['numeric_cells']})")
        print()
    print(f"report: {RESULTS / 'table_cell_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
