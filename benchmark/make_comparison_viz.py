#!/usr/bin/env python3
"""Build a self-contained HTML visualization comparing OCR tools.

Reads results/<dataset>_<tool>_evaluation_results.json and emits
results/comparison.html with SVG charts. Re-run after new tool/dataset
results land (e.g. olmOCR Sask). No external dependencies.
"""
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "comparison.html"

# dataset key -> display label; order is presentation order
DATASETS = [
    ("bln600", "BLN600"),
    ("sask", "Sask articles"),
    ("fullpage", "Full pages"),
    ("manuscripts", "Manuscripts"),
    ("tables", "Tables"),
]
# tool key -> (display label, colour); order is bar order within a group
TOOLS = [
    ("baseline", "Legacy OCR", "#9aa0a6"),
    ("olmocr", "olmOCR", "#3b6ea5"),
    ("chandra", "Chandra 2", "#d98a3d"),
    ("gemini", "Gemini 3.5 Flash", "#4f9d69"),
]
TOOL_LABEL = {k: v for k, v, _ in TOOLS}
TOOL_COLOR = {k: c for k, _, c in TOOLS}


def load():
    """data[dataset][tool] = summary dict (or None if absent/empty)."""
    data = {}
    for ds, _ in DATASETS:
        data[ds] = {}
        for tool, _, _ in TOOLS:
            f = RESULTS / f"{ds}_{tool}_evaluation_results.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            summ = d.get("summary") or {}
            if not summ:  # empty {} -> ran but produced nothing (e.g. sask olmocr)
                data[ds][tool] = "empty"
                continue
            summ["_per_file"] = d.get("per_file_results", [])
            data[ds][tool] = summ
    return data


def nice_ceiling(v):
    """Round a max value up to a clean axis ceiling."""
    if v <= 0:
        return 1.0
    for step in (0.5, 1, 2, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100):
        if v <= step:
            return float(step)
    return float(int(v / 25 + 1) * 25)


def bars_panel(title, subtitle, tool_keys, series, ymax, unit="%",
               note=None, w=288, h=232):
    """Grouped bar panel. series = list of (metric_label, {tool: value}, alpha).

    Values are already in display units (e.g. percent)."""
    ml, mt, mb, mr = 38, 30, 46, 10
    pw, ph = w - ml - mr, h - mt - mb
    n = len(tool_keys)
    gw = pw / max(n, 1)
    nser = len(series)
    bw = min(gw * 0.74 / nser, 30)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
             f'aria-label="{title}">']
    parts.append(f'<text x="{w/2}" y="15" class="ct">{title}</text>')
    if subtitle:
        parts.append(f'<text x="{w/2}" y="26" class="cs">{subtitle}</text>')
    # y gridlines + labels
    for frac in (0, 0.5, 1):
        y = mt + ph * (1 - frac)
        val = ymax * frac
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" '
                     f'class="grid"/>')
        lbl = f"{val:g}" if unit != "%" else f"{val:g}"
        parts.append(f'<text x="{ml-4}" y="{y+3:.1f}" class="yl">{lbl}</text>')
    parts.append(f'<text x="10" y="{mt+ph/2}" class="axt" '
                 f'transform="rotate(-90 10 {mt+ph/2})">{unit}</text>')
    # bars
    for ti, tk in enumerate(tool_keys):
        gx = ml + ti * gw
        gcx = gx + gw / 2
        block = nser * bw
        for si, (mlabel, vals, alpha) in enumerate(series):
            v = vals.get(tk)
            bx = gcx - block / 2 + si * bw
            if v is None:
                parts.append(f'<text x="{bx+bw/2:.1f}" y="{mt+ph-4}" '
                             f'class="na">n/a</text>')
                continue
            bh = ph * min(v / ymax, 1.0)
            by = mt + ph - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-2:.1f}" '
                         f'height="{bh:.1f}" fill="{TOOL_COLOR[tk]}" '
                         f'fill-opacity="{alpha}"><title>{TOOL_LABEL[tk]} '
                         f'{mlabel}: {v:g}{unit}</title></rect>')
            parts.append(f'<text x="{bx+bw/2-1:.1f}" y="{by-3:.1f}" '
                         f'class="bl">{v:g}</text>')
        parts.append(f'<text x="{gcx:.1f}" y="{mt+ph+14}" class="tl">'
                     f'{TOOL_LABEL[tk]}</text>')
    if note:
        parts.append(f'<text x="{w/2}" y="{h-6}" class="note">{note}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build():
    data = load()
    css = """
    body{font-family:Georgia,'Times New Roman',serif;line-height:1.6;
      color:#222;max-width:1180px;margin:0 auto;padding:1.6em 1.6em 4em;
      background:#fff;}
    h1{font-size:1.7em;margin-bottom:.1em;}
    h2{font-size:1.25em;margin-top:1.8em;border-bottom:2px solid #e0e0e0;
      padding-bottom:.2em;}
    p.lede{color:#444;}
    .grid-wrap{display:flex;flex-wrap:wrap;gap:14px;margin-top:.6em;}
    .chart{background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;
      width:288px;}
    .chart.wide{width:100%;max-width:920px;}
    text{font-family:Helvetica,Arial,sans-serif;}
    .ct{font-size:11px;font-weight:700;text-anchor:middle;fill:#222;}
    .cs{font-size:8.5px;text-anchor:middle;fill:#888;}
    .yl{font-size:8px;text-anchor:end;fill:#999;}
    .axt{font-size:8px;text-anchor:middle;fill:#999;}
    .tl{font-size:8.5px;text-anchor:middle;fill:#444;}
    .bl{font-size:8px;text-anchor:middle;fill:#333;font-weight:700;}
    .na{font-size:8px;text-anchor:middle;fill:#bbb;font-style:italic;}
    .grid{stroke:#e6e6e6;stroke-width:1;}
    .note{font-size:8px;text-anchor:middle;fill:#a06a2c;}
    .legend{font-size:.85em;color:#555;margin:.4em 0 0;}
    .legend span{display:inline-block;margin-right:1.1em;}
    .sw{display:inline-block;width:11px;height:11px;border-radius:2px;
      vertical-align:-1px;margin-right:4px;}
    table.tbl{border-collapse:collapse;font-size:.82em;margin-top:.6em;
      font-family:Helvetica,Arial,sans-serif;}
    table.tbl th,table.tbl td{border:1px solid #ddd;padding:3px 8px;
      text-align:right;}
    table.tbl th{background:#f0f0f0;}
    table.tbl td.l,table.tbl th.l{text-align:left;}
    .best{font-weight:700;background:#dceede;}
    footer{margin-top:2.4em;font-size:.8em;color:#888;border-top:1px solid #e0e0e0;
      padding-top:.8em;}
    """

    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>OCR Tool Comparison</title>",
            f"<style>{css}</style></head><body>"]
    html.append("<h1>OCR Tool Comparison</h1>")
    html.append("<p class='lede'>olmOCR, Chandra 2, and Gemini 3.5 Flash across "
                "five gold-standard datasets. A fourth tool, the bundled "
                "<b>Legacy OCR</b> shipped with the source scans, exists only "
                "for BLN600. Lower is better for error and hallucination "
                "rates; higher is better for cell recall.</p>")

    # legend
    leg = "".join(
        f"<span><i class='sw' style='background:{c}'></i>{lbl}</span>"
        for _, lbl, c in TOOLS)
    html.append(f"<p class='legend'>{leg}</p>")

    # ---- Section A: semantic CER / WER ----
    html.append("<h2>1 &middot; Accuracy — semantic character &amp; word error "
                "rate</h2>")
    html.append("<p class='lede'>Corpus-level CER and WER on punctuation-"
                "stripped, lowercased text. Each panel has its own y-axis — "
                "error rates span two orders of magnitude across datasets.</p>")
    html.append("<p class='legend'><span><i class='sw' "
                "style='background:#555'></i>solid = CER</span>"
                "<span><i class='sw' style='background:#555;opacity:.5'></i>"
                "pale = WER</span></p>")
    html.append("<div class='grid-wrap'>")
    for ds, label in DATASETS:
        toolset = [t for t, _, _ in TOOLS if data[ds].get(t)
                   and data[ds][t] != "empty"]
        cer, wer = {}, {}
        for t in toolset:
            s = data[ds][t]["semantic"]
            cer[t] = round(s["overall_cer"] * 100, 2)
            wer[t] = round(s["overall_wer"] * 100, 2)
        ymax = nice_ceiling(max(list(cer.values()) + list(wer.values())))
        note = None
        sub = f"corpus CER / WER"
        if ds == "sask":
            sub = "29/40 articles located"
            note = "olmOCR located 0/40; Gemini not run on Sask"
        if ds == "tables":
            note = "CER/WER caveated for tables — see §2"
        html.append(bars_panel(
            label, sub, toolset,
            [("CER", cer, 1.0), ("WER", wer, 0.5)], ymax, "%", note))
    html.append("</div>")

    # ---- Section B: table cell-value recall ----
    html.append("<h2>2 &middot; Tables — cell-value recall</h2>")
    html.append("<p class='lede'>For tabular documents, CER/WER are not "
                "meaningful (a flattened database cannot align to a printed "
                "page). The metric that matters is <b>cell-value recall</b>: "
                "the share of distinct data values the tool captured. Higher "
                "is better.</p>")
    tbl_tools = [t for t in ("olmocr", "chandra", "gemini")
                 if data["tables"].get(t) not in (None, "empty")]
    # per-file recall, plus corpus
    files = [r["filename"] for r in data["tables"][tbl_tools[0]]["_per_file"]]
    recall = {t: {} for t in tbl_tools}
    corpus = {}
    for t in tbl_tools:
        found = total = 0
        for r in data["tables"][t]["_per_file"]:
            recall[t][r["filename"]] = round(r.get("cell_recall", 0) * 100, 1)
            found += r.get("cells_found", 0)
            total += r.get("cells_total", 0)
        corpus[t] = round(found / total * 100, 1) if total else 0
    # build a wide grouped bar chart manually
    rows = files + ["Corpus"]
    w, h = 920, 320
    ml, mt, mb, mr = 40, 24, 78, 12
    pw, ph = w - ml - mr, h - mt - mb
    gw = pw / len(rows)
    bw = min(gw * 0.78 / len(tbl_tools), 34)
    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart wide" role="img" '
           f'aria-label="cell recall">']
    for frac in (0, 0.25, 0.5, 0.75, 1):
        y = mt + ph * (1 - frac)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" '
                   f'class="grid"/>')
        svg.append(f'<text x="{ml-5}" y="{y+3:.1f}" class="yl">'
                   f'{int(frac*100)}</text>')
    svg.append(f'<text x="11" y="{mt+ph/2}" class="axt" '
               f'transform="rotate(-90 11 {mt+ph/2})">cell recall %</text>')
    for ri, row in enumerate(rows):
        gx = ml + ri * gw
        gcx = gx + gw / 2
        block = len(tbl_tools) * bw
        for ti, t in enumerate(tbl_tools):
            v = corpus[t] if row == "Corpus" else recall[t].get(row, 0)
            bx = gcx - block / 2 + ti * bw
            bh = ph * v / 100
            by = mt + ph - bh
            op = 1.0 if row != "Corpus" else 1.0
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-2:.1f}" '
                       f'height="{bh:.1f}" fill="{TOOL_COLOR[t]}" '
                       f'fill-opacity="{op}"><title>{TOOL_LABEL[t]} '
                       f'{row}: {v:g}%</title></rect>')
            svg.append(f'<text x="{bx+bw/2-1:.1f}" y="{by-3:.1f}" '
                       f'class="bl">{v:g}</text>')
        disp = row.replace("_", " ")
        emph = ' font-weight="700"' if row == "Corpus" else ''
        # wrap long labels into two lines
        svg.append(f'<text x="{gcx:.1f}" y="{mt+ph+14}" class="tl"{emph} '
                   f'transform="rotate(20 {gcx:.1f} {mt+ph+14})">{disp}</text>')
    svg.append(f'<line x1="{ml + len(files)*gw:.1f}" y1="{mt}" '
               f'x2="{ml + len(files)*gw:.1f}" y2="{mt+ph}" '
               f'stroke="#bbb" stroke-dasharray="3 2"/>')
    svg.append("</svg>")
    html.append("<div class='grid-wrap'>" + "".join(svg) + "</div>")
    html.append("<p class='note' style='color:#a06a2c;font-size:.8em'>"
                "Pass System and Whereabouts Census tables are scored under "
                "OCAP principles — metrics only; document content is not "
                "reproduced.</p>")

    # ---- Section C: hallucination rate ----
    html.append("<h2>3 &middot; Hallucination rate</h2>")
    html.append("<p class='lede'>The dangerous error: a wrong but real, "
                "plausible word a reader cannot catch. Measured as real-word "
                "errors absent from the gold standard. Lower is better.</p>")
    html.append("<div class='grid-wrap'>")
    for ds, label in DATASETS:
        toolset = [t for t, _, _ in TOOLS if data[ds].get(t)
                   and data[ds][t] != "empty"]
        hr = {}
        for t in toolset:
            hr[t] = round(data[ds][t]["chapter"]["overall_hallucination_rate"]
                          * 100, 2)
        ymax = nice_ceiling(max(hr.values()))
        sub = "29/40 located" if ds == "sask" else "corpus rate"
        html.append(bars_panel(
            label, sub, toolset,
            [("hallucination", hr, 1.0)], ymax, "%"))
    html.append("</div>")

    # ---- reference table ----
    html.append("<h2>4 &middot; All metrics</h2>")
    html.append("<table class='tbl'><tr><th class='l'>Dataset</th>"
                "<th class='l'>Tool</th><th>Files</th><th>CER %</th>"
                "<th>WER %</th><th>Sig. word acc.</th><th>BLEU-4</th>"
                "<th>Halluc. %</th><th>Cell recall %</th></tr>")
    for ds, label in DATASETS:
        toolset = [t for t, _, _ in TOOLS if data[ds].get(t)]
        for t in toolset:
            s = data[ds][t]
            if s == "empty":
                html.append(f"<tr><td class='l'>{label}</td>"
                             f"<td class='l'>{TOOL_LABEL[t]}</td>"
                             f"<td colspan='6' style='text-align:center;"
                             f"color:#bbb'>ran — 0 articles located</td>"
                             f"<td>—</td></tr>")
                continue
            sem, ch = s["semantic"], s["chapter"]
            cr = "—"
            if ds == "tables":
                found = sum(r.get("cells_found", 0) for r in s["_per_file"])
                tot = sum(r.get("cells_total", 0) for r in s["_per_file"])
                cr = f"{found/tot*100:.1f}" if tot else "—"
            html.append(
                f"<tr><td class='l'>{label}</td>"
                f"<td class='l'>{TOOL_LABEL[t]}</td>"
                f"<td>{s['total_files']}</td>"
                f"<td>{sem['overall_cer']*100:.2f}</td>"
                f"<td>{sem['overall_wer']*100:.2f}</td>"
                f"<td>{ch['average_sig_word_accuracy']:.3f}</td>"
                f"<td>{ch['average_bleu']:.3f}</td>"
                f"<td>{ch['overall_hallucination_rate']*100:.2f}</td>"
                f"<td>{cr}</td></tr>")
    html.append("</table>")

    html.append("<footer>Internal results view — generated by "
                "<code>make_comparison_viz.py</code> from "
                "<code>results/*_evaluation_results.json</code>. "
                "BLN600 and the OCAP-restricted tables are scoring-only; "
                "no document content is reproduced here.</footer>")
    html.append("</body></html>")
    OUT.write_text("".join(html))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
