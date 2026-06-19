#!/usr/bin/env python3
"""Render each Infinity-parsed volume to a readable HTML file.

Reads output/production/<vol>.pdf/result.json (valid JSON after the recovery
finalize; falls back to schema-aware recover() if still raw/invalid) and writes
output/html/<vol>.html. Pages render in reading order; `table` blocks are
emitted as-is (Infinity already produces HTML tables); all other text is
HTML-escaped. $...$ math renders via MathJax (CDN; view with internet).

Usage:  python render_html.py <production_dir> <html_out_dir>
"""
import json, glob, os, sys, html
from recover_infinity_json import recover

_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<script>MathJax={{tex:{{inlineMath:[['$','$']]}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
 body{{font-family:Georgia,'Times New Roman',serif;max-width:52em;margin:2em auto;
   padding:0 1em;line-height:1.45;color:#1a1a1a}}
 h1{{font-size:1.5em;border-bottom:2px solid #333}}
 .page{{border-top:1px solid #ddd;margin-top:2.5em;padding-top:.5em}}
 .pnum{{color:#999;font-size:.75em;text-align:right}}
 .title{{font-weight:bold;font-size:1.1em;margin:1em 0 .3em}}
 .hdr,.ftr{{color:#888;font-size:.82em}}
 .cap{{color:#a60;font-style:italic;font-size:.9em}}
 table{{border-collapse:collapse;font-size:.82em;margin:1em 0;width:100%}}
 td,th{{border:1px solid #bbb;padding:2px 6px;vertical-align:top}}
 th{{background:#f2f2f2}}
</style></head><body>
<h1>{title}</h1>
"""

_TEXTCATS = {"text", "list", "paragraph"}
_CAPCATS = {"figure", "map", "figure_caption", "table_caption", "legend",
            "scale", "currency", "image"}
_FTRCATS = {"footer", "page_footnote", "table_footnote", "footnote"}


def block_html(b: dict) -> str:
    cat = b.get("category", "text")
    t = b.get("text", "")
    if not t:
        return ""
    if cat == "table":
        return f'<div class="tbl">{t}</div>'          # already HTML
    esc = html.escape(t)
    if cat == "title":
        return f'<div class="title">{esc}</div>'
    if cat == "header":
        return f'<div class="hdr">{esc}</div>'
    if cat in _FTRCATS:
        return f'<div class="ftr">{esc}</div>'
    if cat in _CAPCATS:
        return f'<div class="cap">[{cat}] {esc}</div>'
    return f'<p>{esc}</p>'


def load(rj: str):
    raw = open(rj, encoding="utf-8").read()
    try:
        d = json.loads(raw)
        return d if (d and isinstance(d[0], list)) else [d]
    except Exception:
        return recover(raw)


def main():
    base, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    n = 0
    for d in sorted(glob.glob(os.path.join(base, "*.pdf"))):
        rj = os.path.join(d, "result.json")
        if not os.path.exists(rj):
            continue
        name = os.path.basename(d)[:-4]
        pages = load(rj)
        parts = [_HEAD.format(title=html.escape(name))]
        for i, pg in enumerate(pages, 1):
            parts.append(f'<div class="page"><div class="pnum">p.{i}</div>')
            parts.extend(block_html(b) for b in pg if isinstance(b, dict))
            parts.append('</div>')
        parts.append('</body></html>')
        open(os.path.join(out, name + ".html"), "w", encoding="utf-8").write("\n".join(parts))
        n += 1
        print("wrote", name + ".html", f"({len(pages)} pages)")
    print(f"\n{n} HTML files -> {out}")


if __name__ == "__main__":
    main()
