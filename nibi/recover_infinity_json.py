#!/usr/bin/env python3
"""Schema-aware recovery of Infinity-Parser2 result.json files.

The doc2json model emits invalid JSON (unescaped " and \\ inside the free-text
"text" field). But Infinity's block schema is rigid:
    {"bbox": [x,y,x,y], "category": "<word>", "text": "<free text>"}
grouped into pages: [[block,block,...],[...],...]. The bbox and category fields
are always clean; only "text" carries the bad characters. So we regex the clean
block prefixes, take the text as everything up to the reliable block-close
boundary ("} followed by structural chars), and re-serialize with json.dump
(which escapes correctly). Nothing is dropped.

Usage:
    python recover_infinity_json.py <production_dir> [--write] [--pdfdir DIR]
Dry run validates page-count fidelity vs the source PDFs; --write replaces
result.json (original kept as result.json.bad).
"""
import json, glob, os, re, sys

# clean prefix of every block; bbox = 4 numbers, category = quote-free word
_BLOCK = re.compile(r'\{\s*"bbox"\s*:\s*\[([^\]]*)\]\s*,\s*"category"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"')


_NUMRE = re.compile(r'-?\d+\.?\d*')

_HEX = set('0123456789abcdefABCDEF')
_ESC = {'"': '"', '\\': '\\', '/': '/', 'n': '\n', 't': '\t',
        'r': '\r', 'b': '\b', 'f': '\f'}

def _unescape(s: str) -> str:
    """Decode the valid JSON escapes the model emitted (\\" -> ", \\\\ -> \\,
    \\uXXXX -> char) so the text is clean; leave invalid escapes (e.g. LaTeX
    \\circ) as a literal backslash. Raw content quotes pass through untouched."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            nx = s[i + 1]
            if nx in _ESC:
                out.append(_ESC[nx]); i += 2; continue
            if nx == 'u' and i + 6 <= n and all(h in _HEX for h in s[i + 2:i + 6]):
                try:
                    out.append(chr(int(s[i + 2:i + 6], 16))); i += 6; continue
                except ValueError:
                    pass
            out.append('\\'); i += 1; continue        # invalid escape -> literal backslash
        out.append(c); i += 1
    return ''.join(out)


def _num(x):
    # tolerant: the model sometimes garbles a coordinate (e.g. "97x", "95\"> ..").
    # take the first number; bbox is approximate, never worth dropping a block/doc.
    m = _NUMRE.search(x)
    if not m:
        return 0
    v = m.group(0)
    return float(v) if "." in v else int(v)


def recover(raw: str):
    """Return list-of-pages [[{bbox,category,text}, ...], ...] from malformed raw."""
    starts = list(_BLOCK.finditer(raw))
    pages, cur = [], []
    for i, m in enumerate(starts):
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(raw)
        seg = raw[m.end():seg_end]                 # <text>"}<structure>
        cut = seg.rfind('"}')                      # real block close (struct after has none)
        text = seg if cut == -1 else seg[:cut]
        struct = "" if cut == -1 else seg[cut + 2:]
        bbox = [_num(v) for v in m.group(1).split(",") if v.strip()]
        cur.append({"bbox": bbox, "category": m.group(2), "text": _unescape(text)})
        if "]" in struct or i + 1 == len(starts):  # ']' marks a page (or doc) close
            pages.append(cur); cur = []
    if cur:
        pages.append(cur)
    return pages


def main():
    base = sys.argv[1]
    write = "--write" in sys.argv
    pdfdir = sys.argv[sys.argv.index("--pdfdir") + 1] if "--pdfdir" in sys.argv else None
    try:
        import fitz
    except Exception:
        fitz = None
    ok = mismatch = failed = 0
    for d in sorted(glob.glob(os.path.join(base, "*.pdf"))):
        rj = os.path.join(d, "result.json")
        if not os.path.exists(rj):
            continue
        name = os.path.basename(d)
        raw = open(rj, encoding="utf-8").read()
        try:
            pages = recover(raw)
        except Exception as e:
            print("FAIL  %-30s %s" % (name, str(e)[:50])); failed += 1; continue
        exp = None
        if fitz and pdfdir and os.path.exists(os.path.join(pdfdir, name)):
            try: exp = fitz.open(os.path.join(pdfdir, name)).page_count
            except Exception: exp = None
        tag = ""
        if exp is not None and len(pages) != exp:
            tag = "  PAGE-MISMATCH exp=%d" % exp; mismatch += 1
        else:
            ok += 1
        if write:
            json.dump(pages, open(rj + ".fixed", "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(rj, rj + ".bad"); os.replace(rj + ".fixed", rj)
        print("%-30s pages=%-5d%s" % (name, len(pages), tag))
    print("\nok=%d page_mismatch=%d failed=%d  %s" %
          (ok, mismatch, failed, "[written]" if write else "[dry run]"))


if __name__ == "__main__":
    main()
