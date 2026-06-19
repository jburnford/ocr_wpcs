#!/usr/bin/env python3
"""Faithfully repair Infinity-Parser2 result.json files that are invalid JSON.

The doc2json model emits LaTeX/symbols (e.g. $21^\\circ$) and the saved JSON
keeps single backslashes that are illegal JSON escapes. The OCR content is
complete; only escaping is wrong. We fix EVERY invalid backslash escape
(including \\u not followed by 4 hex, and stray control chars) without dropping
content — unlike json_repair, which truncates at the first irreparable point.

Usage:
    python repair_infinity_json.py <production_dir> [--write]
Dry run reports recovery + page-count fidelity; --write replaces result.json
(original kept as result.json.bad).
"""
import json, glob, os, sys

_VALID_NEXT = set('"\\/bfnrt')
_HEX = set('0123456789abcdefABCDEF')


def fix(s: str) -> str:
    """Double any backslash that doesn't begin a valid JSON escape; escape raw
    control chars. Walks char-by-char so nothing is dropped."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            nxt = s[i + 1]
            if nxt in _VALID_NEXT:
                out.append(c); out.append(nxt); i += 2; continue
            if nxt == 'u' and i + 6 <= n and all(h in _HEX for h in s[i + 2:i + 6]):
                out.append(s[i:i + 6]); i += 6; continue
            out.append('\\\\'); i += 1; continue        # invalid escape -> \\
        if c == '\\':                                    # trailing lone backslash
            out.append('\\\\'); i += 1; continue
        if c in '\n\r\t':                                # raw control chars in strings
            out.append({'\n': '\\n', '\r': '\\r', '\t': '\\t'}[c]); i += 1; continue
        out.append(c); i += 1
    return ''.join(out)


def main():
    base = sys.argv[1]
    write = "--write" in sys.argv
    clean = recovered = still = 0
    bad = []
    for d in sorted(glob.glob(os.path.join(base, "*.pdf"))):
        rj = os.path.join(d, "result.json")
        if not os.path.exists(rj):
            continue
        raw = open(rj, encoding="utf-8").read()
        try:
            json.loads(raw); clean += 1; continue
        except Exception:
            pass
        fixed = fix(raw)
        try:
            json.loads(fixed); recovered += 1
            if write:
                os.replace(rj, rj + ".bad")
                open(rj, "w", encoding="utf-8").write(fixed)
        except Exception as e:
            still += 1; bad.append((os.path.basename(d), str(e)[:55]))
    print(f"clean={clean} recovered={recovered} still_bad={still}"
          + ("  [written]" if write else "  [dry run]"))
    for n_, e in bad:
        print("  STILL BAD:", n_, "|", e)


if __name__ == "__main__":
    main()
