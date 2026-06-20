#!/usr/bin/env python3
"""Convert OCR tool outputs into the canonical block JSON (see canonical.py).

The adapters are markup parsers: they consume each tool's formatting into
structure (category / cells) so the scored `text` is clean content. Pilot:
Infinity (already block-structured) + a generic markup adapter for the
text/markdown/HTML tools (Chandra, Gemini, olmOCR).
"""
from __future__ import annotations
from pathlib import Path

import ocr_loaders as ol
import canonical as C

# Infinity block category -> canonical vocab
_INF_MAP = {
    "text": "body", "paragraph": "body", "list": "list", "title": "heading",
    "section": "heading", "table": "table", "figure": "figure",
    "figure_caption": "caption", "table_caption": "caption", "caption": "caption",
    "header": "header", "page_header": "header", "footer": "footer",
    "page_footer": "footer", "footnote": "footnote", "page_footnote": "footnote",
    "page_number": "page_number",
}


def infinity_to_canonical(doc_id: str, infinity_dir, corpus: str) -> dict | None:
    """Recovered Infinity JSON (list of pages of {bbox,category,text}) -> canonical."""
    import json
    jf = Path(infinity_dir) / f"{doc_id}.json"
    if not jf.exists():
        return None
    data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
    pages_in = [data] if (data and isinstance(data[0], dict)) else \
        [p for p in data if isinstance(p, list)]
    pages = []
    for pi, blocks_in in enumerate(pages_in, 1):
        blocks = []
        for i, b in enumerate(blocks_in):
            if not isinstance(b, dict):
                continue
            cat = _INF_MAP.get((b.get("category") or "").lower(), "body")
            raw = b.get("text", "")
            block = {"order": i, "category": cat}
            if cat == "table" and "<" in raw:
                block["cells"] = C.cells_from_html(raw)
                block["text"] = "\n".join(" ".join(c for c in r if c)
                                          for r in block["cells"])
            else:
                block["text"] = C.clean_markup(raw)
            if b.get("bbox"):
                block["bbox"] = b["bbox"]
            if not block.get("text") and not block.get("cells"):
                continue
            blocks.append(block)
        pages.append({"page": pi, "blocks": blocks})
    return {"doc_id": doc_id, "corpus": corpus,
            "meta": {"tool": "infinity"}, "pages": pages}


def markup_to_canonical(doc_id: str, raw_text: str, corpus: str,
                        tool: str) -> dict:
    """Generic adapter for a tool that emits one text blob (markdown/HTML/plain):
    split into body blocks on blank lines; extract markdown/HTML tables as cells.
    Formatting is stripped via clean_markup, so only content is scored."""
    blocks = []
    order = 0
    # pull out HTML tables first as structured blocks
    if "<table" in raw_text.lower():
        import re
        for m in re.finditer(r"(?is)<table.*?</table>", raw_text):
            cells = C.cells_from_html(m.group(0))
            if cells:
                blocks.append({"order": order, "category": "table",
                               "cells": cells,
                               "text": "\n".join(" ".join(c for c in r if c)
                                                 for r in cells)})
                order += 1
        raw_text = re.sub(r"(?is)<table.*?</table>", "\n", raw_text)
    for chunk in C.clean_markup(raw_text).split("\n\n"):
        t = chunk.strip()
        if t:
            blocks.append({"order": order, "category": "body", "text": t})
            order += 1
    return {"doc_id": doc_id, "corpus": corpus,
            "meta": {"tool": tool}, "pages": [{"page": 1, "blocks": blocks}]}


def chandra_to_canonical(doc_id, chandra_dir, corpus):
    md = ol.load_chandra_md(chandra_dir, doc_id)
    return markup_to_canonical(doc_id, md, corpus, "chandra") if md else None


def gemini_to_canonical(doc_id, gemini_dir, corpus):
    t = ol.load_gemini(gemini_dir, doc_id)
    return markup_to_canonical(doc_id, t, corpus, "gemini") if t else None
