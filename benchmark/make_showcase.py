#!/usr/bin/env python3
"""Generate a Quarto (.qmd) showcase of OCR transcriptions on challenging texts.

For each challenge document it emits: the source page image, a note on why the
text is hard, and each tool's transcription rendered as a word-level diff
against the gold standard — misread words in red (gold word on hover), spurious
insertions in amber, missed words struck through. Lets historians SEE what each
tool gets wrong instead of reading a CER number.

Output: benchmark/showcase/showcase.qmd  (+ images in showcase/img/)
"""
from __future__ import annotations
import csv
import difflib
import html
import re
import subprocess
from pathlib import Path

import openpyxl

import gold_loaders as gl
import ocr_loaders as ol
from locate_article import best_match_span
from ocr_metrics import evaluate_pair, is_dictionary_word

ROOT = Path("/home/jic823/plato/wpcs-ocr")
BENCH = ROOT / "benchmark"
OCR_OUT = BENCH / "ocr_output"
SHOW = BENCH / "showcase"
IMG = SHOW / "img"
# NOTE: the BLN600 British Library set is used for scoring only — it cannot be
# republished, so no BLN600 images/transcriptions appear in this showcase.
FULLPAGE_REVIEW = Path("/home/jic823/plato/wpcs-ocr/gold_standard_sask_clone/"
                       "Full_Page_Gold_Standards/Transcription_Files")

CSS = """<style>
.ocr-diff { font-family: Georgia, serif; line-height: 1.7; background:#fafafa;
            border:1px solid #e0e0e0; border-radius:6px; padding:0.8em 1em; }
.ocr-diff .sub { background:#ffd6d6; border-radius:3px; padding:0 2px; }
.ocr-diff .halluc { background:#c81e1e; color:#fff; font-weight:700;
            border-radius:3px; padding:0 2px; }
.ocr-diff .ins { background:#ffe9c7; border-radius:3px; padding:0 2px; }
.ocr-diff .del { color:#b0b0b0; text-decoration:line-through; }
.ocr-diff .goldwrong { background:#a9d8a0; border-radius:3px; padding:0 2px;
            font-weight:600; }
.ocr-tool { font-weight:600; margin-top:0.6em; }
.ocr-gold { font-family: Georgia, serif; line-height:1.7; background:#eef6ee;
            border:1px solid #cfe0cf; border-radius:6px; padding:0.8em 1em; }
table.tbl-recall { border-collapse:collapse; font-size:0.82em; }
table.tbl-recall th, table.tbl-recall td { border:1px solid #ddd;
            padding:2px 6px; text-align:left; }
table.tbl-recall th { background:#f0f0f0; }
td.c3 { background:#74c476; } td.c2 { background:#c7e9c0; }
td.c1 { background:#ffe9c7; } td.c0 { background:#ffd6d6; }
.tbl-legend span { padding:1px 6px; border-radius:3px; margin-right:4px; }
</style>"""


# every suspected hallucination flagged during rendering is logged here so a
# review CSV can be written for human confirmation
_HALLUC_LOG: list[dict] = []

# Reviewer-confirmed gold-standard errors: cases where human review found the
# OCR reading correct and the human transcription wrong. (document substring,
# gold word, correct OCR word) — all lowercased, punctuation-free.
GOLD_ERRORS = [
    ("nilie", "did", "do"),
    ("nilie", "those", "these"),
]


def _is_gold_error(doc: str, gold_seg: str, hyp_seg: str) -> bool:
    gn, hn = _wnorm(gold_seg), _wnorm(hyp_seg)
    return any(sub in doc.lower() and gn == ge and hn == oe
               for sub, ge, oe in GOLD_ERRORS)


def _wnorm(w: str) -> str:
    """Word key for diff alignment — lowercase, punctuation removed, matching
    the semantic normalization used for scoring. Ensures 'Q.' and 'Q' align."""
    return re.sub(r"[^\w]", "", w.lower())


def word_diff_html(gold: str, hyp: str, doc: str = "", tool: str = "") -> str:
    """Render `hyp` as HTML with word-level differences from `gold` marked.

    Alignment is on semantically-normalized words, so case- and punctuation-
    only differences are NOT highlighted — only genuine word changes show,
    matching what the CER/WER scores actually count.

    A substitution where every OCR word is a real dictionary word is flagged
    as a SUSPECTED hallucination — an automated proxy (per the evaluation
    chapter's dictionary method) that still needs human confirmation, since it
    cannot distinguish a true model hallucination from a gold-standard error
    or an alignment artifact. Each flag is logged to _HALLUC_LOG.
    """
    g = gold.split()
    h = hyp.split()
    gn = [_wnorm(w) for w in g]
    hn = [_wnorm(w) for w in h]
    sm = difflib.SequenceMatcher(None, gn, hn, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        gold_seg = html.escape(" ".join(g[i1:i2]))
        hyp_seg = html.escape(" ".join(h[j1:j2]))
        if tag == "equal":
            out.append(hyp_seg)
        elif tag == "replace":
            hyp_words = h[j1:j2]
            if _is_gold_error(doc, " ".join(g[i1:i2]), " ".join(h[j1:j2])):
                out.append(f'<span class="goldwrong" title="OCR correct — the '
                           f'gold-standard transcription is wrong here (gold: '
                           f'{gold_seg})">{hyp_seg}</span>')
                continue
            is_halluc = hyp_words and all(is_dictionary_word(w) for w in hyp_words)
            cls = "halluc" if is_halluc else "sub"
            tip = ("suspected hallucination (wrong but real word) — gold: "
                   if is_halluc else "misread — gold: ") + gold_seg
            out.append(f'<span class="{cls}" title="{tip}">{hyp_seg}</span>')
            if is_halluc:
                ctx_b = " ".join(h[max(0, j1 - 6):j1])
                ctx_a = " ".join(h[j2:j2 + 6])
                _HALLUC_LOG.append({
                    "document": doc, "tool": tool,
                    "gold": " ".join(g[i1:i2]), "ocr": " ".join(h[j1:j2]),
                    "context": f"...{ctx_b} «{' '.join(h[j1:j2])}» {ctx_a}...",
                    "verdict": "",
                })
        elif tag == "insert":
            out.append(f'<span class="ins" title="not in gold">{hyp_seg}</span>')
        elif tag == "delete":
            out.append(f'<span class="del" title="missed by OCR">{gold_seg}</span>')
    return " ".join(out)


def rasterize(src: Path, dst: Path, page: int = 1) -> None:
    """Make a web-displayable JPEG from a PDF page or an image file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".pdf":
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", "150", "-f", str(page), "-l", str(page),
             "-singlefile", str(src), str(dst.with_suffix(""))],
            check=True, capture_output=True)
    else:
        from PIL import Image
        with Image.open(src) as im:
            im.seek(0)
            im.convert("RGB").save(dst, "JPEG", quality=88)


def manuscript_card(stem: str, gold_docx: str, title: str, why: str,
                    src_img: Path) -> str:
    segments = gl.manuscript_gold_segments(gold_docx)
    gold = "\n".join(segments)
    gem_full = gl.strip_brackets(gl.strip_eol_hyphens(gl.strip_markdown(
        ol.load_gemini(OCR_OUT / "gemini_manuscripts", stem) or "")))
    recs = ol.load_olmocr_jsonl(OCR_OUT / "olmocr_manuscripts")
    olm_full = gl.strip_brackets(gl.strip_eol_hyphens(gl.strip_markdown(
        ol.olmocr_full_text(recs.get(f"{stem}.pdf")) if recs.get(f"{stem}.pdf") else "")))

    chan_full = gl.strip_brackets(gl.strip_eol_hyphens(gl.strip_markdown(
        ol.load_chandra_md(OCR_OUT / "chandra_manuscripts", stem) or "")))

    def aligned(hyp_full: str) -> str:
        if not hyp_full:
            return ""
        return "\n".join(best_match_span(s, hyp_full)["located_text"] for s in segments)

    tools = {"olmOCR": aligned(olm_full), "Chandra 2": aligned(chan_full),
             "Gemini 3.5 Flash": aligned(gem_full)}
    return _card(title, why, src_img, gold, tools)


def fullpage_card(stem: str, title: str, why: str) -> str:
    gold = gl.load_fullpage_review(f"{stem}_review.md")
    gem = ol.load_gemini(OCR_OUT / "gemini_fullpage", stem) or ""
    chan = ol.load_chandra_md(OCR_OUT / "chandra_fullpage", stem) or ""
    recs = ol.load_olmocr_jsonl(OCR_OUT / "olmocr_fullpage")
    olm = ol.olmocr_full_text(recs.get(f"{stem}.pdf")) if recs.get(f"{stem}.pdf") else ""
    img = IMG / f"{stem}.jpg"
    rasterize(ROOT / "fullpage_pdfs" / f"{stem}.pdf", img)
    return _card(title, why, img, gold,
                 {"olmOCR": gl.strip_eol_hyphens(gl.strip_markdown(olm)),
                  "Chandra 2": gl.strip_eol_hyphens(gl.strip_markdown(chan)),
                  "Gemini 3.5 Flash": gl.strip_eol_hyphens(gl.strip_markdown(gem))})


def _tnorm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def table_card(stem: str, gold_xlsx: str, title: str, why: str) -> str:
    """Render a gold table as HTML with each data cell colour-coded by which
    tool(s) captured that value: green=both, amber=Gemini only, blue=olmOCR
    only, red=neither. Visualises cell-value recall on the table's structure."""
    wb = openpyxl.load_workbook(gold_xlsx, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    headers = [str(c) if c is not None else "" for c in rows[0]]
    keep = gl.table_content_columns(headers)
    datacols = []
    for ci in keep:
        col = [str(r[ci]).strip() for r in rows[1:]
               if ci < len(r) and r[ci] is not None and str(r[ci]).strip()]
        if len(set(col)) >= 2:  # data-bearing columns only
            datacols.append(ci)

    gem_raw = gl.strip_markdown(ol.load_gemini(OCR_OUT / "gemini_tables", stem) or "")
    chan_raw = gl.strip_markdown(ol.load_chandra_md(OCR_OUT / "chandra_tables", stem) or "")
    recs = ol.load_olmocr_jsonl(OCR_OUT / "olmocr_tables")
    olm_raw = gl.strip_markdown(ol.olmocr_full_text(recs.get(f"{stem}.pdf"))
                                if recs.get(f"{stem}.pdf") else "")
    gem, chan, olm = _tnorm(gem_raw), _tnorm(chan_raw), _tnorm(olm_raw)

    img = IMG / f"{stem}.jpg"
    rasterize(ROOT / "table_pdfs" / f"{stem}.pdf", img)

    out = [f"## {title}\n", f"::: {{.callout-note}}\n{why}\n:::\n"]
    if img.exists():
        out.append(f"![Source table](img/{img.name}){{width=70%}}\n")
    out.append('<p class="tbl-legend">Cell value captured by: '
               '<span class="c3" style="background:#74c476">all 3 tools</span>'
               '<span class="c2" style="background:#c7e9c0">2 tools</span>'
               '<span class="c1" style="background:#ffe9c7">1 tool</span>'
               '<span class="c0" style="background:#ffd6d6">no tool</span></p>\n')
    th = "".join(f"<th>{html.escape(headers[c])}</th>" for c in datacols)
    trs = [f"<tr>{th}</tr>"]
    for r in rows[1:]:
        tds = []
        for c in datacols:
            v = r[c] if c < len(r) else None
            vs = "" if v is None else str(v).strip()
            vn = _tnorm(vs)
            if len(vn) < 3:
                tds.append(f"<td>{html.escape(vs)}</td>")
                continue
            n = (vn in gem) + (vn in olm) + (vn in chan)
            tds.append(f'<td class="c{n}">{html.escape(vs)}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")
    out.append('<table class="tbl-recall">' + "".join(trs) + "</table>\n")
    ro = gl.table_cell_recall(gold_xlsx, olm_raw)
    rc = gl.table_cell_recall(gold_xlsx, chan_raw)
    rg = gl.table_cell_recall(gold_xlsx, gem_raw)
    out.append(f"\n*Cell-value recall — olmOCR {ro['recall']*100:.0f}% "
               f"({ro['found']}/{ro['total']}), "
               f"Chandra {rc['recall']*100:.0f}% ({rc['found']}/{rc['total']}), "
               f"Gemini {rg['recall']*100:.0f}% ({rg['found']}/{rg['total']}).*\n")
    return "\n".join(out)


def _card(title: str, why: str, src_img: Path, gold: str,
          tools: dict[str, str]) -> str:
    lines = [f"## {title}\n", f"::: {{.callout-note}}\n{why}\n:::\n"]
    if src_img and src_img.exists():
        lines.append(f"![Source page](img/{src_img.name}){{width=60%}}\n")
    lines.append("**Gold standard transcription**\n")
    lines.append(f'<div class="ocr-gold">{html.escape(gold)}</div>\n')
    for tool, hyp in tools.items():
        if not hyp:
            continue
        m = evaluate_pair(title, gold, hyp)
        lines.append(f'<p class="ocr-tool">{tool} — '
                     f'CER {m["cer_semantic"]*100:.1f}% · '
                     f'WER {m["wer_semantic"]*100:.1f}%</p>')
        lines.append('<div class="ocr-diff">'
                      f'{word_diff_html(gold, hyp, title, tool)}</div>\n')
    return "\n".join(lines)


def main() -> int:
    SHOW.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    ck_img = IMG / "charles_kelly.jpg"
    rasterize(Path("/home/jic823/plato/wpcs-ocr/Transcription/Textual "
                   "Transcriptions/Charles Kelly Excerpt/IMG_9242.jpg"), ck_img)
    cb_img = IMG / "colonel_bernard.jpg"
    rasterize(ROOT / "manuscript_pdfs" / "Colonel_Bernard.pdf", cb_img)

    ck_docx = ("/home/jic823/plato/wpcs-ocr/Transcription/Textual Transcriptions/"
               "Charles Kelly Excerpt/Charles Kelly Excerpt.docx")
    cb_docx = ("/home/jic823/plato/wpcs-ocr/Transcription/Textual Transcriptions/"
               "Colonel Bernard Minister of Justice Ottawa/"
               "Colonel Bernard Minister of Justice Ottawa.docx")
    nh_docx = ("/home/jic823/plato/wpcs-ocr/Transcription/Textual Transcriptions/"
               "Testimony of Nilie M. Hyland/Testimony of Nilie M. Hyland.docx")
    nh_img = IMG / "nilie_hyland.jpg"
    rasterize(ROOT / "manuscript_pdfs" / "Testimony_Nilie_Hyland.pdf", nh_img)

    excels = [
        manuscript_card(
            "Testimony_Nilie_Hyland", nh_docx,
            "Testimony of Nilie M. Hyland — witness deposition, 1907",
            "A clean clerk's hand — both tools transcribe it almost perfectly. "
            "Look at the <b>green</b> words: olmOCR and Gemini both read “do” "
            "and “these”, which human review confirmed are <b>correct</b> — "
            "the gold-standard transcription itself has “did” and “those”. On "
            "a clean source the OCR is more faithful to the page than the "
            "transcriber.",
            nh_img),
        fullpage_card(
            "SHNO_Cut_Knife_Journal_1914-01-01_p01",
            "Cut Knife Journal — full front page, 1 January 1914",
            "A whole multi-column newspaper page. On a clean print run Gemini "
            "transcribes the entire page at single-digit error — full-page "
            "OCR is not inherently broken."),
    ]
    struggles = [
        manuscript_card(
            "Charles_Kelly_Excerpt", ck_docx,
            "Charles Kelly — handwritten extradition file, 1858",
            "A dense cursive legal document. Note where Gemini reads "
            "“Her Majesty Victoria” for what the page says — "
            "“Lord Napier, E.E. & M.P.”: a confident, plausible-sounding "
            "hallucination, the most dangerous error type for the historian "
            "because it is not visibly wrong.",
            ck_img),
        manuscript_card(
            "Colonel_Bernard", cb_docx,
            "Colonel Bernard — letter on extradition costs, 1873",
            "Several words the human transcriber marked illegible were read "
            "correctly by Gemini; the recipient address block is written "
            "mid-page, so tools and the gold disagree partly on reading order.",
            cb_img),
        fullpage_card(
            "SHNO_Davidson_Leader_1914-01-01_p02",
            "Davidson Leader — full inside page, 1 January 1914",
            "A densely set, unevenly inked broadsheet page. Compare a model "
            "that reads carefully against one that drifts into invented text "
            "when the print defeats it."),
    ]

    doc = [
        "---",
        'title: "Seeing OCR Errors: Tool Transcriptions of Historical Texts"',
        "format:",
        "  html:",
        "    toc: true",
        "---",
        "",
        CSS,
        "",
        "Rather than summarise OCR quality as a single error rate, this page "
        "shows what each tool actually produced — on documents it handles well "
        "and on documents that defeat it. In every transcription below: "
        "<span style=\"background:#c81e1e;color:#fff;font-weight:700;"
        "padding:0 3px;border-radius:3px\">suspected hallucination</span> "
        "is the dangerous error — a wrong but real, plausible word the reader "
        "cannot catch; "
        "<span style=\"background:#ffd6d6;padding:0 2px;border-radius:3px\">"
        "red</span> is a visibly garbled misread; "
        "<span style=\"background:#ffe9c7;padding:0 2px;border-radius:3px\">"
        "amber</span> a word the tool invented; ~~struck-through~~ a word "
        "it missed; and "
        "<span style=\"background:#a9d8a0;padding:0 2px;border-radius:3px;"
        "font-weight:600\">green</span> marks a word human review confirmed "
        "the OCR got <i>right</i> where the gold standard is wrong. "
        "Suspected hallucinations are flagged automatically (a wrong but real "
        "dictionary word) and still need human confirmation — they cannot be "
        "told apart from gold-standard errors without review.",
        "",
        "*The British Library BLN600 set is used for scoring only and is not "
        "reproduced here.*",
        "",
        "# Where modern OCR excels",
        "",
    ]
    doc.extend(excels)
    doc.append("\n# Where it still struggles\n")
    doc.extend(struggles)

    tdir = "/home/jic823/plato/wpcs-ocr/Transcription/Tables"
    doc.append("\n# Reading tables\n")
    doc.append("Tables are scored by *cell-value recall* — the share of the "
               "table's data values the OCR captured — not CER/WER. Each gold "
               "cell below is shaded by which tool(s) captured its value. "
               "(The Indian Affairs pass registers and the band Whereabouts "
               "Census are used for scoring only and are not reproduced here, "
               "in keeping with OCAP® principles.)\n")
    doc.append(table_card(
        "Canadian_Customs_1897", f"{tdir}/Canadian Customs Department, 1897.xlsx",
        "Canadian Customs Department staff register, 1897",
        "A cleanly printed administrative table — both tools capture nearly "
        "every cell, Gemini almost all of it."))
    doc.append(table_card(
        "NWMP_1880", f"{tdir}/NWMP, 1880.xlsx",
        "North-West Mounted Police distribution state, 1885",
        "A numeric distribution table. Watch which tool holds column "
        "alignment across the rank counts."))
    out = SHOW / "showcase.qmd"
    out.write_text("\n".join(doc))
    print(f"wrote {out}")
    print(f"images in {IMG}")

    # human-review worksheet for the automatically-flagged hallucinations
    review = SHOW / "hallucination_review.csv"
    with review.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["document", "tool", "gold", "ocr",
                                          "context", "verdict"])
        w.writeheader()
        w.writerows(_HALLUC_LOG)
    print(f"wrote {review}  ({len(_HALLUC_LOG)} suspected hallucinations to review)")
    print("  verdict column — fill with: hallucination | gold-error | "
          "misread | alignment-artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
