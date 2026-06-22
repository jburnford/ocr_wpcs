#!/usr/bin/env python3
"""Run GLM-OCR (zai-org/GLM-OCR) over a benchmark dataset.

GLM-OCR is a lightweight 0.9B image-text-to-text document OCR model. It is
served locally on the GB10 via vLLM (OpenAI-compatible API) and called here
one page-image at a time with the model's native "Text Recognition:" prompt.
Each input PDF is rasterized page-by-page with PyMuPDF; the per-page model
outputs are concatenated in reading order into a single whole-document
transcription, matching the {"text": ...} record the other tools emit.

One <stem>.json is written per input PDF; already-done files are skipped so the
run is resumable. Mirrors run_gemini.py's contract so run_eval.py can score it
as a new tool "glmocr" (load it exactly like gemini: {"text": ...}).

Usage:
  python3 run_glmocr.py <dataset> [<dataset> ...]
  dataset in: bln600 | manuscripts | tables | fullpage | all

Env:
  GLMOCR_ENDPOINT  OpenAI-compatible base URL (default http://localhost:8010/v1)
  GLMOCR_MODEL     served model name           (default glm-ocr)
  GLMOCR_DPI       rasterization DPI            (default 200)
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

# Local repo root on the GB10 inference box (the cluster paths in run_eval.py /
# run_gemini.py do not apply here; this is the inference side).
ROOT = Path("/home/jic823/Documents/infinity/wpcs-ocr")
OUT = ROOT / "benchmark" / "ocr_output"

ENDPOINT = os.environ.get("GLMOCR_ENDPOINT", "http://localhost:8010/v1").rstrip("/")
MODEL = os.environ.get("GLMOCR_MODEL", "glm-ocr")
DPI = int(os.environ.get("GLMOCR_DPI", "200"))

# GLM-OCR's native document-parsing instruction. The same prompt is used for
# every dataset: GLM-OCR is purpose-trained for verbatim document OCR and emits
# markdown (incl. table markup) without per-task prompting.
PROMPT = "Text Recognition:"

# dataset -> pdf directory
DATASETS = {
    "bln600": ROOT / "bln600_pdfs",
    "manuscripts": ROOT / "manuscript_pdfs",
    "tables": ROOT / "table_pdfs",
    "fullpage": ROOT / "fullpage_pdfs",
}


def _page_images(pdf: Path, dpi: int = DPI) -> list[str]:
    """Rasterize each page to a base64 PNG data URI (reading order)."""
    uris: list[str] = []
    doc = fitz.open(pdf)
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png = pix.tobytes("png")
            b64 = base64.b64encode(png).decode("ascii")
            uris.append(f"data:image/png;base64,{b64}")
    finally:
        doc.close()
    return uris


def _ocr_image(data_uri: str, retries: int = 4) -> tuple[str, str]:
    """Send one page image to GLM-OCR; return (text, finish_reason).

    Sampling follows the model's shipped generation_config (greedy) and the
    vendor's documented max_new_tokens=8192. Note: on extremely dense full
    pages a 0.9B model can lose its place and re-transcribe a passage,
    exhausting the budget; that surfaces here as finish_reason == "length".
    """
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{ENDPOINT}/chat/completions", json=payload, timeout=600
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choice = data["choices"][0]
            finish = choice.get("finish_reason") or ""
            text = (choice["message"].get("content") or "").strip()
            return text, finish
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt + 1)
    raise RuntimeError(f"page failed after {retries} tries: {last_err}")


def transcribe(pdf: Path) -> tuple[str, list[int]]:
    """OCR every page of a PDF. Return (joined_text, [1-indexed pages that hit
    the token cap])."""
    pages = _page_images(pdf)
    out: list[str] = []
    capped: list[int] = []
    for i, uri in enumerate(pages):
        text, finish = _ocr_image(uri)
        out.append(text)
        if finish == "length":
            capped.append(i + 1)
    return "\n\n".join(out).strip(), capped


def run_dataset(name: str) -> None:
    pdf_dir = DATASETS[name]
    out_dir = OUT / f"glmocr_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    todo = [p for p in pdfs if not (out_dir / f"{p.stem}.json").exists()]
    print(f"[{name}] {len(pdfs)} PDFs, {len(todo)} to do "
          f"({len(pdfs) - len(todo)} cached)", file=sys.stderr)

    done = errors = 0
    capped_docs: dict[str, list[int]] = {}
    t0 = time.monotonic()
    for pdf in todo:
        try:
            text, capped = transcribe(pdf)
            (out_dir / f"{pdf.stem}.json").write_text(
                json.dumps({"text": text}, ensure_ascii=False))
            done += 1
            if capped:
                capped_docs[pdf.stem] = capped
                print(f"  [warn] {pdf.name}: pages hit token cap: {capped}",
                      file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERROR {pdf.name}: {e}", file=sys.stderr)
        if (done + errors) % 10 == 0:
            rate = (done + errors) / (time.monotonic() - t0) * 3600
            print(f"  [{name}] {done + errors}/{len(todo)} "
                  f"({rate:.0f} docs/hr)", file=sys.stderr)
    # Record which docs hit the cap (merge with any prior run's record).
    cap_file = out_dir / "_capped_pages.json"
    prior = {}
    if cap_file.exists():
        prior = json.loads(cap_file.read_text())
    prior.update(capped_docs)
    cap_file.write_text(json.dumps(prior, indent=2))
    print(f"[{name}] done: {done}  errors: {errors}  "
          f"capped_docs: {len(capped_docs)}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    targets = sys.argv[1:]
    if targets == ["all"]:
        targets = list(DATASETS)
    for name in targets:
        if name not in DATASETS:
            print(f"unknown dataset: {name}", file=sys.stderr)
            return 1
    for name in targets:
        run_dataset(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
