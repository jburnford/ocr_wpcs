#!/usr/bin/env python3
"""Run GLM-OCR over a corpus in the ocr_benchmark data repo.

The ocr_benchmark repo (git.cs.usask.ca:history-graphrag/ocr_benchmark) is the
data-of-record: each corpus has <corpus>/images/ (page images or source PDFs)
and gold under <corpus>/gold/ + <corpus>/gold_json/. This runner reads the
images, OCRs each with the local GLM-OCR vLLM endpoint, and writes one
{"text": ...} record per doc (keyed by the image stem == gold doc_id), matching
the gemini tool format so the project repo's ocr_to_canonical.py adapter can
convert + score it.

Handles both raster images (jpg/jpeg/png/tif/tiff/bmp) and PDFs (rasterized
page-by-page, pages concatenated). Resumable; already-done stems are skipped.

Usage:
  python3 run_glmocr_corpus.py <corpus> [<corpus> ...]
  e.g. python3 run_glmocr_corpus.py jacob

Env:
  OCR_BENCH_ROOT   data repo root (default /home/jic823/Documents/ocr_benchmark)
  GLMOCR_OUT_ROOT  output root     (default <wpcs-ocr>/benchmark/ocr_output)
  GLMOCR_DPI       PDF raster DPI  (default 200)
  (GLMOCR_ENDPOINT / GLMOCR_MODEL inherited from run_glmocr)
"""
from __future__ import annotations
import base64
import json
import os
import sys
import time
from pathlib import Path

import run_glmocr as G  # reuse the endpoint, prompt, and _ocr_image / _page_images

BENCH_ROOT = Path(os.environ.get(
    "OCR_BENCH_ROOT", "/home/jic823/Documents/ocr_benchmark"))
OUT_ROOT = Path(os.environ.get(
    "GLMOCR_OUT_ROOT", str(G.ROOT / "benchmark" / "ocr_output")))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _image_to_uri(path: Path) -> str:
    """Base64 data URI for a raster image, passed to the model as-is."""
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def transcribe_doc(path: Path) -> tuple[str, list[int]]:
    """OCR one input file (image or PDF). Return (text, capped_pages)."""
    if path.suffix.lower() == ".pdf":
        uris = G._page_images(path)
    else:
        uris = [_image_to_uri(path)]
    out, capped = [], []
    for i, uri in enumerate(uris):
        text, finish = G._ocr_image(uri)
        out.append(text)
        if finish == "length":
            capped.append(i + 1)
    return "\n\n".join(out).strip(), capped


def run_corpus(corpus: str) -> None:
    img_dir = BENCH_ROOT / corpus / "images"
    if not img_dir.is_dir():
        print(f"[{corpus}] no images dir at {img_dir}", file=sys.stderr)
        return
    out_dir = OUT_ROOT / f"glmocr_{corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".pdf")
    todo = [p for p in inputs if not (out_dir / f"{p.stem}.json").exists()]
    print(f"[{corpus}] {len(inputs)} inputs, {len(todo)} to do "
          f"({len(inputs) - len(todo)} cached)", file=sys.stderr)

    done = errors = 0
    capped_docs: dict[str, list[int]] = {}
    t0 = time.monotonic()
    for path in todo:
        try:
            text, capped = transcribe_doc(path)
            (out_dir / f"{path.stem}.json").write_text(
                json.dumps({"text": text}, ensure_ascii=False))
            done += 1
            if capped:
                capped_docs[path.stem] = capped
                print(f"  [warn] {path.name}: pages hit token cap: {capped}",
                      file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERROR {path.name}: {e}", file=sys.stderr)
        if (done + errors) % 10 == 0:
            rate = (done + errors) / (time.monotonic() - t0) * 3600
            print(f"  [{corpus}] {done + errors}/{len(todo)} "
                  f"({rate:.0f} docs/hr)", file=sys.stderr)

    cap_file = out_dir / "_capped_pages.json"
    prior = json.loads(cap_file.read_text()) if cap_file.exists() else {}
    prior.update(capped_docs)
    cap_file.write_text(json.dumps(prior, indent=2))
    print(f"[{corpus}] done: {done}  errors: {errors}  "
          f"capped_docs: {len(capped_docs)}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for corpus in sys.argv[1:]:
        run_corpus(corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
