#!/usr/bin/env python3
"""Convert BLN600 source images to single-page PDFs for OCR.

Source images live in the extracted BLN600 dataset (546 .tif + 54 .jpg).
Each becomes <basename>.pdf in the output dir. img2pdf is lossless; if it
rejects an image (unusual TIFF compression/mode) we fall back to Pillow:
re-encode to RGB JPEG in memory, then wrap with img2pdf.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import img2pdf
from PIL import Image

SRC = Path("/home/jic823/ocr_bldata/25439023/BLN600/Images")
OUT = Path("/home/jic823/plato/wpcs-ocr/bln600_pdfs")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in SRC.iterdir()
        if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg")
    )
    print(f"Found {len(images)} source images", file=sys.stderr)
    ok = 0
    fallback = 0
    failed: list[str] = []
    for img in images:
        dst = OUT / f"{img.stem}.pdf"
        try:
            pdf_bytes = img2pdf.convert(str(img))
        except Exception:
            try:
                with Image.open(img) as im:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="JPEG", quality=95)
                    buf.seek(0)
                    pdf_bytes = img2pdf.convert(buf.read())
                fallback += 1
            except Exception as e:
                failed.append(f"{img.name}: {e}")
                continue
        dst.write_bytes(pdf_bytes)
        ok += 1
    print(f"Converted: {ok}  (img2pdf direct: {ok - fallback}, Pillow fallback: {fallback})",
          file=sys.stderr)
    if failed:
        print(f"FAILED {len(failed)}:", file=sys.stderr)
        for f in failed:
            print("  " + f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
