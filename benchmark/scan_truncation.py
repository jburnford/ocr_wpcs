#!/usr/bin/env python3
"""Flag Gemini output files whose transcription looks truncated.

Heuristic: a complete transcription ends with sentence-final punctuation or a
closing bracket/quote. One ending mid-word/mid-sentence is likely a truncated
(MAX_TOKENS) response. Deletes flagged files when run with --delete so they
are re-done on the next run_gemini pass.
"""
import glob
import json
import sys
from pathlib import Path

ENDERS = set('.!?"”’)]}>')


def main() -> int:
    delete = "--delete" in sys.argv
    for ds in ["fullpage", "manuscripts", "tables", "bln600"]:
        files = sorted(glob.glob(f"ocr_output/gemini_{ds}/*.json"))
        bad = []
        for f in files:
            t = json.load(open(f))["text"].rstrip()
            if t and t[-1] not in ENDERS:
                bad.append(f)
        print(f"{ds}: {len(bad)}/{len(files)} suspect")
        for f in bad:
            t = json.load(open(f))["text"].rstrip()
            print(f"   {Path(f).name}  ...{t[-55:]!r}")
            if delete:
                Path(f).unlink()
        if delete and bad:
            print(f"   deleted {len(bad)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
