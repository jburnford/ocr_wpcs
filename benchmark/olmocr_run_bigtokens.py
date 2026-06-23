#!/usr/bin/env python3
"""Launcher for olmocr.pipeline that raises the per-page output token cap.

olmocr hardcodes MAX_TOKENS=8000 in build_page_query(); dense full-page scans
(newspapers, tables, manuscripts) generate more than that, hit finish_reason
"length", get marked invalid, and fail after 8 retries -> empty fallback.

This wrapper monkeypatches build_page_query at runtime (no edits to the
installed package) to override query["max_tokens"] with $OLMOCR_MAX_TOKENS,
then hands off to the normal CLI entrypoint. All other args pass through
verbatim (this file is argv[0]; the pipeline reads argv[1:]).
"""
import os
import olmocr.pipeline as P

CAP = int(os.environ.get("OLMOCR_MAX_TOKENS", "16000"))
_orig_build_page_query = P.build_page_query


async def build_page_query(*args, **kwargs):
    query = await _orig_build_page_query(*args, **kwargs)
    query["max_tokens"] = CAP
    return query


# pipeline.py calls build_page_query by bare global name (line ~170), so
# rebinding the module attribute is sufficient.
P.build_page_query = build_page_query

if __name__ == "__main__":
    print(f"[bigtokens launcher] overriding per-page max_tokens -> {CAP}", flush=True)
    P.cli_main()
