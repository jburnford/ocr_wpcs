#!/bin/bash
# Pre-download Infinity-Parser2-Pro weights on a Nibi LOGIN node (internet).
# Compute nodes are offline, so the model must be cached in /project first.
# Validated 2026-06-17 (~66 GB, resumable).  Usage:  bash download_model.sh
set -uo pipefail

export HF_HOME=/project/def-jic823/models/huggingface_cache
mkdir -p "$HF_HOME"
module load StdEnv/2023 python/3.12 >/dev/null 2>&1

# NOTE: do NOT enable HF_HUB_ENABLE_HF_TRANSFER — the hf_transfer package isn't
# in Nibi's huggingface-hub build and it crashes the download.
HF_HUB_ENABLE_HF_TRANSFER=0 python3 - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download("infly/Infinity-Parser2-Pro", cache_dir=os.environ["HF_HOME"])
print("DONE:", p)
PY
# model lands at $HF_HOME/models--infly--Infinity-Parser2-Pro (no hub/ subdir);
# run scripts set HF_HUB_CACHE=$HF_HOME so vLLM finds it offline.
du -sh "$HF_HOME/models--infly--Infinity-Parser2-Pro" 2>/dev/null || true
