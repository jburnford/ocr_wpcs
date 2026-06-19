#!/bin/bash
# One-time setup of the Infinity-Parser2 Python venv on a Nibi LOGIN node
# (needs internet for the PyPI packages). Native venv + DRAC wheelhouse — no
# container (the vLLM container base uses dash and breaks `set -o pipefail`).
#
# Validated 2026-06-17. Run once:  bash setup_infinity.sh
set -uo pipefail
BASE=/project/def-jic823/infinity
cd "$BASE"

# opencv MODULE satisfies vLLM's `opencv-noinstall` wheelhouse stub; Infinity
# needs Python >= 3.12 (so we can't reuse the 3.11 chandra venv).
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9
echo "[setup] $(python --version)"

[ -d venv ] || virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip

# Heavy stack from the wheelhouse (vllm pulls torch). Pin transformers 5.3.0
# (Qwen3.5-VL needs 5.x; vLLM's `transformers<5` metadata pin is stale & works).
pip install --no-index "vllm==0.17.1"
pip install --no-index "transformers==5.3.0"
pip install --no-index PyMuPDF              # provides `import fitz`

# infinity_parser2 from PyPI. --no-deps: its pins (scipy>=1.17.1, exact torch/
# vllm) conflict with the wheelhouse builds and cause ResolutionImpossible. We
# already have the heavy deps; add only the two pure-Python pieces it imports.
# (Do NOT `pip install fitz` — that PyPI name is a squatter; PyMuPDF is correct.)
pip install --no-deps infinity_parser2
pip install --no-deps qwen_vl_utils

echo "[setup] verifying imports..."
python -c "import torch,vllm,transformers,infinity_parser2,fitz,cv2; \
print('torch',torch.__version__,'| vllm',vllm.__version__,'| tf',transformers.__version__)"
parser --help >/dev/null && echo "[setup] parser CLI OK"

# The `parser` CLI's own model resolver looks for the model at
# <HF cache>/infly_Infinity-Parser2-Pro (its own name), not the HF
# models--infly--... snapshot. Symlink so the vllm-engine backend works offline.
C=/project/def-jic823/models/huggingface_cache
SNAP=$(ls -d "$C"/models--infly--Infinity-Parser2-Pro/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$SNAP" ] && [ ! -e "$C/infly_Infinity-Parser2-Pro" ]; then
    ln -s "${SNAP%/}" "$C/infly_Infinity-Parser2-Pro"
    echo "[setup] symlinked parser-resolver model path"
fi
echo "[setup] DONE"
