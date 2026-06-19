---
name: cluster-vlm-ocr
description: >
  Run modern vision-language OCR models — Chandra 2 (datalab-to/chandra-ocr-2,
  ~9B) and Infinity Parser 2 Pro (infly/Infinity-Parser2-Pro, ~35B) — on a
  SLURM HPC cluster with H100 GPUs, fully offline on compute nodes. Use when a
  scholar wants to reproduce the WPCS OCR benchmark, or to OCR a collection of
  historical PDFs/page images at scale on shared research computing
  (DRAC/Alliance, SHARCNET, or any Lmod + SLURM + Apptainer site).
---

# Running Chandra 2 and Infinity Parser 2 on an HPC cluster

This is a reproducibility guide (and WPCS paper appendix) for getting two
state-of-the-art VLM OCR models running on a research cluster. Both are driven
through **vLLM**; the recipes below were validated on the Digital Research
Alliance of Canada "Nibi" cluster (AlmaLinux 9.6, 8× H100-80GB per node, SLURM,
Lmod, internet on login nodes only) but call out every site-specific value so
you can port them.

If your site differs, the four things you must localize are: the **SLURM
account**, the **module names** (`python`, `gcc`, `cuda`, `opencv`), the
**project/scratch paths**, and whether **compute nodes have internet** (most
don't — so you pre-stage models on a login node).

---

## 0. Mental model — why a vLLM server + a thin client

Both tools wrap a large multimodal model. The robust pattern is:

1. **Pre-stage the model** to a shared filesystem on a login node (internet),
   because compute nodes are usually offline.
2. In the SLURM job, **start a `vllm serve` process** in the background on a
   random free port, then **poll it until it answers a real request**.
3. Run the tool's **CLI as a client** against that local server.

Starting your own server (rather than letting the CLI spawn one) is what lets
you control quantization, context length, and GPU memory — and is the only way
to fit the 35B Infinity model on a single 80 GB H100 (see FP8 below).

---

## 1. Shared prerequisites

- **GPU**: an H100 (or A100-80GB). Compute-capability ≥ 8.0. Chandra fits
  comfortably; Infinity Pro needs FP8 to fit one card (or 2 cards in BF16).
- **Software stack**: vLLM `0.17.1` from your cluster's wheelhouse (or pip),
  CUDA 12.x, Python ≥ 3.11 (Chandra) / **≥ 3.12** (Infinity).
- **Disk**: home directories are often near-full; put HuggingFace caches and
  vLLM/Triton/FlashInfer JIT caches on **`/scratch`**, not `/home`.
- **Offline flags**: once models are staged, set `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` in jobs so nothing tries to phone home.

```bash
# put these in every job; redirect JIT/compile caches off the full /home
export XDG_CACHE_HOME=/scratch/$USER/.cache
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME/flashinfer
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
```

---

## 2. Chandra 2  (`datalab-to/chandra-ocr-2`)

Chandra is the simpler of the two: a single `pip install chandra-ocr`, a `vllm
serve`, and the `chandra` CLI. It emits Markdown + HTML + a metadata JSON per
input document.

### 2.1 One-time setup (login node, has internet)

```bash
BASE=/project/<acct>/chandra2
export HF_HOME=/scratch/$USER/hf_cache        # model cache on scratch
mkdir -p "$HF_HOME"; cd "$BASE"
module load python/3.11 gcc opencv/4.13.0     # opencv module satisfies vLLM's opencv stub
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index "vllm==0.17.1"         # from the wheelhouse; drop --no-index off-Alliance
pip install "chandra-ocr"
# pre-download the weights so compute nodes can run offline:
hf download datalab-to/chandra-ocr-2 || huggingface-cli download datalab-to/chandra-ocr-2
```

### 2.2 SLURM run script

```bash
#!/bin/bash
#SBATCH --account=<acct>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -uo pipefail
BASE=/project/<acct>/chandra2
PDFDIR=$1; OUTDIR=$2                  # dir of PDFs/images; output dir
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/$USER/.cache VLLM_CACHE_ROOT=/scratch/$USER/.cache/vllm
mkdir -p "$OUTDIR" "$VLLM_CACHE_ROOT"
module load python/3.11 gcc opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"

PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
export VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
vllm serve datalab-to/chandra-ocr-2 \
  --host 127.0.0.1 --port "$PORT" --served-model-name chandra \
  --dtype bfloat16 --max-model-len 18000 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --trust-remote-code \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":6291456}' &
VLLM_PID=$!

# wait until the server actually answers (not just that the port is open)
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "server died"; exit 1; }
  sleep 10
done

# one server, looped client — process every document, skip ones already done
for f in "$PDFDIR"/*.pdf; do
  stem=$(basename "$f" .pdf)
  [ -s "$OUTDIR/$stem/$stem.md" ] && continue
  chandra "$f" "$OUTDIR" --method vllm || echo "FAIL: $stem"
done
kill $VLLM_PID
```

**Output**: `<OUTDIR>/<stem>/<stem>.md` (plus `.html` and `_metadata.json`).
The `.md` is the transcription.

---

## 3. Infinity Parser 2 Pro  (`infly/Infinity-Parser2-Pro`)

The strongest model in our benchmark, but a 35B parameter model and fussier to
install. Use a **native venv + wheelhouse, not a container** (the vLLM base
image's `dash` shell breaks `set -o pipefail` in `%post`).

### 3.1 FP8 is the key to a single H100

BF16 weights are ~70 GB — they fit on an 80 GB card but leave no room for the
KV cache. **`vllm serve --quantization fp8`** halves the weights to ~35 GB and
runs comfortably on **one** H100 at ~99–100 % of BF16 quality. (For bit-exact
benchmark reproduction instead, use `--tensor-parallel-size 2` in BF16 across
two cards.)

Infinity has no `--quantization` flag on its own CLI, so you **must** run your
own `vllm serve` and point the `parser` client at it with
`--backend vllm-server`.

### 3.2 One-time setup (login node) — and its non-obvious gotchas

```bash
BASE=/project/<acct>/infinity; cd "$BASE"
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9   # Python >=3.12 required
virtualenv --no-download venv && source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index "vllm==0.17.1"
pip install --no-index "transformers==5.3.0"   # Qwen3.5-VL needs tf 5.x
pip install --no-index PyMuPDF                  # provides `import fitz` — NEVER `pip install fitz` (squatter)
pip install --no-deps infinity_parser2          # --no-deps: its pins cause ResolutionImpossible
pip install --no-deps qwen_vl_utils

# stage the ~66 GB model (do NOT set HF_HUB_ENABLE_HF_TRANSFER on Alliance):
export HF_HOME=/project/<acct>/models/huggingface_cache; mkdir -p "$HF_HOME"
HF_HUB_ENABLE_HF_TRANSFER=0 python -c \
 'from huggingface_hub import snapshot_download; snapshot_download("infly/Infinity-Parser2-Pro", cache_dir="'"$HF_HOME"'")'

# the parser CLI's resolver looks for the model under its OWN name; symlink it:
C=$HF_HOME; SNAP=$(ls -d "$C"/models--infly--Infinity-Parser2-Pro/snapshots/*/ | head -1)
ln -sf "${SNAP%/}" "$C/infly_Infinity-Parser2-Pro"
```

Gotchas, all real and all bite silently:
- **Python ≥ 3.12**, install `infinity_parser2` with `--no-deps` (its exact
  torch/vllm/scipy pins are unsatisfiable against wheelhouse builds).
- Load the **opencv module** so vLLM's `opencv-noinstall` stub is satisfied.
- Set **`HF_HUB_CACHE=$HF_HOME`** in jobs — the model caches at
  `$HF_HOME/models--…` with no `hub/` subdir, so the default offline lookup misses it.
- The FP8 client hardcodes a **32768-token output cap**, so the server needs
  **`--max-model-len 65536`** (room for that plus the image/text input).

### 3.3 SLURM run script (FP8, 1× H100)

```bash
#!/bin/bash
#SBATCH --account=<acct>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -uo pipefail
BASE=/project/<acct>/infinity; PDFDIR=$1; OUTDIR=$2; mkdir -p "$OUTDIR"
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"
export HF_HOME=/project/<acct>/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/$USER/.cache/inf VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME/flashinfer
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
vllm serve infly/Infinity-Parser2-Pro \
  --host 127.0.0.1 --port "$PORT" --served-model-name infinity-pro \
  --quantization fp8 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --trust-remote-code \
  --enable-prefix-caching \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":12845056}' &
VLLM_PID=$!

# warm-up probe must POST a REAL request (the client's 5s connection ping
# otherwise times out on a cold engine):
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST \
    "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"infinity-pro","messages":[{"role":"user","content":"ping"}],"max_tokens":1}')
  [ "$code" = "200" ] && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "server died"; exit 1; }
  sleep 10
done

parser "$PDFDIR"/*.pdf -o "$OUTDIR" \
  --task doc2json --output-format json \
  --backend vllm-server --api-url "http://127.0.0.1:${PORT}/v1" --api-key EMPTY \
  --model-name infinity-pro --batch-size 16 --verbose
kill $VLLM_PID
```

**Output**: `<OUTDIR>/<stem>.pdf/result.json` per document, in Infinity's
block-list schema `[[{bbox,category,text}, …]]` (a list of pages, each a list
of reading-order blocks).

### 3.4 Repair the JSON before you use it

Infinity's `doc2json` model emits **invalid JSON** — it does not escape `"` and
`\` inside the free-text `text` field. A plain `json.loads` fails. But the
block schema is rigid (`bbox` and `category` are always clean), so a
schema-aware recovery re-parses every block losslessly. Run a recovery pass
over the output directory before scoring or downstream use. The reference
implementation is `recover_infinity_json.py` in the benchmark repo:

```bash
python recover_infinity_json.py <OUTDIR> --write --pdfdir <PDFDIR>
# dry-run (no --write) validates page-count fidelity vs the source PDFs
```

---

## 4. Verify it worked

```bash
# Chandra: every input has a non-empty markdown transcription
find <OUTDIR> -name '*.md' -size +0c | wc -l

# Infinity: every input produced a result.json that parses after recovery
find <OUTDIR> -name result.json | wc -l
```

Spot-check one transcription against the page image. For a quantitative check,
score against gold with CER/WER (strict + semantic), BLEU, and a hallucination
rate — see the benchmark harness and `CHUNK_EVAL_METHOD.md` for the
order-invariant, chunk-aware scoring used in the paper.

---

## 5. Porting to a non-Alliance cluster

| Alliance-specific thing | Generic equivalent |
|---|---|
| `--account=<acct>` | your SLURM allocation |
| `module load … cuda/12.9` | `module load cuda` or a CUDA-enabled conda env |
| `pip install --no-index` (wheelhouse) | drop `--no-index`; install from PyPI |
| `opencv` module | `pip install opencv-python-headless` |
| offline compute nodes | if your nodes have internet, skip pre-staging + the OFFLINE flags |
| `/project`, `/scratch` | any shared (model cache) + fast-scratch (JIT cache) paths |

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ResolutionImpossible` installing `infinity_parser2` | install it `--no-deps`; provide vllm/transformers/PyMuPDF/qwen_vl_utils yourself |
| `import fitz` fails or behaves oddly | you installed the `fitz` squatter; `pip uninstall fitz && pip install PyMuPDF` |
| vLLM can't find the model offline | set `HF_HUB_CACHE=$HF_HOME`; for Infinity's CLI also symlink `infly_Infinity-Parser2-Pro` |
| Infinity OOMs / no KV cache on one H100 | you're in BF16 — add `--quantization fp8` (or use `-tp 2`) |
| Client times out connecting to a cold server | warm up with a real POST to `/v1/chat/completions`, not just `/v1/models` |
| Output truncated mid-document (Infinity) | raise `--max-model-len` (≥ 65536; the client caps output at 32768 tokens) |
| `json.loads` fails on Infinity output | expected — run the schema-aware recovery pass first |
| caches fill up `/home` and jobs fail | point `XDG_CACHE_HOME`/`VLLM_CACHE_ROOT`/`HF_HOME` at `/scratch` |
| running olmOCR's container picks up host packages | isolate with `apptainer exec --cleanenv` **and** `PYTHONNOUSERSITE=1` |
