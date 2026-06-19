# Infinity Parser 2 on Nibi (DRAC / SHARCNET)

Run Infinity-Parser2-**Pro** (35B, the variant that won our OCR benchmark) on
Nibi's H100 nodes. **Validated 2026-06-17**: Pro runs on a **single H100 via FP8
quantization**, reproducing the local BF16 output (100% on a clean page, 98.9%
token recall on a hard multi-page doc).

## Approach: native venv, NOT a container

The proven pattern (from the existing `chandra2` setup) is a Python 3.12 venv +
the DRAC wheelhouse — vLLM 0.17.1 is in the wheelhouse. A custom Apptainer image
was tried and dropped (the vLLM base image's `dash` shell breaks `%post`).

Verified env: AlmaLinux 9.6, x86_64, 8× H100-80GB/node, SLURM, Lmod, account
`def-jic823`. Login nodes have internet; **compute nodes do not** (pre-stage the
model). `/home` is ~94% full, so caches go to `/scratch`.

## Two ways to run Pro

| | FP8, 1× H100 *(recommended)* | TP=2 BF16, 2× H100 |
|---|---|---|
| Script | `run_infinity_fp8_1gpu.slurm` | `run_infinity.slurm` |
| Mechanism | `vllm serve --quantization fp8` + parser vllm-server client | `parser --backend vllm-engine --tensor-parallel-size 2` |
| Quality | ~99–100% of BF16 | exact |
| Throughput | 8 jobs/node, faster backfill | 4 jobs/node |

FP8 is the scaling choice. TP=2 is for bit-exact benchmark reproduction.

## Workflow (run from `~/projects/def-jic823/infinity/` on Nibi)

```bash
bash setup_infinity.sh          # 1. one-time: build the venv (login node)
bash download_model.sh          # 2. one-time: cache the 66 GB model (login node)
# put PDFs/images in ./input/ (or pass a dir), then:
sbatch --time=00:45:00 run_infinity_fp8_1gpu.slurm [INPUT_DIR] [OUTPUT_DIR]
```

Output is per-document `…/<name>.pdf/result.json` in the block-list schema
`[{bbox,category,text}]` — copy into `wpcs-ocr/infinity_output/<dataset>/` and
re-run the benchmark harness, or use directly downstream.

Use a **45-minute walltime** for fast GPU backfill; raise it for large volumes
(a 335-page document is fine on one H100).

## Gotchas (all handled in the scripts)

- `infinity_parser2` needs **Python ≥3.12**; install it `--no-deps` (its pins
  cause `ResolutionImpossible`) and add `qwen_vl_utils` + **PyMuPDF** (never the
  `fitz` PyPI squatter).
- Load the **opencv module** so vLLM's `opencv-noinstall` stub is satisfied.
- Set **`HF_HUB_CACHE=$HF_HOME`** — the model is cached at `$HF_HOME/models--…`
  with no `hub/` subdir, so default lookups miss it offline.
- The FP8 client hardcodes a **32768-token output cap**, so the server needs
  `--max-model-len 65536` (room for that + the image/text input).
- The `parser` CLI's own resolver wants the model at
  `…/huggingface_cache/infly_Infinity-Parser2-Pro` — `setup_infinity.sh`
  symlinks it to the HF snapshot.

## Files

| File | Purpose |
|---|---|
| `setup_infinity.sh` | One-time venv build (login node) |
| `download_model.sh` | One-time model pre-download (login node) |
| `run_infinity_fp8_1gpu.slurm` | **FP8, 1× H100** — serve+client (recommended) |
| `run_infinity.slurm` | TP=2 BF16, 2× H100 — parser vllm-engine |
