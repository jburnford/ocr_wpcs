#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --job-name=infinity-ocr
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/project/def-jic823/infinity/logs/job-%j.out

# Infinity-Parser2-Pro (35B) over a dir of PDFs/images on Nibi.
# Native venv + wheelhouse vLLM (no container) — mirrors the chandra2 recipe.
# Pro is 35B -> 2x H100 with --tensor-parallel-size 2.
#   Usage: sbatch run_infinity.slurm [INPUT_DIR] [OUTPUT_DIR]
set -uo pipefail
BASE=/project/def-jic823/infinity
INPUT="${1:-$BASE/input}"
OUTDIR="${2:-$BASE/output/$SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$BASE/logs"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"

# model lives in the project HF cache; compute nodes are offline.
export HF_HOME=/project/def-jic823/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME          # model lives at $HF_HOME/models--... (no hub/ subdir)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# JIT/compile caches off the nearly-full /home, onto /scratch (chandra pattern).
export XDG_CACHE_HOME=/scratch/jic823/.cache
export VLLM_CACHE_ROOT=/scratch/jic823/.cache/vllm
export TRITON_CACHE_DIR=/scratch/jic823/.cache/triton
export FLASHINFER_WORKSPACE_BASE=/scratch/jic823/.cache/flashinfer
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi -L
echo "[job] input=$INPUT  out=$OUTDIR"

parser "$INPUT" \
  -o "$OUTDIR" \
  --task doc2json \
  --output-format json \
  --backend vllm-engine \
  --model-name infly/Infinity-Parser2-Pro \
  --model-cache-dir "$HF_HOME" \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --verbose

echo "[job] parser exit=$? at $(date); output in $OUTDIR"
ls "$OUTDIR" | head
