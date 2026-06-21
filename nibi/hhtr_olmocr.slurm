#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --job-name=hhtr-olmocr
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/def-jic823/hhtr_bench/logs/olmocr-%j.out

# olmOCR over the 50 single-page HHTR (handwritten historical) PDFs, via the olmocr.sif container.
# The container's olmocr (--pdfs/--model/--markdown) is shadowed by the host's
# ~/.local CC wheels unless we isolate it: --cleanenv + PYTHONNOUSERSITE=1.
# Model: allenai/olmOCR-7B-0825-FP8 (cached at ~/.cache/huggingface/hub),
# passed as an absolute snapshot path so no network/HF-cache resolution is needed.
# Output: hhtr_bench/olmocr/results/output_*.jsonl  (Dolma; Source-File=<stem>.pdf)
set -uo pipefail
SIF=/project/def-jic823/olmocr/olmocr.sif
PDFDIR=/project/def-jic823/hhtr_bench/pdfs
WS=/project/def-jic823/hhtr_bench/olmocr
mkdir -p "$WS" /project/def-jic823/hhtr_bench/logs

module load apptainer/1.4.5

# resolve the local model snapshot dir (no hardcoded hash)
MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--allenai--olmOCR-7B-0825-FP8/snapshots/*/ 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "[job] model snapshot not found"; exit 1; }
echo "[job] $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi -L
echo "[job] model=$MODEL"
echo "[job] pdfs=$(ls "$PDFDIR"/*.pdf | wc -l)"

# isolate container python from host ~/.local; pin caches to /scratch
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_XDG_CACHE_HOME=/scratch/jic823/.cache/hhtr_olmocr
export APPTAINERENV_VLLM_CACHE_ROOT=/scratch/jic823/.cache/hhtr_olmocr/vllm
mkdir -p /scratch/jic823/.cache/hhtr_olmocr/vllm

apptainer exec --nv --cleanenv \
  -B /project -B /scratch -B "$HOME/.cache" \
  "$SIF" \
  python -m olmocr.pipeline "$WS" \
    --markdown \
    --pdfs "$PDFDIR"/*.pdf \
    --model "$MODEL" \
    --gpu-memory-utilization 0.85 \
    --max_model_len 16384 \
    --tensor-parallel-size 1
RC=$?
echo "[job] olmocr exit=$RC at $(date)"
echo "[job] jsonl: $(find "$WS" -name 'output_*.jsonl' | wc -l); records: $(cat "$WS"/results/output_*.jsonl 2>/dev/null | wc -l)"
exit $RC
