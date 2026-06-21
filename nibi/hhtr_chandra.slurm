#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --job-name=hhtr-chandra
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/def-jic823/hhtr_bench/logs/chandra-%j.out

# Chandra 2 over the 50 single-page HHTR (handwritten historical) PDFs. One vLLM server, looped client.
# Mirrors chandra2/run_chandra.sh (the recipe that produced the BLN600 results).
# Output: hhtr_bench/chandra/<stem>/<stem>.md  (matches load_chandra_md)
set -uo pipefail
BASE=/project/def-jic823/chandra2
PDFDIR=/project/def-jic823/hhtr_bench/pdfs
OUTDIR=/project/def-jic823/hhtr_bench/chandra
mkdir -p "$OUTDIR" /project/def-jic823/hhtr_bench/logs

export HF_HOME=/scratch/jic823/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/jic823/.cache
export VLLM_CACHE_ROOT=/scratch/jic823/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=/scratch/jic823/.cache/flashinfer
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
module load python/3.11 gcc opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"

SLOG="/project/def-jic823/hhtr_bench/logs/chandra-server-${SLURM_JOB_ID}.log"
PORT=$(python -c "import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()")
export VLLM_API_BASE="http://127.0.0.1:${PORT}/v1"
echo "[job] node=$(hostname) port=$PORT"; nvidia-smi -L

vllm serve datalab-to/chandra-ocr-2 \
  --host 127.0.0.1 --port "$PORT" --served-model-name chandra \
  --dtype bfloat16 --max-model-len 18000 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --trust-remote-code \
  --mm-processor-kwargs "{\"min_pixels\":3136,\"max_pixels\":6291456}" \
  > "$SLOG" 2>&1 &
VLLM_PID=$!

echo "[job] waiting for server readiness..."
READY=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then READY=1; echo "[job] server ready after ${i}0s"; break; fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "[job] SERVER DIED"; tail -60 "$SLOG"; exit 1; }
  sleep 10
done
[ $READY -eq 1 ] || { echo "[job] server not ready"; tail -80 "$SLOG"; kill $VLLM_PID; exit 1; }

n=0; ok=0; fail=0
for pdf in "$PDFDIR"/*.pdf; do
  n=$((n+1))
  stem=$(basename "$pdf" .pdf)
  if [ -s "$OUTDIR/$stem/$stem.md" ]; then ok=$((ok+1)); continue; fi
  echo "[$n/100] $stem"
  if chandra "$pdf" "$OUTDIR" --method vllm; then ok=$((ok+1)); else echo "  FAIL: $stem"; fail=$((fail+1)); fi
done
echo "[job] done: ok=$ok fail=$fail of $n; stopping server"
kill $VLLM_PID 2>/dev/null; sleep 3
echo "[job] md files: $(find "$OUTDIR" -name "*.md" | wc -l)"
exit 0
