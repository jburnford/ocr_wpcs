#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --job-name=inf-prod
#SBATCH --array=0-7
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/project/def-jic823/infinity/logs/prod-%A_%a.out

# Production: parse a page-balanced shard of PDFs with Infinity-Parser2-Pro
# (FP8, 1 H100) — one model load per array task, then all PDFs in the shard.
# Shards are built by make_shards.sh into shards/shard_NN.txt (one PDF path/line).
set -uo pipefail
BASE=/project/def-jic823/infinity
SHARD=$(printf "%s/shards/shard_%02d.txt" "$BASE" "$SLURM_ARRAY_TASK_ID")
OUTDIR="$BASE/output/production"
mkdir -p "$OUTDIR" "$BASE/logs"
[ -s "$SHARD" ] || { echo "empty/missing shard: $SHARD"; exit 0; }
mapfile -t FILES < "$SHARD"
echo "[task $SLURM_ARRAY_TASK_ID] $(hostname) $(date) — ${#FILES[@]} PDFs"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"

export HF_HOME=/project/def-jic823/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# per-task private caches on /scratch (avoid cross-task collisions + full /home)
export XDG_CACHE_HOME=/scratch/jic823/.cache/prod_${SLURM_ARRAY_TASK_ID}
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME/flashinfer
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

nvidia-smi -L
PORT=$(python -c "import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()")
SLOG="$BASE/logs/prod-server-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
echo "[task] FP8 vLLM server on $PORT -> $SLOG"

vllm serve infly/Infinity-Parser2-Pro \
  --host 127.0.0.1 --port "$PORT" --served-model-name infinity-pro \
  --quantization fp8 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --trust-remote-code \
  --enable-prefix-caching \
  --mm-processor-kwargs "{\"min_pixels\":3136,\"max_pixels\":12845056}" \
  > "$SLOG" 2>&1 &
VLLM_PID=$!

# Probe REAL inference (not just /v1/models) — this also warms the engine so the
# parser client's hardcoded 5s connection-test ping won't time out on cold start.
READY=0
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 -X POST \
    "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"infinity-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}")
  [ "$code" = "200" ] && { READY=1; echo "[task] server warm after ~${i}0s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "[task] SERVER DIED"; tail -60 "$SLOG"; exit 1; }
  sleep 10
done
[ $READY -eq 1 ] || { echo "[task] server not warm in time"; tail -80 "$SLOG"; kill $VLLM_PID; exit 1; }

echo "[task] parsing ${#FILES[@]} PDFs -> $OUTDIR"
parser "${FILES[@]}" \
  -o "$OUTDIR" \
  --task doc2json --output-format json \
  --backend vllm-server --api-url "http://127.0.0.1:${PORT}/v1" --api-key EMPTY \
  --model-name infinity-pro --batch-size 16 --verbose
RC=$?
echo "[task] parser exit=$RC at $(date); stopping server"
kill $VLLM_PID 2>/dev/null; sleep 3
exit $RC
