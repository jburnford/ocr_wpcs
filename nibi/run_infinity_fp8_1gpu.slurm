#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --job-name=infinity-fp8
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/project/def-jic823/infinity/logs/fp8-%j.out

# Infinity-Parser2-Pro (35B) on a SINGLE H100 via on-the-fly FP8 quantization.
# BF16 weights (~70GB) don't leave room for KV cache on one 80GB card; FP8
# (~35GB) does. Uses the serve+client pattern (parser's vllm-server backend),
# because the parser CLI has no --quantization flag — we control quant on the
# vLLM server. Mirrors the proven chandra2/run_chandra.sh recipe.
#   Usage: sbatch run_infinity_fp8_1gpu.slurm [INPUT_DIR] [OUTPUT_DIR]
set -uo pipefail
BASE=/project/def-jic823/infinity
INPUT="${1:-$BASE/input}"
OUTDIR="${2:-$BASE/output/fp8-$SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$BASE/logs"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.12 opencv/4.13.0 cuda/12.9
source "$BASE/venv/bin/activate"

export HF_HOME=/project/def-jic823/models/huggingface_cache
export HF_HUB_CACHE=$HF_HOME          # model lives at $HF_HOME/models--... (no hub/ subdir)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/jic823/.cache
export VLLM_CACHE_ROOT=/scratch/jic823/.cache/vllm
export TRITON_CACHE_DIR=/scratch/jic823/.cache/triton
export FLASHINFER_WORKSPACE_BASE=/scratch/jic823/.cache/flashinfer
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi -L

PORT=$(python -c "import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()")
SLOG="$BASE/logs/fp8-server-${SLURM_JOB_ID}.log"
echo "[job] starting FP8 vLLM server on port $PORT -> $SLOG"

vllm serve infly/Infinity-Parser2-Pro \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name infinity-pro \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  `# 65536 = room for the client'\''s hardcoded 32768-token output cap + image/text input` \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mm-processor-kwargs "{\"min_pixels\":3136,\"max_pixels\":12845056}" \
  > "$SLOG" 2>&1 &
VLLM_PID=$!

echo "[job] warming up server (real inference probe; avoids client 5s ping timeout)..."
READY=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 -X POST \
    "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"infinity-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}")
  if [ "$code" = "200" ]; then READY=1; echo "[job] server warm after ~${i}0s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[job] SERVER DIED during startup:"; tail -60 "$SLOG"; exit 1; fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "[job] server not ready in time"; tail -80 "$SLOG"; kill $VLLM_PID; exit 1; fi

echo "[job] running parser (vllm-server backend) input=$INPUT out=$OUTDIR"
parser "$INPUT" \
  -o "$OUTDIR" \
  --task doc2json \
  --output-format json \
  --backend vllm-server \
  --api-url "http://127.0.0.1:${PORT}/v1" \
  --api-key EMPTY \
  --model-name infinity-pro \
  --batch-size 16 \
  --verbose
RC=$?
echo "[job] parser exit=$RC at $(date); stopping server"
kill $VLLM_PID 2>/dev/null; sleep 3
ls "$OUTDIR" | head
exit $RC
