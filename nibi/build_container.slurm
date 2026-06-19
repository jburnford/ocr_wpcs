#!/bin/bash
#SBATCH --job-name=infinity_build
#SBATCH --output=build_infinity_%j.out
#SBATCH --error=build_infinity_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --account=def-jic823

# Build the Infinity-Parser2 Apptainer container (CPU-only job, no GPU).
# Mirrors the working mineru build. The vLLM-base def assembles fast (no
# flash-attn compile); 1.5h is generous. If you switch to the nvidia/cuda
# from-source fallback def, raise --time to 3:00:00 and --mem to 64G.

set -euo pipefail
module load apptainer/1.3.5

BUILD_DIR="${HOME}/projects/def-jic823/infinity"
cd "${BUILD_DIR}"

echo "Building infinity_parser2.sif on $(hostname) at $(date)"
echo "Apptainer: $(apptainer --version)"

apptainer build --force infinity_parser2.sif infinity_parser2.def

echo "Built:"; ls -lh "${BUILD_DIR}/infinity_parser2.sif"

echo "Smoke test:"
apptainer exec infinity_parser2.sif python3 -c \
    "import torch, vllm, infinity_parser2; print('torch', torch.__version__, '| vllm', vllm.__version__)"
apptainer exec infinity_parser2.sif parser --help >/dev/null && echo "parser CLI OK"
echo "Done at $(date)"
