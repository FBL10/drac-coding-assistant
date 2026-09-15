#!/bin/bash
#SBATCH --job-name=vllm-qwen3.8-27b-fp8
#SBATCH --time=3:00:00
#SBATCH --dependency=singleton
# NOTE: GPU/CPU/mem/account are NOT set here on purpose -- they differ per cluster
# and come from pyproject.toml [tool.cluv.clusters.*.sbatch_args] on `cluv submit`.
# (Trillium-gpu rejects --mem and needs --gpus-per-node; Tamia needs whole-node
# --gpus-per-node=h100:4.) For direct sbatch, pass them on the CLI, e.g.:
#   sbatch --gpus=h100:1 --cpus-per-task=12 --mem=64G --account=def-azouaq scripts/vllm-serve.sh
# NOTE: cluv sbatch_args in pyproject.toml override the above on submit.
# Tamia requests --gpus=h100:4 --cpus-per-task=48 --mem=0 (whole node).
# Rorqual/Fir/Nibi/Trillium-gpu request --gpus=h100:1.
#
# Usage:
#   cluv submit fir                      # uses this script via job_script_path
#   cluv submit first                    # race all H100 clusters, keep first to start
#   cluv submit tamia scripts/vllm-serve.sh -- --model Qwen/Qwen3.8-27B-FP8
#
# Pre-download weights on login node (required where compute has no internet:
# rorqual, trillium-gpu, tamia; optional on fir/nibi which have internet):
#   huggingface-cli download Qwen/Qwen3.8-27B-FP8
#   # or: python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.8-27B-FP8')"
#
# Connect locally via SSH tunnel, then point opencode at http://localhost:8000/v1:
#   ssh -L 8000:<compute-node>:8000 <cluster>   # get node from: squeue -j $JOBID -o '%N'

set -euo pipefail

# Unbuffered logs so `tail -f` on the Slurm output shows progress live.
export PYTHONUNBUFFERED=1

# CUDA toolkit (nvcc) is required at runtime: flashinfer JIT-compiles kernels
# on the compute node. The pip torch bundles CUDA libs but NOT nvcc.
# Trillium needs a toolchain first (StdEnv/2023 + gcc); other clusters take
# bare cuda/12.6. No-op on machines without environment modules (e.g. laptop).
if command -v module &>/dev/null; then
    module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null \
    || module load cuda/12.6 2>/dev/null \
    || module load cuda 2>/dev/null || true
fi

MODEL="${1:-${MODEL:-Qwen/Qwen3.8-27B-FP8}}"
PORT="${PORT:-8000}"
# Tensor-parallel size: default to allocated GPU count (1 on fir/nibi/rorqual/trillium-gpu, 4 on tamia).
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    N_GPUS=1
fi
TP_SIZE="${TP_SIZE:-$N_GPUS}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Qwen3.8 is hybrid attention (48 linear/Mamba layers): vLLM needs one Mamba
# cache block per decode sequence. Small --max-model-len values yield few blocks
# (e.g. 821 at 8k), so cap --max-num-seqs (default 1024) to fit. 512 is safe here.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

# Offline clusters (rorqual/trillium-gpu/tamia) have no internet on compute:
# use cached weights only. Fir/nibi have internet so this is a no-op there.
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export TRANSFORMERS_OFFLINE=1
fi
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
# Keep compiler/download caches off $HOME (quota/small) and on $SCRATCH.
# Required on some clusters (e.g. trillium-gpu denies writes to ~/.triton).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
# Trillium GPU nodes mount $HOME read-only: redirect every home-based cache.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH/.cache/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$SCRATCH/.config/vllm}"
# FlashInfer derives ALL its dirs (cache + JIT workspace) from this one var:
# $BASE/.cache/flashinfer. (FLASHINFER_CACHE_DIR/WORKSPACE_DIR are IGNORED.)
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SCRATCH}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$SCRATCH/.cache/flashinfer}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$SCRATCH/.cache/flashinfer}"
# DeepGEMM FP8 JIT fails to nvcc-compile on H100 with the default cuda/12.6
# module ("NVCC compilation failed"). Fall back to CUTLASS FP8 (works, slightly
# slower). Re-enable for max throughput with: VLLM_USE_DEEP_GEMM=1 + cuda/12.9+.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

echo "=== vLLM serve ==="
echo "date: $(date -u +%FT%TZ)  host: $(hostname)  model: $MODEL  tp=$TP_SIZE  port=$PORT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi -L || true

# Tool-call flags prevent: '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser'
# qwen3_coder matches the coding-agent use case from the issue; upstream recipe uses qwen3_xml
# for this non-Coder checkpoint -- override with: TP_PARSER=qwen3_xml
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"

# vLLM binary: prefer the uv-managed venv (installed per doc setup), fall back to PATH.
# Install with: uv venv $SCRATCH/vllm-env && VIRTUAL_ENV=$SCRATCH/vllm-env uv pip install vllm
if [[ -x "${VLLM_BIN:-$SCRATCH/vllm-env/bin/vllm}" ]]; then
    VLLM_BIN="${VLLM_BIN:-$SCRATCH/vllm-env/bin/vllm}"
else
    VLLM_BIN="${VLLM_BIN:-vllm}"
fi
echo "vllm binary: $VLLM_BIN ($("$VLLM_BIN" --version 2>/dev/null || echo 'version unknown'))"

# shellcheck disable=SC2086
"$VLLM_BIN" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 &
VLLM=$!

# Grace period: let weights download/load before arming the killswitch.
sleep "${GRACE_PERIOD:-900}"

idle=0
while kill -0 $VLLM 2>/dev/null; do
    sleep 30
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -rn | head -1)
    if (( util > 0 )); then idle=0; else idle=$(( idle + 30 )); fi
    if (( idle >= ${IDLE_TIMEOUT:-600} )); then
        echo "GPU idle for ${IDLE_TIMEOUT:-600}s, stopping vllm (job $SLURM_JOB_ID on $(hostname))"
        kill $VLLM 2>/dev/null || true
        break
    fi
done
wait $VLLM
