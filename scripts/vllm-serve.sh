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
# All H100 clusters request 4 GPUs single-node (tamia: whole-node
# --gpus=h100:4; rorqual/fir/nibi: --gpus=h100:4; trillium-gpu:
# --gpus-per-node=h100:1); TP auto-detects (=4).
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

MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
PORT="${PORT:-8000}"
# Long-context overrides, e.g.: cluv submit nibi scripts/vllm-serve.sh -- \
#   --max-model-len 131072 --max-num-seqs 32
# (env still works too: MAX_MODEL_LEN=... MAX_NUM_SEQS=... TP_SIZE=... PORT=...).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2;;
        --tp|--tensor-parallel-size) TP_SIZE="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --help) echo "usage: vllm-serve.sh [--model M] [--max-model-len N] [--max-num-seqs N] [--tp N] [--port P]"; exit 0;;
        *) MODEL="$1"; shift;;  # bare first arg stays the model for back-compat
    esac
done
# Tensor-parallel size: default to allocated GPU count (1 on fir/nibi/rorqual/trillium-gpu, 4 on tamia).
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    N_GPUS=1
fi
TP_SIZE="${TP_SIZE:-$N_GPUS}"
# Default serving shape: 256k native context (262144) with low concurrency (32).
# KV memory scales as max-model-len x max-num-seqs: 256k x 512 would OOM even on
# 4xH100, 256k x 32 fits (~same KV as the old 32k x 512). Qwen3.8 is hybrid
# attention (48 linear/DeltaNet + 16 attention layers): vLLM needs one Mamba
# cache block per decode sequence, hence the low --max-num-seqs.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

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

# Readiness gate: arm the idle killswitch ONLY after the endpoint answers.
# A fixed grace period murders slow loads: 256k startup (weights + KV profiling
# + torch.compile) sits at 0% GPU util for 20-40+ min, and tamia 466697 was
# killed at 25m41s (900s grace + 600s idle) mid-load. Poll localhost instead.
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
_t0=$(date +%s)
while ! curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"data"'; do
    if ! kill -0 $VLLM 2>/dev/null; then
        echo "vllm process exited during load (see traceback above)"
        break
    fi
    if (( $(date +%s) - _t0 > READY_TIMEOUT )); then
        echo "WARNING: endpoint not ready after ${READY_TIMEOUT}s, arming idle watchdog anyway"
        break
    fi
    sleep 30
done

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
