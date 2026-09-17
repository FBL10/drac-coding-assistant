#!/bin/bash
# Serve an LLM with vLLM on a DRAC H100 compute node.
# GPU/CPU/mem/account come from pyproject.toml [tool.cluv.clusters.*.sbatch_args],
# so this script stays cluster-agnostic. Examples:
#   cluv submit first                    # race enabled H100 clusters
#   cluv submit rorqual
#   sbatch --gpus=h100:4 --cpus-per-task=12 --mem=64G --account=YOUR-ACCOUNT scripts/vllm-serve.sh
# Options (also settable via env MODEL/PORT/MAX_MODEL_LEN/MAX_NUM_SEQS/TP_SIZE/TOOL_PARSER):
#   scripts/vllm-serve.sh [--model M] [--max-model-len N] [--max-num-seqs N]
#                         [--tp N] [--port P] [--tool-parser P] [--reasoning-parser P]
set -euo pipefail
export PYTHONUNBUFFERED=1

if command -v module &>/dev/null; then
    module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null \
    || module load cuda/12.6 2>/dev/null \
    || module load cuda 2>/dev/null || true
fi

MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
PORT="${PORT:-8000}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2;;
        --tp|--tensor-parallel-size) TP_SIZE="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --tool-parser) TOOL_PARSER="$2"; shift 2;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2;;
        --help) echo "usage: vllm-serve.sh [--model M] [--max-model-len N] [--max-num-seqs N] [--tp N] [--port P] [--tool-parser P] [--reasoning-parser P]"; exit 0;;
        *) MODEL="$1"; shift;;
    esac
done
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    N_GPUS=1
fi
TP_SIZE="${TP_SIZE:-$N_GPUS}"
# Default serving shape for Qwen3.8-27B-FP8: 256k context x 32 seqs.
# KV memory scales as len x seqs; other models override via flags above.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export TRANSFORMERS_OFFLINE=1
fi
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH/.cache/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$SCRATCH/.config/vllm}"
# FlashInfer derives all dirs from this one var ($BASE/.cache/flashinfer).
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SCRATCH}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$SCRATCH/.cache/flashinfer}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$SCRATCH/.cache/flashinfer}"
# DeepGEMM FP8 JIT fails under cuda/12.6; use CUTLASS FP8 instead.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

echo "=== vLLM serve ==="
echo "date: $(date -u +%FT%TZ)  host: $(hostname)  model: $MODEL  tp=$TP_SIZE  port=$PORT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi -L || true

TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Prefer the setup-script venv, fall back to PATH.
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
    --reasoning-parser "$REASONING_PARSER" &
VLLM=$!

# Arm the idle killswitch only after the endpoint answers: 256k loads sit at
# 0% GPU for 20-40+ min, and a fixed grace period would murder them mid-load.
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
