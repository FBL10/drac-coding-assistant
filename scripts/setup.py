#!/usr/bin/env python3
"""One-time per-cluster setup: vLLM venv + model weights in $SCRATCH cache.

Idempotent — skips steps that are already done. Runs on login nodes via cluv.

Usage:
    uv run --with cluster-uv scripts/setup.py --clusters rorqual,tamia,nibi --model Qwen/Qwen3.8-27B-FP8
    uv run --with cluster-uv scripts/setup.py --clusters fir --model Qwen/Qwen3.8-27B-FP8 --check-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys

try:
    from cluv.remote import Remote
except ModuleNotFoundError:
    print("Re-run with:\n  uv run --with cluster-uv scripts/setup.py ...", file=sys.stderr)
    raise SystemExit(2)

VLLM_VERSION = "0.29.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clusters", default="rorqual,tamia,nibi",
                   help="comma-separated cluv cluster names")
    p.add_argument("--model", required=True, help="HF model id to cache")
    p.add_argument("--revision", default=None,
                   help="optional pinned snapshot revision (omit = latest)")
    p.add_argument("--vllm-version", default=VLLM_VERSION)
    p.add_argument("--venv", default=None, help="default: $SCRATCH/vllm-env (-313 on trillium-gpu)")
    p.add_argument("--python", default=None, help="e.g. 3.13 (default: cluster default, 3.13 on trillium-gpu)")
    p.add_argument("--check-only", action="store_true", help="report status, change nothing")
    return p.parse_args()


SETUP_SCRIPT = r"""
set -u
MODEL="$1"; REV="$2"; VLLM_VER="$3"; VENV_IN="${4:-}"; PY_IN="${5:-}"; CHECK_ONLY="${6:-0}"
if [[ "$VENV_IN" == "-" ]]; then VENV_IN=""; fi
if [[ "$PY_IN" == "-" ]]; then PY_IN=""; fi
if [[ "$CLUSTER" == trillium-gpu* ]]; then
    VENV="${VENV_IN:-$SCRATCH/vllm-env-313}"; PY="${PY_IN:-3.13}"
    module load StdEnv/2023 gcc/12.3 2>/dev/null || true
else
    VENV="${VENV_IN:-$SCRATCH/vllm-env}"; PY="${PY_IN:-}"
fi
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
CACHED="$HF_HOME/hub/models--${MODEL//\//--}"
echo "--- $CLUSTER: venv=$VENV vllm=$VLLM_VER model=$MODEL ---"
ver_dir() { ls -d "$VENV/lib/python"*/site-packages/vllm-"$VLLM_VER".dist-info 2>/dev/null | head -n 1; }
if [[ "$CHECK_ONLY" == 1 ]]; then
    if [[ -n "$(ver_dir)" ]]; then echo "vllm: $VLLM_VER OK ($(ver_dir))"; else echo "vllm: MISSING (want $VLLM_VER)"; fi
    if [[ -d "$CACHED" ]]; then echo "weights: $CACHED"; else echo "weights: MISSING"; fi
    exit 0
fi
if [[ -n "$(ver_dir)" ]]; then
    echo "vllm $VLLM_VER already installed, skipping"
else
    if [[ -n "$PY" ]]; then UVPY="--python $PY"; else UVPY=""; fi
    # shellcheck disable=SC2086
    uv venv "$VENV" $UVPY && VIRTUAL_ENV="$VENV" uv pip install "vllm==$VLLM_VER"
fi
if [[ -d "$CACHED" ]]; then
    echo "weights already cached, skipping download"
else
    VIRTUAL_ENV="$VENV" uv pip install -q huggingface_hub
    if [[ -n "$REV" && "$REV" != "-" ]]; then REV_FLAG="--revision $REV"; else REV_FLAG=""; fi
    # shellcheck disable=SC2086
    "$VENV/bin/huggingface-cli" download "$MODEL" $REV_FLAG
fi
echo "vllm: $(ver_dir || echo "missing (install may have failed)")"
"""


async def setup_one(cluster: str, args: argparse.Namespace) -> bool:
    print(f"===== {cluster} =====")
    try:
        remote = await Remote.connect(hostname=cluster)
    except Exception as e:
        print(f"{cluster}: connect failed: {e}", file=sys.stderr)
        return False
    cmd = (f"CLUSTER={cluster} bash -s -- {args.model} {args.revision or '-'} "
           f"{args.vllm_version} {args.venv or '-'} {args.python or '-'} "
           f"{1 if args.check_only else 0}")
    try:
        res = await remote.run(cmd, input=SETUP_SCRIPT, display=False, hide=True)
        print(res.stdout)
        if res.returncode != 0:
            print(f"{cluster}: exit {res.returncode}\n{res.stderr}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"{cluster}: setup failed: {e}", file=sys.stderr)
        return False


async def main() -> None:
    args = parse_args()
    clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]
    ok = True
    for c in clusters:
        ok = await setup_one(c, args) and ok
    if not ok:
        raise SystemExit(1)
    print(f"done. Next: uv run --with cluster-uv scripts/serve-and-code.py --model {args.model} --launch")


if __name__ == "__main__":
    asyncio.run(main())
