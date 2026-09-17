# DRAC coding assistant — self-hosted LLM via vLLM

Serve an LLM with [vLLM](https://github.com/vllm-project/vllm) on DRAC H100
clusters and point [OpenCode](https://opencode.ai) at it. Cluster orchestration
via [cluv](https://github.com/mila-iqia/cluv) (`cluster-uv` on PyPI).

No model is hardcoded: you pass `--model` everywhere, and serving flags
(`--max-model-len`, `--max-num-seqs`, `--kv-cache-dtype`, `--tool-parser`,
`--reasoning-parser`) are forwarded to vLLM only when set. The examples below
use `Qwen/Qwen3.8-27B-FP8` at 256k context × 32 seqs on 4×H100 single-node.

## Cluster status

| Cluster | GPUs/node | Status |
|---|---|---|
| rorqual | 4× H100-80GB | ✅ serving verified |
| tamia | 4× H100/node, whole-node only | ✅ serving verified |
| nibi | 8× H100-80GB | ✅ serving verified |
| fir | 4× H100-80GB | best-effort, needs GPU availability |
| trillium-gpu | 4× H100-80GB | known-blocked (compute-node filesystem stalls, see `docs/HANDOFF-2026-09.md`) |

Slurm accounts live in `pyproject.toml` (`[tool.cluv.clusters.*.sbatch_args]`) —
replace with your own allocation
(`ssh <cluster> 'sacctmgr show assoc user=$USER'`). See `docs/HANDOFF-2026-09.md`
for session history.

## Prerequisites

- Alliance/DRAC account with MFA, `ssh` access to the clusters above
- SSH mux config so cluv + tunnel share one authenticated connection
  (DRAC MFA blocks fresh logins): see `docs/ssh-config.snippet`
- Local: [`uv`](https://docs.astral.sh/uv/), `cluv login <clusters>`, `opencode` binary

## Setup (once per cluster)

Installs the vLLM venv (`$SCRATCH/vllm-env`, pinned vLLM) and downloads model
weights to `$SCRATCH/.cache/huggingface` on the login node (idempotent):

```bash
uv run --with cluster-uv scripts/setup.py --clusters rorqual,tamia,nibi --model Qwen/Qwen3.8-27B-FP8
uv run --with cluster-uv scripts/setup.py --clusters fir --model Qwen/Qwen3.8-27B-FP8 --check-only
```

## Usage

```bash
MODEL=Qwen/Qwen3.8-27B-FP8

# submit, wait for endpoint, tunnel, patch opencode config, launch opencode:
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL \
  --max-model-len 262144 --max-num-seqs 32 --kv-cache-dtype fp8 \
  --tool-parser qwen3_coder --reasoning-parser qwen3 --launch

# minimal (vLLM defaults for everything except the model):
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL --launch

# target one cluster / reuse the newest ready job:
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL --cluster nibi --launch
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL --no-submit --launch

# extra sbatch overrides (repeatable):
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL --launch --sbatch-arg gpus=h100:8
```

`scripts/vllm-serve.sh` accepts the same serving flags
(`--model/--max-model-len/--max-num-seqs/--tp/--port/--kv-cache-dtype/--tool-parser/--reasoning-parser`).
The opencode context limit is autodetected from `/v1/models`.

## Notes / troubleshooting

- 256k loads take 20–45+ min at 0% GPU; the script waits for `/v1/models`.
- Idle killswitch: the job exits 10 min after last GPU use (only real
  inference resets it — `/v1/models` polls don't). Connect promptly.
- `ssh tunnel exited early (rc=255)` + MFA in `~/.config/opencode/tunnel.log`:
  handled automatically via mux-shared fallback (`ssh -O forward`).
- Stale `:8000`: `lsof -ti tcp:8000 | xargs kill -9`, or
  `ssh -O cancel -L 8000:<node>:8000 <cluster>`.
- `job X no longer in squeue`: it finished/expired —
  `ssh <cluster> 'sacct -j X --format=JobID,State,ExitCode -P'`, then resubmit.
- Smoke test (login node, no tunnel):
  `curl http://<node>:8000/v1/models` (node from `squeue -j <jobid> -o '%N'`).
