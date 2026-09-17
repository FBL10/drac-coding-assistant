# DRAC coding assistant — self-hosted LLM via vLLM

Serve an LLM with [vLLM](https://github.com/vllm-project/vllm) on DRAC H100
clusters and point [OpenCode](https://opencode.ai) at it. Cluster orchestration
via [cluv](https://github.com/mila-iqia/cluv).

## Clusters

Currently tested with rorqual, tamia and nibi. fir is too slow to schedule and trillium vllm install did not work for now. Other clusters don't have H100 (killarney?) so ignored.

Slurm accounts live in `pyproject.toml` (`[tool.cluv.clusters.*.sbatch_args]`) —
replace with your own allocation
(`ssh <cluster> 'sacctmgr show assoc user=$USER'`).

## Prerequisites

- Alliance/DRAC account with MFA, `ssh` access to the clusters above
- SSH mux config so cluv + tunnel share one authenticated connection
  (DRAC MFA blocks fresh logins): see `docs/ssh-config.snippet`
- Local: [`uv`](https://docs.astral.sh/uv/), `cluv login <clusters>`, `opencode` binary

## Setup (once per cluster)

Installs the vLLM venv (`$SCRATCH/vllm-env`, pinned vLLM) and downloads model
weights to `$SCRATCH/.cache/huggingface` on the login node :

```bash
MODEL=Qwen/Qwen3.8-27B-FP8  # example
uv run --with cluster-uv scripts/setup.py --clusters rorqual,tamia,nibi --model $MODEL
```

## Usage

```bash
# submit, wait for endpoint, tunnel, patch opencode config, launch opencode:
# minimal (vLLM defaults for everything except the model, see config options):
uv run --with cluster-uv scripts/serve-and-code.py --model $MODEL --launch
```

`scripts/vllm-serve.sh` accepts the same serving flags
(`--model/--max-model-len/--max-num-seqs/--tp/--port/--kv-cache-dtype/--tool-parser/--reasoning-parser`).

## Notes / troubleshooting

 - Idle killswitch: the job exits 30 min after last GPU use (only real
  inference resets it — `/v1/models` polls don't). Connect promptly.
