# DRAC coding assistant — self-hosted LLM via vLLM (Option 2)

Serve `Qwen/Qwen3.8-27B-FP8` with vLLM on Alliance/DRAC H100 clusters and point a
local coding harness (OpenCode) at it. Cluster orchestration via
[cluv](https://github.com/mila-iqia/cluv) (`cluster-uv` on PyPI).

Model + runtime (identical on all clusters, verified 2026-09-15):

- Model: `Qwen/Qwen3.8-27B-FP8` (~29 GB), snapshot `017b9c7a…`,
  in `$SCRATCH/.cache/huggingface` (`HF_HOME`), usable offline (`HF_HUB_OFFLINE=1`).
- Runtime: `uv venv $SCRATCH/vllm-env && uv pip install vllm` → vLLM **0.29.0**.
  (Trillium-gpu only: venv is `$SCRATCH/vllm-env-313`, built with
  `uv venv --python 3.13` — see retry notes.)
- Serve flags that matter (see `scripts/vllm-serve.sh`):
  `--tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 512
  --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`, plus `module load cuda/12.6` (Trillium:
  `module load StdEnv/2023 gcc/12.3 cuda/12.6`), `VLLM_USE_DEEP_GEMM=0`,
  and all caches on `$SCRATCH` (`HF_HOME`, `TRITON_CACHE_DIR`,
  `VLLM_CACHE_ROOT/CONFIG_ROOT`, `FLASHINFER_WORKSPACE_BASE`).

## Cluster status

| Cluster | GPUs/node | Account | Status |
|---|---|---|---|
| fir | 4× H100-80GB | `def-azouaq` | ⏳ config OK, never got a GPU (0 idle). Just retry. |
| nibi | 8× H100-80GB | `def-azouaq` | ✅ serving verified (endpoint + inference) |
| rorqual | 4× H100-80GB | `def-azouaq` | ✅ serving verified (endpoint + inference) |
| trillium-gpu | 4× H100-80GB | `def-azouaq` | ❌ blocked, see retry notes |
| tamia | 4× H100/node, whole-node only | `aip-azouaq` | ✅ serving verified (endpoint live) |

Slurm quirks (already in `pyproject.toml` + `scripts/vllm-serve.sh`):

- Trillium-gpu: `--gpus-per-node` (not `--gpus`), **no `--mem`** flag,
  needs `StdEnv/2023 gcc/12.3` before cuda, `$HOME` read-only on GPU nodes.
- Tamia: whole-node allocation → `--gpus-per-node=h100:4 --cpus-per-task=48 --mem=0`
  (model still served with `--tensor-parallel-size 1`).

## Retrying fir (expected to just work)

```bash
cluv enable fir
cluv submit fir   # or: sbatch with --gpus=h100:1 --cpus-per-task=12 --mem=64G --account=def-azouaq
# watch: squeue -u $USER ; tail -f $SCRATCH/vllm-test*.log
# verify: curl http://<node>:8000/v1/models
```

## Retrying trillium-gpu (needs investigation)

Working so far: weights cached, `vllm-env-313` (py3.13, vLLM 0.29.0) installed,
job submission syntax fixed, all five cache env vars set, nvcc resolves.
Symptom: the `vllm serve` process stays alive at ~0.5% CPU / 0% RAM / 0 MiB GPU
for 15+ min with zero log output, stuck in page-cache wait (`folio_wait_bit_common`,
then `read`) on venv files under
`vllm-env-313/lib/python3.13/site-packages/transformers/models/`
(first `.../models` dir scan, then `univnet/configuration_univnet.py`).
The same files read from the **login node in ~1 ms**. Seen on trig0044, trig0011,
trig0030, trig0042. Force-reinstalling `transformers` (new inodes) did not help.

Next steps:

1. `cluv enable trillium-gpu`
2. Resubmit the last script, maybe with a longer limit:
   `sbatch $SCRATCH/vllm-test11-trig.sbatch` (test11 file is the py3.13 one).
3. If it still wedges, escalate to SciNet with: job IDs 921031/921073/921099/921115/921370,
   `wchan=folio_wait_bit_common` + `rpc_wait_bit_killable`, 0% MEM, login-vs-compute
   read discrepancy, other users' heavy I/O on the same nodes.
4. Workaround candidates if support is slow: `TRANSFORMERS_OFFLINE=1` won't help
   (already offline); try `uv pip install --no-cache` full venv rebuild, or
   `HF_XET`? No — most promising: `strace -f -p <pid>` (if available) to name the
   exact hanging file, or `py-spy dump --pid` to see the Python stack.

## Harness (local OpenCode → cluster)

```bash
# on your laptop, after a serve job starts on <node>:
ssh -L 8000:<node>:8000 <cluster>   # node from: squeue -j <jobid> -o '%N'
```

`~/.config/opencode/opencode.jsonc` provider:

```json
{
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (DRAC)",
      "options": { "baseURL": "http://localhost:8000/v1" }
    }
  }
}
```

Smoke test used throughout (from a login node, no tunnel needed):

```bash
NODE=$(squeue -u $USER -o "%.12N %.16j" | awk "/vllm-test/{print \$1}")
curl -s http://$NODE:8000/v1/models | head -c 300
curl -s http://$NODE:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Reply with exactly: hello world"}],"max_tokens":30,"temperature":0}'
```
