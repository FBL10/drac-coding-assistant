# Handoff — DRAC coding assistant (vLLM on H100 clusters), 2026-09-15

Goal: serve `Qwen/Qwen3.8-27B-FP8` with vLLM on DRAC H100 clusters and point a
local coding harness (OpenCode) at it. Orchestration via `cluv` (`cluster-uv`).
Repo: `github.com/FBL10/drac-coding-assistant` (private). Local dir:
`/Users/fl/code/drac_coding_assistant`.

## Canonical stack (verified 2026-09-15)

- Model: `Qwen/Qwen3.8-27B-FP8`, snapshot `017b9c7a…`, in
  `$SCRATCH/.cache/huggingface` (`HF_HOME`), offline (`HF_HUB_OFFLINE=1`).
- Runtime: `uv venv $SCRATCH/vllm-env && uv pip install vllm` → vLLM **0.29.0**.
- Serve flags (`scripts/vllm-serve.sh`): `--tensor-parallel-size 1
  --max-model-len 32768 --max-num-seqs 512 --kv-cache-dtype fp8
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3` (+ `module load cuda/12.6`, `VLLM_USE_DEEP_GEMM=0`).
- All caches on `$SCRATCH`: `HF_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`,
  `VLLM_CONFIG_ROOT`, **`FLASHINFER_WORKSPACE_BASE`** (= `$SCRATCH`; this is the
  only FlashInfer var that matters — `FLASHINFER_CACHE_DIR`/`WORKSPACE_DIR` are
  ignored, derived as `$BASE/.cache/flashinfer`).
- Trillium-gpu only: venv is `$SCRATCH/vllm-env-313` (`uv venv --python 3.13`),
  modules `StdEnv/2023 gcc/12.3 cuda/12.6`, sbatch uses `--gpus-per-node`
  (not `--gpus`) and **no `--mem`**.

## Cluster status (end of session)

| Cluster | GPUs | Account | Status |
|---|---|---|---|
| rorqual | 4×H100/node | def-azouaq | ✅ serving verified 32k, then cleaned up (no jobs left) |
| tamia | 4×H100/node, whole-node (`--gpus-per-node=h100:4 --cpus-per-task=48 --mem=0`, serve TP=1) | aip-azouaq | ✅ serving verified earlier, cleaned up |
| nibi | 8×H100/node | def-azouaq | ✅ serving verified earlier; SSH now MFA-blocked |
| fir | 4×H100/node | def-azouaq | ⏳ config OK, never got a GPU (0 idle); SSH now MFA-blocked |
| trillium-gpu | 4×H100/node | def-azouaq | ❌ blocked (see below); SSH now MFA-blocked |

`cluv` scope at handoff: rorqual + tamia enabled; fir, trillium-gpu, nibi
disabled. Rorqual has a read-only GitHub deploy key (`rorqual-cluv`); nibi/tamia
use existing keys. Note: `logs/` is a symlink to `$SCRATCH/...` — removed from
git (`logs/` in `.gitignore`); cluv results live in
`$SCRATCH/logs/drac_coding_assistant/<cluster>_<jobid>/`.

## Job history (2026-09-15, times ~UTC)

- Manual 8k-context tests passed on rorqual/tamia/nibi (endpoint + inference).
- test12 (32k, 2h): rorqual `21140056` (script still `--job-name=vllm-test4`,
  log `$SCRATCH/vllm-test12-21140056.log`) ran 40+ min on **rg32401**,
  `Application startup complete`, `/v1/models` OK (`max_model_len: 32768`),
  chat completion returned tokens. **Killed at handoff.**
- tamia `465564` (`vllm-test12-tami`, tg10801) was RUNNING at handoff check —
  earlier `scancel -n vllm-test12` missed it (name truncation). **Killed by ID
  at handoff; verified COMPLETING, other tamia jobs untouched.**
- `cluv submit first` race: rorqual `21140906` won in seconds, nibi `22019077`
  + tamia `465570` auto-cancelled. `21140906` was manually cancelled — it showed
  8+ min with no EngineCore children, 0 MiB GPU, stuck in `connect()`; may have
  been impatience (32k loads take ~15–20 min). Lesson: check child procs + GPU
  trend before cancelling.
- Replacement `cluv submit rorqual` job `21141284` (rg31607) **FAILED** after
  25 min: `KeyboardInterrupt: terminated` in `_interrupt_init` during socket
  `create_connection` — i.e. got SIGTERM, not a model crash. Same node
  (rg31607) had the earlier suspect job. Possible node/scheduler kills on
  rg31607; the survivor ran on rg32401.
- Trillium-gpu test11 `921370` (py3.13 venv): fate unknown at handoff (cluster
  MFA-blocked). Earlier symptom across trig0044/0011/0030/0042: process alive,
  ~0.5% CPU, 0 MiB GPU, 15+ min zero log output, `wchan=folio_wait_bit_common`
  then `read` on venv files under `transformers/models/`; login-node reads of
  same files ~1 ms. Six trillium fixes already landed (Slurm syntax, toolchain
  modules, 5 cache env vars, py3.13 venv). If reproducible → SciNet ticket
  (job IDs 921031/921073/921099/921115/921370). Debug candidates:
  `strace -f -p`, `py-spy dump --pid`.
- Minor: vLLM warns `ulimit 51200` can't auto-increase (fd-limit risk);
  harmless so far.

## Cleanup done at handoff

- `scancel 21140056` (rorqual) — verified: no jobs left on rorqual.
- `scancel 465564` (tamia) — verified: only foreign jobs `465486`/`465487`
  (`cluv-tac_adr_tra`, another project, **do not touch**) still RUNNING.
- NOT verifiable (SSH MFA-blocked at handoff): trillium-gpu `921370`
  (1h limit, likely expired), any pending fir job, any nibi leftovers
  (nibi jobs were all cancelled during session, likely clean).

## Next session

1. `cluv login nibi fir trillium-gpu` (re-auth), then verify no leftover jobs:
   `squeue -u $USER` on each.
2. To serve again: `cluv submit first` (needs all target clusters enabled +
   `cluv sync`), or `cluv submit rorqual` directly. Avoid rg31607 if kills recur.
3. Tunnel from laptop: `ssh -L 8000:<node>:8000 <cluster>` (node from
   `squeue -j <jobid> -o '%N'`), provider in `~/.config/opencode/opencode.jsonc`:
   `@ai-sdk/openai-compatible`, `baseURL http://localhost:8000/v1`.
4. Smoke test (login node, no tunnel):
   `curl http://<node>:8000/v1/models` + `/v1/chat/completions` with
   `{"model":"Qwen/Qwen3.8-27B-FP8",...}` (use generous `max_tokens`; the
   reasoning parser eats tokens).
5. fir retry expected to just work when GPUs free. Trillium-gpu needs the
   investigation above.
6. Don't poll with `sleep 280` wrappers; check directly. Don't cancel RUNNING
   jobs on timing hunches.

---

## Session 2026-09-16 (times EDT = UTC-4)

Canonical one-shot command (from repo root) — submits, waits for the
endpoint, tunnels, patches opencode, launches it:

```bash
uv run --with cluster-uv scripts/serve-and-code.py --cluster first --launch
# reuse newest ready job without resubmitting:
uv run --with cluster-uv scripts/serve-and-code.py --no-submit --launch
```

Current serving shape is **256k x 32** (`--max-model-len 262144
--max-num-seqs 32`, `pyproject.toml` + `serve-and-code.py` defaults), not the
32k x 512 in "Canonical stack" above.

### Jobs

- rorqual `21201669` (submitted 15:38): allocated a compressed multi-node
  list `rg[31801-31802,31901,31903]`, ran on **rg31801** with `tp=1`
  (1 GPU, `CUDA_VISIBLE_DEVICES=0`). Engine init 250 s, `Application startup
  complete` **16:01:21**, then `GPU idle for 600s, stopping vllm` at
  **16:11:47** → `COMPLETED` (17m47s). Log:
  `~/scratch/logs/drac_coding_assistant/rorqual_21201669/slurm-21201669.out`.
  Root cause of the idle death: `get_node()` comma-split the compressed
  hostlist into garbage `rg[31801-31802`, so the watcher polled a nonexistent
  host and never sent traffic. Fixed the same day (see below).
- tamia `466840` (submitted 16:33): **RUNNING** on **tg10702**,
  `Application startup complete` **16:38:42** (only ~5 min load), two
  `GET /v1/models 200 OK` (on-node gate + login-node watcher — i.e. the
  fixed `get_node()` worked). Log:
  `~/scratch/logs/drac_coding_assistant/tamia_466840/slurm-466840.out`.
- `cluv status` at ~16:40: tamia + rorqual + nibi enabled (fir/killarney/
  vulcan/trillium/trillium-gpu disabled), **0 idle H100** on all three, no
  own jobs except `466840`.

### `scripts/serve-and-code.py` fixes (landed 2026-09-16, verified `py_compile`)

1. `get_node()`: `squeue` compressed hostlists (`rg[...]`) are expanded via
   `scontrol show hostnames <nodelist> | head -n 1`; single hosts trim domain
   suffix; `(null)`/`n/a` still retry. (The old
   `re.split(r"[, ]")[0].strip("[]")` produced unresolvable names.)
2. `start_tunnel()`: ssh now bypasses the shared ControlMaster mux
   (`-S none -o ControlMaster=no -o ControlPersist=no`) so the `-L 8000`
   forward is owned by our child process and dies with it. Previously the
   forward stuck to the persistent tamia mux master (pid 8664, since 12:19)
   and kept serving TCP resets on `:8000` after the job died → opencode
   "socket connection was closed unexpectedly" + next run couldn't bind.
   If `:8000` is already listening, the script probes `/v1/models`: healthy
   → reuse; otherwise abort with the exact fix instead of handing opencode a
   dead endpoint. Tunnel health (not just TCP accept) is also the new
   post-start check; `tunnel.log` gets a per-run header line.
3. `main()` order is now tunnel → patch config → smoke (was patch first, so
   `server_max_len` autodetect always missed and fell back to context
   `32768` instead of the real `262144`).

### Still to do once (stale state from before the fix)

- `ssh -O exit tamia` to drop the stale `:8000` forward pinned on mux 8664,
  then `uv run --with cluster-uv scripts/serve-and-code.py --no-submit
  --launch` (reuses ready `tamia 466840`, rebuilds tunnel to `tg10702`,
  re-patches `opencode.jsonc` with context `262144`). Verify with
  `curl -s http://localhost:8000/v1/models | head -c 300` before using opencode.
