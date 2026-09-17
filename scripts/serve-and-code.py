#!/usr/bin/env python3
"""Serve an LLM with vLLM on DRAC H100s, tunnel it, patch opencode, launch opencode.

Usage:
    uv run --with cluster-uv scripts/serve-and-code.py --launch
    uv run --with cluster-uv scripts/serve-and-code.py --cluster nibi --model Qwen/Qwen3.8-27B-FP8 --launch
    uv run --with cluster-uv scripts/serve-and-code.py --no-submit --launch  # reuse newest job
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from cluv.cli.submit import submit
    from cluv.remote import Remote
except ModuleNotFoundError:
    print(
        "cluv not importable in this interpreter. Re-run with:\n"
        "  uv run --with cluster-uv scripts/serve-and-code.py ...",
        file=sys.stderr,
    )
    raise SystemExit(2)

# Per-model serving defaults. --max-model-len / --max-num-seqs / --tool-parser /
# --reasoning-parser override these (and are forwarded to scripts/vllm-serve.sh).
MODELS = {
    "Qwen/Qwen3.8-27B-FP8": {
        "max_model_len": 262144,  # 256k native context; KV ~ len x seqs
        "max_num_seqs": 32,  # low: hybrid attention needs a Mamba block per seq
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "output_tokens": 8192,
    },
}
DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
PORT = 8000
ENDPOINT_TIMEOUT_S = 90 * 60  # RUNNING != ready: 256k loads take 20-45+ min
POLL_S = 15


def profile(model: str) -> dict:
    return MODELS.get(model, {"max_model_len": 32768, "max_num_seqs": 32,
                              "tool_parser": "qwen3_coder", "reasoning_parser": "qwen3",
                              "output_tokens": 8192})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", default="first",
                   help="'first' races all enabled H100 clusters, or name one")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--tool-parser", default=None)
    p.add_argument("--reasoning-parser", default=None)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-submit", action="store_true",
                   help="reuse newest cached job instead of submitting")
    p.add_argument("--job-id", type=int, default=None,
                   help="with --no-submit, reuse this job (needs --cluster)")
    p.add_argument("--no-tunnel", action="store_true", help="don't (re)start ssh -L tunnel")
    p.add_argument("--no-wait", action="store_true", help="don't wait for /v1/models")
    p.add_argument("--launch", action="store_true",
                   help="exec opencode after setup (default: print next steps)")
    p.add_argument("--endpoint-timeout", type=int, default=ENDPOINT_TIMEOUT_S)
    p.add_argument("--sbatch-arg", action="append", default=[],
                   help="extra sbatch override, e.g. --sbatch-arg gpus=h100:8 (repeatable)")
    p.add_argument("--autocommit", action="store_true",
                   help="git commit tracked changes before submit")
    return p.parse_args()


def resolve(model: str, args: argparse.Namespace) -> dict:
    prof = profile(model)
    return {
        "max_model_len": args.max_model_len or prof["max_model_len"],
        "max_num_seqs": args.max_num_seqs or prof["max_num_seqs"],
        "tool_parser": args.tool_parser or prof["tool_parser"],
        "reasoning_parser": args.reasoning_parser or prof["reasoning_parser"],
        "output_tokens": prof["output_tokens"],
    }


async def submit_first(cluster: str, sbatch_args: list[str],
                       program_args: list[str], autocommit: bool = False) -> tuple[str, int]:
    """Submit and block until cluv picks a winner. Returns (cluster, job_id)."""
    job = await submit(cluster, None, sbatch_args, program_args, autocommit=autocommit)
    if job is None:
        raise SystemExit("cluv submit returned None (all sbatch failed or interrupted).")
    print(f"winner: job {job.job_id} on {job.cluster}")
    return job.cluster, job.job_id


def newest_cached_job() -> tuple[str, int] | None:
    """Most recently submitted cluv job (for --no-submit)."""
    try:
        from cluv.cache import load_jobs
    except ImportError:
        return None
    jobs = load_jobs()
    if not jobs:
        return None
    j = max(jobs, key=lambda x: x.submitted_at)
    return j.cluster, j.job_id


async def get_node(cluster: str, job_id: int, timeout: int = 1800) -> str:
    """Poll squeue until the job has a compute node. Returns bare node name."""
    remote = await Remote.connect(hostname=cluster)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = await remote.get_output(f"squeue -j {job_id} -h -o '%N'")
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"job {job_id} on {cluster} is no longer in squeue "
                f"({(e.stderr or '').strip() or 'Invalid job id'}). It likely finished/expired — "
                f"check `ssh {cluster} 'sacct -j {job_id} --format=JobID,State,ExitCode -P'`. "
                f"Resubmit (drop --no-submit) or pick another --job-id.") from e
        raw = out.strip().split()[0] if out.strip() else ""
        if not raw or raw in ("(null)", "n/a", "None"):
            await asyncio.sleep(POLL_S)
            continue
        if "[" in raw:
            # Compressed hostlist (e.g. rg[31801-31802,31901]) — expand to first host.
            expanded = await remote.get_output(f"scontrol show hostnames {raw} | head -n 1")
            node = expanded.strip().split()[0] if expanded.strip() else ""
            if not node or node in ("(null)", "n/a", "None"):
                await asyncio.sleep(POLL_S)
                continue
        else:
            node = raw.split(".")[0]  # trim domain suffix
        print(f"node: {node} ({cluster})")
        return node
    raise TimeoutError(f"job {job_id} on {cluster} got no node within {timeout}s")


async def wait_for_endpoint(cluster: str, node: str, port: int, timeout: int) -> bool:
    """Poll /v1/models from the login node until vLLM answers."""
    remote = await Remote.connect(hostname=cluster)
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = await remote.get_output(f"curl -s -m 10 http://{node}:{port}/v1/models || true")
        if "max_model_len" in out or '"data"' in out:
            print(f"endpoint ready: http://{node}:{port}/v1/models")
            print(out[:300])
            return True
        print(f"  ... endpoint not up yet ({time.strftime('%H:%M:%S')}), waiting {POLL_S}s")
        await asyncio.sleep(POLL_S)
    print(f"WARNING: endpoint not ready after {timeout}s", file=sys.stderr)
    return False


def strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments outside strings (minimal JSONC tolerance)."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1] if i + 1 < n else "")
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        else:
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                while i < n and text[i] != "\n":
                    i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
            else:
                out.append(c)
                i += 1
    return "".join(out)


def http_get(url: str) -> str:
    try:
        return subprocess.run(["curl", "-s", "-m", "10", url],
                              capture_output=True, text=True, timeout=15).stdout or ""
    except Exception:
        return ""


def endpoint_healthy(body: str) -> bool:
    return "max_model_len" in body or '"data"' in body


def server_max_len(port: int) -> int | None:
    m = re.search(r'"max_model_len"\s*:\s*(\d+)', http_get(f"http://localhost:{port}/v1/models"))
    return int(m.group(1)) if m else None


def patch_opencode_config(port: int, model: str, context: int | None = None,
                          output_tokens: int = 8192) -> Path:
    """Ensure ~/.config/opencode/opencode.jsonc has the vllm provider."""
    cfg = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if cfg.exists():
        raw = cfg.read_text()
        shutil.copy(cfg, cfg.with_suffix(".jsonc.bak"))
        try:
            data = json.loads(strip_jsonc_comments(raw))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{cfg} unparsable even as JSONC ({e}); backup at .bak, fix manually.")
    else:
        data = {}
    if not isinstance(data, dict):
        raise SystemExit(f"{cfg} root is not an object; backup made, fix manually.")
    if context is None:
        context = server_max_len(port) or 32768
    data.setdefault("provider", {})["vllm"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "vLLM (DRAC)",
        "options": {"baseURL": f"http://localhost:{port}/v1"},
        "models": {model: {"name": model, "limit": {"context": context, "output": output_tokens}}},
    }
    cfg.write_text(json.dumps(data, indent=2) + "\n")
    print(f"patched {cfg} (backup .bak): provider.vllm -> http://localhost:{port}/v1")
    return cfg


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _mux_forward(cluster: str, node: str, port: int) -> bool:
    """Add -L forward to the existing ControlMaster mux (no re-auth)."""
    fwd = f"{port}:{node}:{port}"
    r = subprocess.run(["ssh", "-O", "forward", "-L", fwd, cluster],
                       capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        print(f"mux forward failed: {r.stderr.strip() or r.stdout.strip()}", file=sys.stderr)
        return False
    atexit.register(lambda: subprocess.run(
        ["ssh", "-O", "cancel", "-L", fwd, cluster], capture_output=True, timeout=15))
    for _ in range(20):
        time.sleep(1)
        if endpoint_healthy(http_get(f"http://localhost:{port}/v1/models")):
            return True
    print("WARNING: mux forward added but /v1/models not serving.", file=sys.stderr)
    return False


def start_tunnel(cluster: str, node: str, port: int) -> subprocess.Popen | None:
    """Start `ssh -N -L port:node:port cluster`; falls back to mux on MFA failure.

    The owned connection (`-S none`) dies with us, avoiding stale forwards stuck
    on the persistent mux. But DRAC enforces MFA on fresh logins, so when the
    owned connection fails with Permission denied we reuse the existing
    ControlMaster mux (`ssh -O forward`), which needs no re-auth.
    """
    if port_open(port):
        if endpoint_healthy(http_get(f"http://localhost:{port}/v1/models")):
            print(f"localhost:{port} already serves /v1/models — reusing existing tunnel.")
            return None
        raise SystemExit(
            f"localhost:{port} is already listening but does NOT serve /v1/models "
            f"(stale forward?). Free it and retry:\n"
            f"  lsof -ti tcp:{port} | xargs kill -9   # or: ssh -O exit {cluster}\n"
            f"Then re-run with --no-submit to reuse the ready job."
        )
    cmd = ["ssh", "-N", "-S", "none", "-o", "ControlMaster=no", "-o", "ControlPersist=no",
           "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
           "-o", "ServerAliveCountMax=3", "-L", f"{port}:{node}:{port}", cluster]
    print("+", " ".join(cmd))
    log = Path.home() / ".config" / "opencode" / "tunnel.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logfh = open(log, "a")
    logfh.write(f"\n--- {time.strftime('%F %T')} cluster={cluster} node={node} port={port} ---\n")
    logfh.flush()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logfh,
                            stderr=subprocess.STDOUT, start_new_session=True)
    atexit.register(lambda: (proc.poll() is None and (proc.terminate(), print("tunnel closed"))))
    for _ in range(20):
        time.sleep(1)
        if endpoint_healthy(http_get(f"http://localhost:{port}/v1/models")):
            print(f"tunnel up: http://localhost:{port}/v1/models")
            return proc
        if proc.poll() is not None:
            break
    else:
        print("WARNING: tunnel not serving /v1/models yet; check `ssh -L` manually.", file=sys.stderr)
        return proc
    logfh.flush()
    tail = ""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 4096))
            tail = f.read().decode(errors="replace")
    except OSError:
        pass
    if re.search(r"Permission denied|Multifactor|keyboard-interactive|multifacteur", tail, re.I):
        print(f"owned tunnel failed (rc={proc.returncode}, likely MFA) — "
              f"falling back to `ssh -O forward -L {port}:{node}:{port} {cluster}` ...")
        if _mux_forward(cluster, node, port):
            print(f"tunnel up (mux-shared): http://localhost:{port}/v1/models")
            print(f"cleanup when done: ssh -O cancel -L {port}:{node}:{port} {cluster}")
            return None
        raise SystemExit(
            f"both owned and mux-shared tunnels failed. Check `ssh -O check {cluster}` "
            f"(re-auth with `cluv login` if the master is dead), then re-run with --no-submit.")
    raise SystemExit(f"ssh tunnel exited early (rc={proc.returncode}); node/cluster wrong? See {log}")


async def main() -> None:
    args = parse_args()
    port, model = args.port, args.model
    cfg = resolve(model, args)

    if args.no_submit:
        cached = newest_cached_job()
        if args.job_id:
            cluster = args.cluster if args.cluster != "first" else None
            if not cluster and cached:
                cluster = cached[0]
            if not cluster or cluster == "first":
                raise SystemExit("--no-submit --job-id needs --cluster <name>")
            cluster, job_id = cluster, args.job_id
        elif cached:
            cluster, job_id = cached
            print(f"reusing cached job {job_id} on {cluster}")
        else:
            raise SystemExit("--no-submit but no cached cluv jobs found.")
    else:
        sbatch_args = [a if a.startswith("-") else f"--{a}" for a in args.sbatch_arg]
        program_args = ["--model", model, "--max-model-len", str(cfg["max_model_len"]),
                        "--max-num-seqs", str(cfg["max_num_seqs"]),
                        "--tool-parser", cfg["tool_parser"],
                        "--reasoning-parser", cfg["reasoning_parser"]]
        cluster, job_id = await submit_first(args.cluster, sbatch_args,
                                             program_args, autocommit=args.autocommit)

    node = await get_node(cluster, job_id)

    if not args.no_wait:
        ok = await wait_for_endpoint(cluster, node, port, args.endpoint_timeout)
        if not ok and not args.launch:
            print("Endpoint not ready; tunnel/config steps below still valid once it is.")

    # Tunnel BEFORE patching: context autodetect reads max_model_len via localhost.
    if not args.no_tunnel:
        start_tunnel(cluster, node, port)

    patch_opencode_config(port, model, output_tokens=cfg["output_tokens"])

    print(http_get(f"http://localhost:{port}/v1/models")[:300])
    print(f"\ncompute: ssh {cluster}  # then: curl http://{node}:{port}/v1/models")
    print(f"job: squeue -j {job_id} -o '%N' on {cluster}")

    if args.launch:
        opencode = shutil.which("opencode")
        if not opencode:
            raise SystemExit("opencode binary not found on PATH.")
        print(f"launching opencode (model {model} via localhost:{port}) ...")
        os.execvp(opencode, [opencode])
    else:
        print("\nre-run with --launch to exec opencode, or run `opencode` yourself.")


if __name__ == "__main__":
    asyncio.run(main())
