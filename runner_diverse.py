#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast, resilient OpenBench runner.

- Auto-resolves benchmark names across builds (underscore/hyphen, mrcr variants).
- Optional parallel execution across benchmarks (--parallel).
- "Fast" mode with conservative per-task --max-tokens caps (--fast).
- Resolves openai/* served model id once, optional 1-token warmup (--warmup).
- Graceful Ctrl+C (soft cancel, then hard abort).
- Writes consolidated JSON + markdown table.

Usage examples:

  # quick smoke
  python runner_diverse.py \
    --model openai/24b-kto \
    --base-url http://localhost:8001/v1 \
    --only mmlu,gpqa_diamond,openai_mrcr \
    --limit 50 --fast --parallel 2 --batch 64 --warmup --resume

  # skip graders if you lack API keys
  python runner_diverse.py --skip hle_text,simpleqa,math_500 ...
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------- Tasks you asked for ----------------
TASKS = [
    "mmlu", "mmlu_pro", "gpqa_diamond",
    "hle_text",
    "humaneval", "jsonschemabench",
    "aime_2023_I", "aime_2023_II",
    "aime_2024", "aime_2024_I", "aime_2024_II",
    "aime_2025", "aime_2025_II",
    "math_500",
    "simpleqa",
    "mrcr",
]

# Some builds need a sandbox for humaneval; allow disabling via env.
# Set RUNNER_DISABLE_SANDBOX=1 to skip forcing docker.
_NEEDS_SANDBOX_BASE = {"humaneval"}
NEEDS_SANDBOX = set() if os.environ.get("RUNNER_DISABLE_SANDBOX") else _NEEDS_SANDBOX_BASE

# These tasks do extra model-graded work; they can be slow.
GRADER_HEAVY  = {"hle_text", "simpleqa", "math_500"}

# Conservative per-task max-token caps for "fast" mode
FAST_MAX_TOKENS: Dict[str, int] = {
    "mmlu": 4, "mmlu_pro": 8, "mmlu-pro": 8, "gpqa_diamond": 8,
    "aime_2023_I": 64, "aime_2023_II": 64,
    "aime_2024": 64, "aime_2024_I": 64, "aime_2024_II": 64,
    "aime_2025": 64, "aime_2025_II": 64,
    "math_500": 128,
    "simpleqa": 32,
    "mrcr": 64, "openai_mrcr": 64, "openai_mrcr_2n": 64, "openai_mrcr_4n": 64, "openai_mrcr_8n": 64,
    "jsonschemabench": 128,
    "hle_text": 64,
    "humaneval": 512,
}

# --------------- Ctrl+C handling ----------------
CANCEL_REQUESTED = False
HARD_CANCEL = False

def sigint_handler(signum, frame):
    global CANCEL_REQUESTED, HARD_CANCEL
    if not CANCEL_REQUESTED:
        CANCEL_REQUESTED = True
        print("\n[runner] Ctrl+C — will stop after this benchmark (press again to hard-abort).", flush=True)
    else:
        HARD_CANCEL = True
        print("\n[runner] Second Ctrl+C — hard-aborting now.", flush=True)

# --------------- Utilities ----------------
def sh(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def bench_supported_flags(bench_cmd: str) -> Set[str]:
    """
    Parse `bench eval --help` and return supported long/short flags.
    Different OpenBench builds expose different knobs; we auto-detect.
    """
    try:
        out = sh([bench_cmd, "eval", "--help"], check=True).stdout
    except Exception:
        out = sh([bench_cmd, "--help"], check=True).stdout

    flags: Set[str] = set()
    for line in out.splitlines():
        for m in re.finditer(r"(--[a-zA-Z0-9\-]+)", line):
            flags.add(m.group(1))
        if re.search(r"(^|\s)-M(\s|,|$)", line):
            flags.add("-M")
    return flags

def bench_available_tasks(bench_cmd: str) -> Set[str]:
    """Get available benchmark ids from `bench list`."""
    try:
        out = sh([bench_cmd, "list"], check=True).stdout.strip()
    except Exception:
        return set()
    # often a single line, comma/space separated
    tokens = re.split(r"[\s,]+", out)
    return {t for t in (tok.strip() for tok in tokens) if t}

def resolve_bench_name(requested: str, available: Set[str]) -> str:
    """Map friendly names to build-specific ids."""
    if requested in available:
        return requested
    # underscore ↔ hyphen
    a = requested.replace("_", "-")
    if a in available:
        return a
    b = requested.replace("-", "_")
    if b in available:
        return b
    # mrcr family fallback
    if requested.lower() == "mrcr":
        for cand in ["openai_mrcr", "openai_mrcr_2n", "openai_mrcr_4n", "openai_mrcr_8n"]:
            if cand in available:
                return cand
    # loose contains (last resort)
    norm_req = re.sub(r"[_\-]", "", requested.lower())
    for cand in available:
        if norm_req and norm_req in re.sub(r"[_\-]", "", cand.lower()):
            return cand
    return requested  # let it fail loudly

def preflight(bench_cmd: str, base_url: Optional[str]) -> None:
    # CLI present?
    try:
        sh([bench_cmd, "list"], check=False)
    except FileNotFoundError:
        raise RuntimeError("`bench` not found. Install/activate OpenBench first.")
    # vLLM OpenAI endpoint reachable?
    if base_url:
        url = base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status >= 400:
                    raise RuntimeError(f"Model base URL returned HTTP {r.status}: {url}")
        except Exception as e:
            raise RuntimeError(f"Cannot reach {url}: {e}")

def resolve_openai_model_id(base_url: str, user_suffix: Optional[str]) -> str:
    """
    Resolve actual model id advertised by vLLM (/v1/models).
    If one model is served, use that. If multiple, prefer exact match on user_suffix.
    """
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=3) as r:
        data = json.loads(r.read().decode("utf-8"))
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise RuntimeError(f"No models advertised by {url}")
    if user_suffix and user_suffix in ids:
        return user_suffix
    if len(ids) == 1:
        return ids[0]
    if user_suffix:
        for mid in ids:
            if user_suffix in mid:
                return mid
    return ids[0]

def warmup_model(base_url: str, model_id: str, api_key: Optional[str]) -> None:
    """Send a tiny chat completion to warm up the server & GPU kernels."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status >= 400:
                raise RuntimeError(f"warmup HTTP {r.status}")
    except Exception as e:
        print(f"[warn] Warmup failed (ignored): {e}", flush=True)

def choose_primary_metric(metrics: Dict[str, Any]) -> Optional[str]:
    if not isinstance(metrics, dict):
        return None
    preferred = ["acc", "accuracy", "pass@1", "pass1", "score", "auc", "exact_match", "compliance_rate"]
    for k in preferred:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k
    return None

def summarize_row(bench: str, res: Dict[str, Any]) -> Dict[str, Any]:
    m = res.get("metrics", {}) if isinstance(res, dict) else {}
    primary = choose_primary_metric(m)
    n = None
    if isinstance(res, dict):
        n = res.get("num_samples") or (res.get("samples") or {}).get("count") or res.get("n")
    return {"benchmark": bench, "metric": primary or "", "value": m.get(primary), "n": n}

def write_md(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = ["| Benchmark | Metric | Value | N |", "|---|---:|---:|---:|"]
    for r in rows:
        v = r["value"]
        vstr = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")
        nstr = str(r["n"]) if r["n"] is not None else ""
        lines.append(f"| `{r['benchmark']}` | {r['metric']} | {vstr} | {nstr} |")
    path.write_text("\n".join(lines), encoding="utf-8")

def normalize_eval_result(data: Any) -> Dict[str, Any]:
    """Extract a final aggregate with 'metrics' from various OpenBench outputs."""
    if isinstance(data, dict) and "metrics" in data:
        return data
    if isinstance(data, list):
        for item in reversed(data):
            if isinstance(item, dict) and "metrics" in item:
                return item
        for item in reversed(data):
            if isinstance(item, dict):
                return item
    return {"metrics": {}, "raw": data}

# --------------- Core executor ----------------
def build_cmd_for_task(
    supported: Set[str],
    bench_cmd: str,
    bench_name: str,
    model: str,
    base_url: Optional[str],
    batch: int,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: Optional[int],
    limit: Optional[str],
    extra_model_args: List[str],
    timeout_s: int,
    log_filename_no_ext: str,  # e.g. "mmlu.eval" -> tool will add .json
) -> Tuple[List[str], Path]:
    cmd = [bench_cmd, "eval", bench_name, "--model", model]

    if "--temperature" in supported:
        cmd += ["--temperature", str(temperature)]
    if "--top-p" in supported:
        cmd += ["--top-p", str(top_p)]
    if "--timeout" in supported:
        cmd += ["--timeout", str(timeout_s)]
    if "--max-connections" in supported:
        cmd += ["--max-connections", str(batch)]
    if max_tokens is not None and "--max-tokens" in supported:
        cmd += ["--max-tokens", str(max_tokens)]

    # logging
    json_target = Path(f"{log_filename_no_ext}.json")
    if "--log-format" in supported:
        cmd += ["--log-format", "json"]
    if "--logfile" in supported:
        cmd += ["--logfile", str(log_filename_no_ext)]
    if "--display" in supported:
        cmd += ["--display", "none"]

    # base url
    if base_url and "--model-base-url" in supported:
        cmd += ["--model-base-url", base_url]

    # misc
    if seed is not None and "--seed" in supported:
        cmd += ["--seed", str(seed)]
    if limit is not None and "--limit" in supported:
        cmd += ["--limit", str(limit)]
    if bench_name in NEEDS_SANDBOX and "--sandbox" in supported:
        cmd += ["--sandbox", "docker"]

    # extra model args
    if "-M" in supported:
        for m in extra_model_args:
            cmd += ["-M", m]

    return cmd, json_target

def run_bench(
    supported: Set[str],
    bench_cmd: str,
    bench_name: str,
    model: str,
    base_url: Optional[str],
    out_dir: Path,
    batch: int,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    seed: Optional[int],
    limit: Optional[str],
    extra_model_args: List[str],
    timeout_s: int,
    openai_api_key: Optional[str],
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = out_dir / f"{bench_name}.stdout.log"
    stderr_log = out_dir / f"{bench_name}.stderr.log"

    log_stem = bench_name + ".eval"  # let OpenBench append .json
    cmd, expected_json_rel = build_cmd_for_task(
        supported=supported,
        bench_cmd=bench_cmd,
        bench_name=bench_name,
        model=model,
        base_url=base_url,
        batch=batch,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        limit=limit,
        extra_model_args=extra_model_args,
        timeout_s=timeout_s,
        log_filename_no_ext=log_stem,
    )

    print(f"[runner] Launching: {' '.join(cmd)}", flush=True)

    # Environment: force logs into out_dir and set OPENAI_API_KEY if using OpenAI provider
    env = os.environ.copy()
    env.setdefault("INSPECT_LOG_DIR", str(out_dir.resolve()))
    if model.startswith("openai/"):
        env.setdefault("OPENAI_API_KEY", openai_api_key or "dummy")

    with open(stdout_log, "w", encoding="utf-8") as so, open(stderr_log, "w", encoding="utf-8") as se:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        try:
            while True:
                if HARD_CANCEL:
                    proc.kill()
                    raise KeyboardInterrupt("Hard abort")

                o = proc.stdout.readline() if proc.stdout else ""
                e = proc.stderr.readline() if proc.stderr else ""
                if o: so.write(o)
                if e: se.write(e)

                if proc.poll() is not None:
                    # drain
                    if proc.stdout:
                        rest = proc.stdout.read() or ""
                        if rest: so.write(rest)
                    if proc.stderr:
                        rest = proc.stderr.read() or ""
                        if rest: se.write(rest)
                    break
                time.sleep(0.05)

            if proc.returncode != 0:
                tail = ""
                try:
                    tail = "\n".join(stderr_log.read_text(encoding="utf-8").splitlines()[-80:])
                except Exception:
                    pass
                raise RuntimeError(f"`bench eval {bench_name}` exited {proc.returncode}. Stderr tail:\n{tail}")

            # Because we set INSPECT_LOG_DIR and gave a simple stem, the tool should create:
            # <out_dir>/<bench_name>.eval.json
            expected_json = (out_dir / expected_json_rel).resolve()
            if not expected_json.exists():
                candidates = sorted(out_dir.glob(f"{bench_name}.eval*.json"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    raise RuntimeError(f"No JSON logfile found for {bench_name} in {out_dir}.")
                expected_json = candidates[0]

            data = json.loads(expected_json.read_text(encoding="utf-8"))
            data = normalize_eval_result(data)
            return data

        finally:
            try:
                proc.terminate()
            except Exception:
                pass

# --------------- Main ----------------
def main():
    signal.signal(signal.SIGINT, sigint_handler)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Provider/model (e.g., openai/24b-kto). Use openai/* with --base-url.")
    p.add_argument("--base-url", default=os.environ.get("BENCH_MODEL_BASE_URL"),
                   help="OpenAI-compatible base URL, e.g., http://localhost:8001/v1")
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", default=None)
    p.add_argument("--timeout", type=int, default=10000)
    p.add_argument("--extra-model-arg", "-M", action="append", default=[])
    p.add_argument("--only", default=None)
    p.add_argument("--skip", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bench-cmd", default="bench")
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--parallel", type=int, default=int(os.environ.get("RUN_PARALLEL", "1")),
                   help="Run up to N benchmarks concurrently (default: 1). Total inflight ≈ parallel*batch.")
    p.add_argument("--fast", action="store_true",
                   help="Use conservative per-task max-token caps (ignored if --max-tokens is set).")
    p.add_argument("--warmup", action="store_true",
                   help="Send a 1-token request to warm the model before the first benchmark.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature-detect flags and preflight connectivity
    try:
        supported = bench_supported_flags(args.bench_cmd)
        preflight(args.bench_cmd, args.base_url)
        if "--max-connections" not in supported:
            print("[note] This OpenBench build lacks --max-connections; using default concurrency.", flush=True)
    except Exception as e:
        print(f"[fatal] Preflight failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve openai/* model id ONCE, optionally warm up
    if args.model.startswith("openai/") and args.base_url:
        suffix = args.model.split("/", 1)[1] if "/" in args.model else None
        try:
            resolved = resolve_openai_model_id(args.base_url, suffix)
            if resolved != suffix:
                print(f"[runner] Resolved model id: {suffix!r} -> {resolved!r}")
            args.model = f"openai/{resolved}"
        except Exception as e:
            print(f"[warn] Could not resolve model id from {args.base_url}/models: {e}. Using {args.model} as-is.")
        if args.warmup:
            try:
                warmup_model(args.base_url, args.model.split("/", 1)[1], args.openai_api_key)
            except Exception as e:
                print(f"[warn] Warmup error (ignored): {e}")

    # Task selection
    tasks = TASKS[:]
    if args.only:
        keep = {t.strip() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t in keep]
    if args.skip:
        drop = {t.strip() for t in args.skip.split(",") if t.strip()}
        tasks = [t for t in tasks if t not in drop]

    if any(t in GRADER_HEAVY for t in tasks) and not os.environ.get("OPENAI_API_KEY"):
        print("[warn] Some tasks use an OpenAI grader by default: hle_text, simpleqa, math_500. "
              "Set OPENAI_API_KEY for those to run, or skip them.", flush=True)

    # Fetch available benchmark ids and pre-resolve
    available = bench_available_tasks(args.bench_cmd)
    resolved_tasks: List[str] = []
    for name in tasks:
        r = resolve_bench_name(name, available) if available else name
        if r != name:
            print(f"[runner] Using benchmark alias: {name!r} -> {r!r}")
        resolved_tasks.append(r)
    tasks = resolved_tasks

    consolidated: Dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "batch": args.batch,
        "parallel": args.parallel,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "limit": args.limit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
        "errors": {}
    }
    rows: List[Dict[str, Any]] = []
    exit_code = 0
    lock = threading.Lock()

    # If we parallelize, scale per-bench connections so total inflight stays reasonable.
    per_bench_batch = max(1, args.batch // max(1, args.parallel))
    if args.parallel > 1 and "--max-connections" in supported:
        print(f"[runner] Parallel={args.parallel} -> per-bench --max-connections={per_bench_batch} "
              f"(total≈{per_bench_batch*args.parallel})", flush=True)

    def effective_max_tokens(name: str) -> Optional[int]:
        if args.max_tokens is not None:
            return args.max_tokens
        if args.fast:
            return FAST_MAX_TOKENS.get(name)
        return None

    def do_one(bench_name: str):
        nonlocal exit_code
        if CANCEL_REQUESTED:
            return
        # resume?
        existing = out_dir / f"{bench_name}.eval.json"
        if args.resume and existing.exists():
            try:
                data = json.loads(existing.read_text(encoding="utf-8"))
                with lock:
                    consolidated["results"][bench_name] = data
                    rows.append(summarize_row(bench_name, data))
                print(f"[runner] Resume: loaded {bench_name} from {existing}")
                return
            except Exception:
                pass
        t0 = time.time()
        try:
            data = run_bench(
                supported=supported,
                bench_cmd=args.bench_cmd,
                bench_name=bench_name,
                model=args.model,
                base_url=args.base_url,
                out_dir=out_dir,
                batch=per_bench_batch if "--max-connections" in supported else args.batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=effective_max_tokens(bench_name),
                seed=args.seed,
                limit=args.limit,
                extra_model_args=args.extra_model_arg,
                timeout_s=args.timeout,
                openai_api_key=args.openai_api_key,
            )
            dt = time.time() - t0
            with lock:
                consolidated["results"][bench_name] = data
                rows.append(summarize_row(bench_name, data))
            print(f"[runner] {bench_name} finished in {dt:.1f}s")
        except KeyboardInterrupt:
            print("[runner] Interrupted by user.")
            exit_code = 130
        except Exception as e:
            with lock:
                exit_code = 1
                consolidated["errors"][bench_name] = str(e)
            print(f"[error] {bench_name}: {e}", file=sys.stderr)

    if args.parallel <= 1:
        for bench_name in tasks:
            if CANCEL_REQUESTED:
                print("[runner] Cancel requested: skipping remaining tasks.")
                break
            do_one(bench_name)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(do_one, t) for t in tasks]
            for _ in as_completed(futs):
                if CANCEL_REQUESTED:
                    break

    consolidated_name = f"consolidated_{args.model.replace("/", "_")}.json"
    results_name = f"results_{args.model.replace("/", "_")}.md"
    # Always write artifacts
    (out_dir / consolidated_name).write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_md(rows, out_dir / results_name)
    print(f"\n[runner] Wrote: {out_dir/consolidated_name}")
    print(f"[runner] Wrote: {out_dir/results_name}")
    if consolidated["errors"]:
        print("[runner] Some tasks failed. Check per-task logs (*.stderr.log) and consolidated.json:errors.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
