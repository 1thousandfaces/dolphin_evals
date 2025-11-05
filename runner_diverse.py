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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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

NEEDS_SANDBOX = {"humaneval"}                         # code execution
GRADER_HEAVY  = {"hle_text", "simpleqa", "math_500"}  # model-graded by default

# --------------- Ctrl+C handling ----------------
CANCEL_REQUESTED = False
HARD_CANCEL = False

def sigint_handler(signum, frame):
    global CANCEL_REQUESTED, HARD_CANCEL
    if not CANCEL_REQUESTED:
        CANCEL_REQUESTED = True
        print("\n[runner] Ctrl+C — will stop scheduling new benchmarks and wait for currently running ones (press again to hard-abort).", flush=True)
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
        # collect long flags
        for m in re.finditer(r"(--[a-zA-Z0-9\-]+)", line):
            flags.add(m.group(1))
        # collect short -M if present
        if re.search(r"(^|\s)-M(\s|,|$)", line):
            flags.add("-M")
    return flags

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
    # fallback to first with best-effort substring match
    if user_suffix:
        for mid in ids:
            if user_suffix in mid:
                return mid
    # last resort: first id
    return ids[0]

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
    # Different builds store sample counts differently
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
    """
    OpenBench may emit a single dict or a list of entries. We want the
    final aggregate that includes 'metrics' (accuracy/score/etc).
    """
    # Case 1: already a dict with metrics
    if isinstance(data, dict) and "metrics" in data:
        return data

    # Case 2: list of log records; pick the last dict that has metrics
    if isinstance(data, list):
        for item in reversed(data):
            if isinstance(item, dict) and "metrics" in item:
                return item
        # Fallback: last dict-ish record
        for item in reversed(data):
            if isinstance(item, dict):
                return item

    # Last resort: wrap bare values
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
    # logging
    json_target = Path(f"{log_filename_no_ext}.json")  # OpenBench sometimes appends .json; INSPECT_LOG_DIR will pin location
    if "--log-format" in supported:
        cmd += ["--log-format", "json"]
    if "--logfile" in supported:
        cmd += ["--logfile", str(log_filename_no_ext)]  # give it without .json to avoid .json.json
    if "--display" in supported:
        cmd += ["--display", "none"]
    # base url
    if base_url and "--model-base-url" in supported:
        cmd += ["--model-base-url", base_url]
    # misc
    if seed is not None and "--seed" in supported:
        cmd += ["--seed", str(seed)]
    if max_tokens is not None and "--max-tokens" in supported:
        cmd += ["--max-tokens", str(max_tokens)]
    if limit is not None and "--limit" in supported:
        cmd += ["--limit", str(limit)]
    if bench_name in NEEDS_SANDBOX and "--sandbox" in supported:
        cmd += ["--sandbox", "docker"]
    # extra model args
    if "-M" in supported:
        for m in extra_model_args:
            cmd += ["-M", m]

    # Expected file we will read (because we set INSPECT_LOG_DIR)
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
        # Fallback for older OpenBench builds that don't accept --model-base-url
        if base_url and "--model-base-url" not in supported:
            env.setdefault("OPENAI_BASE_URL", base_url)

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
            # In case some builds add another ".json" suffix, search fallback.
            if not expected_json.exists():
                candidates = sorted(out_dir.glob(f"{bench_name}.eval*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
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
    p.add_argument("--base-url", default=os.environ.get("BENCH_MODEL_BASE_URL"), help="OpenAI-compatible base URL, e.g., http://localhost:8001/v1")
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
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))  # for OpenAI provider (vLLM ignores content unless auth enabled)

    # NEW: parallelism controls
    p.add_argument("--jobs", type=int, default=min(4, (os.cpu_count() or 4)),
                   help="How many benchmarks to run in parallel (non-heavy). Default 4.")
    p.add_argument("--heavy-jobs", type=int, default=1,
                   help="Parallelism for heavy grader tasks (hle_text, simpleqa, math_500). Default 1.")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature-detect flags and preflight connectivity
    try:
        supported = bench_supported_flags(args.bench_cmd)
        preflight(args.bench_cmd, args.base_url)
        if "--max-connections" not in supported:
            print("[note] This OpenBench build lacks --max-connections; using tool defaults for per-process concurrency.", flush=True)
    except Exception as e:
        print(f"[fatal] Preflight failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve model id ONCE if using OpenAI provider + base-url
    resolved_model = args.model
    if args.model.startswith("openai/") and args.base_url:
        suffix = args.model.split("/", 1)[1] if "/" in args.model else None
        try:
            mid = resolve_openai_model_id(args.base_url, suffix)
            if mid != suffix:
                print(f"[runner] Resolved model id once: {suffix!r} -> {mid!r}", flush=True)
            resolved_model = f"openai/{mid}"
        except Exception as e:
            print(f"[warn] Could not resolve model id from {args.base_url}/models: {e}. Using {args.model} as-is.", flush=True)
            resolved_model = args.model

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

    consolidated: Dict[str, Any] = {
        "model": resolved_model,
        "base_url": args.base_url,
        "batch": args.batch,
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

    # Resume support: pre-load any existing results and shrink task list
    to_run: List[str] = []
    for bench_name in tasks:
        existing = out_dir / f"{bench_name}.eval.json"
        if args.resume and existing.exists():
            try:
                data = json.loads(existing.read_text(encoding="utf-8"))
                consolidated["results"][bench_name] = data
                rows.append(summarize_row(bench_name, data))
                print(f"[runner] Resume: loaded {bench_name} from {existing}")
            except Exception:
                to_run.append(bench_name)
        else:
            to_run.append(bench_name)

    if not to_run:
        # Still write artifacts to confirm consolidation
        (out_dir / "consolidated.json").write_text(json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8")
        write_md(rows, out_dir / "results.md")
        print(f"\n[runner] Wrote: {out_dir/'consolidated.json'}")
        print(f"[runner] Wrote: {out_dir/'results.md'}")
        sys.exit(0)

    normal_tasks = [t for t in to_run if t not in GRADER_HEAVY]
    heavy_tasks  = [t for t in to_run if t in GRADER_HEAVY]

    print(f"[runner] Parallel mode: jobs={args.jobs}, heavy-jobs={args.heavy_jobs}. "
          f"Normal: {len(normal_tasks)}; Heavy: {len(heavy_tasks)}", flush=True)

    # Closure to run one task
    def run_task(bench_name: str) -> Dict[str, Any]:
        return run_bench(
            supported=supported,
            bench_cmd=args.bench_cmd,
            bench_name=bench_name,
            model=resolved_model,
            base_url=args.base_url,
            out_dir=out_dir,
            batch=args.batch,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            limit=args.limit,
            extra_model_args=args.extra_model_arg,
            timeout_s=args.timeout,
            openai_api_key=args.openai_api_key,
        )

    # Incremental scheduler: only keep at most max_workers running; stop scheduling on first Ctrl+C.
    def run_pool(task_list: List[str], max_workers: int) -> None:
        nonlocal exit_code
        if not task_list or max_workers <= 0:
            return
        iterator = iter(task_list)
        active = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Seed initial submissions
            try:
                while len(active) < max_workers:
                    if CANCEL_REQUESTED:
                        break
                    bench_name = next(iterator)
                    fut = ex.submit(run_task, bench_name)
                    active[fut] = bench_name
            except StopIteration:
                pass

            while active:
                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    bench_name = active.pop(fut)
                    try:
                        data = fut.result()
                        consolidated["results"][bench_name] = data
                        rows.append(summarize_row(bench_name, data))
                    except KeyboardInterrupt:
                        print("[runner] Interrupted by user.")
                        exit_code = 130
                        return
                    except Exception as e:
                        exit_code = 1
                        consolidated["errors"][bench_name] = str(e)
                        print(f"[error] {bench_name}: {e}", file=sys.stderr)

                # Top-up the queue unless cancellation requested
                try:
                    while not CANCEL_REQUESTED and len(active) < max_workers:
                        bench_name = next(iterator)
                        fut = ex.submit(run_task, bench_name)
                        active[fut] = bench_name
                except StopIteration:
                    pass

    # Run normal pool then heavy pool (unless canceled)
    run_pool(normal_tasks, args.jobs)
    if CANCEL_REQUESTED and heavy_tasks:
        print("[runner] Cancel requested: skipping heavy tasks.")
    else:
        run_pool(heavy_tasks, args.heavy_jobs)
    # Make it filesystem-safe (keep alnum, dot, dash, underscore; replace others with "_")
    # Choose a clean filename stem that ignores any provider prefix like "openai/"
    if "/" in args.model:
        _, model_suffix = args.model.split("/", 1)
    else:
        model_suffix = args.model
        
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model_suffix)
    
    consolidated_name = f"consolidated_{safe_stem}.json"
    results_name      = f"results_{safe_stem}.md"
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
