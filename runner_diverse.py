#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenBench + vLLM eval runner with:
- batching via --max-connections (drives vLLM's dynamic batching)
- graceful Ctrl+C (first SIGINT: finish current eval if possible; second: hard stop)
- per-task JSON artifacts + consolidated JSON + markdown table

Usage (examples):
  python eval_runner.py \
    --model vllm/meta-llama/Llama-3.1-8B-Instruct \
    --base-url http://localhost:8000/v1 \
    --out-dir runs/llama31-8b \
    --batch 64 \
    --temperature 0.0 \
    --seed 0

You can subset tasks with --only or --skip (comma-separated names).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# ---- Canonical OpenBench benchmark names (per docs) ----
# Knowledge
TASK_MMLU           = "mmlu"          # MMLU
TASK_MMLU_PRO       = "mmlu_pro"      # MMLU-Pro
TASK_GPQA_DIAMOND   = "gpqa_diamond"  # GPQA (graduate-level)

# HLE text-only (no vision)
TASK_HLE_TEXT       = "hle_text"      # HLE_text

# Coding
TASK_HUMANEVAL      = "humaneval"     # 164 problems
TASK_JSONSCHEMABENCH= "jsonschemabench"

# Math
TASK_AIME_2023_I    = "aime_2023_I"
TASK_AIME_2023_II   = "aime_2023_II"
TASK_AIME_2024      = "aime_2024"     # Some builds also have I/II variants
TASK_AIME_2024_I    = "aime_2024_I"
TASK_AIME_2024_II   = "aime_2024_II"
TASK_AIME_2025      = "aime_2025"
TASK_AIME_2025_II   = "aime_2025_II"
TASK_MATH_500       = "math_500"

# Reasoning / factuality
TASK_SIMPLEQA       = "simpleqa"

# Long-context retrieval
TASK_MRCR           = "mrcr"          # OpenAI MRCR (multi-needle)

DEFAULT_TASKS = [
    TASK_MMLU, TASK_MMLU_PRO, TASK_GPQA_DIAMOND,
    TASK_HLE_TEXT,
    TASK_HUMANEVAL, TASK_JSONSCHEMABENCH,
    TASK_AIME_2023_I, TASK_AIME_2023_II,
    TASK_AIME_2024, TASK_AIME_2024_I, TASK_AIME_2024_II,
    TASK_AIME_2025, TASK_AIME_2025_II,
    TASK_MATH_500,
    TASK_SIMPLEQA,
    TASK_MRCR,
]

# Which tasks need a code sandbox (docker) for test execution?
SANDBOX_NEEDED = {TASK_HUMANEVAL}

# Some tasks rely on model-graded scoring; ensure OPENAI_API_KEY is present or override with flags.
GRADER_HEAVY = {TASK_SIMPLEQA, TASK_HLE_TEXT, TASK_MATH_500}

# Global cancel state (for graceful Ctrl+C)
CANCEL_REQUESTED = False
HARD_CANCEL = False

def sigint_handler(signum, frame):
    global CANCEL_REQUESTED, HARD_CANCEL
    if not CANCEL_REQUESTED:
        CANCEL_REQUESTED = True
        print("\n[runner] Caught Ctrl+C — will stop after current benchmark completes (press Ctrl+C again to hard-abort).", flush=True)
    else:
        HARD_CANCEL = True
        print("\n[runner] Second Ctrl+C — hard-aborting now.", flush=True)

def run_one_bench(
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
    per_task_overrides: Dict[str, List[str]],
    timeout_s: int,
) -> Dict[str, Any]:
    """Invoke `bench eval` for a single benchmark, capture JSON, write files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    task_out_json = out_dir / f"{bench_name}.json"
    task_stdout_log = out_dir / f"{bench_name}.stdout.log"
    task_stderr_log = out_dir / f"{bench_name}.stderr.log"

    # Build base command
    cmd = ["bench", "eval", bench_name,
           "--model", model,
           "--max-connections", str(batch),
           "--temperature", str(temperature),
           "--top-p", str(top_p),
           "--timeout", str(timeout_s),
           "--json",
           "--logfile", str(task_out_json)]

    if base_url:
        cmd += ["--model-base-url", base_url]

    if seed is not None:
        cmd += ["--seed", str(seed)]

    if max_tokens is not None:
        cmd += ["--max-tokens", str(max_tokens)]

    if limit is not None:
        # supports number or "start,end"
        cmd += ["--limit", str(limit)]

    # Per-task knobs: e.g., sandbox=docker for HumanEval
    if bench_name in SANDBOX_NEEDED:
        cmd += ["--sandbox", "docker"]

    # Allow passing arbitrary -M model args (e.g., reasoning_effort=high)
    for m in extra_model_args:
        cmd += ["-M", m]

    # Per-task overrides (rarely needed; example placeholder)
    if bench_name in per_task_overrides:
        cmd += per_task_overrides[bench_name]

    print(f"[runner] Launching: {' '.join(cmd)}")
    with open(task_stdout_log, "w", encoding="utf-8") as out_f, open(task_stderr_log, "w", encoding="utf-8") as err_f:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            stdout_chunks = []
            stderr_chunks = []

            # Stream output so logs are flushed (useful if we get interrupted)
            while True:
                if HARD_CANCEL:
                    proc.kill()
                    raise KeyboardInterrupt("Hard-abort requested")

                # Non-blocking-ish reads
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    out_f.write(line)
                    stdout_chunks.append(line)
                err_line = proc.stderr.readline() if proc.stderr else ""
                if err_line:
                    err_f.write(err_line)
                    stderr_chunks.append(err_line)

                if proc.poll() is not None:
                    # drain remaining
                    rest_out = proc.stdout.read() if proc.stdout else ""
                    rest_err = proc.stderr.read() if proc.stderr else ""
                    if rest_out:
                        out_f.write(rest_out)
                        stdout_chunks.append(rest_out)
                    if rest_err:
                        err_f.write(rest_err)
                        stderr_chunks.append(rest_err)
                    break

                # If user requested cancel, we don't kill immediately; we wait for the eval to flush
                if CANCEL_REQUESTED:
                    # No-op: just refrain from starting the next benchmark
                    pass

                time.sleep(0.05)

            rc = proc.returncode
            if rc != 0:
                raise RuntimeError(f"`bench eval {bench_name}` exited with {rc}. See logs: {task_stderr_log}")

            # Parse JSON results: prefer the file we asked for via --logfile
            if task_out_json.exists():
                with open(task_out_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                # Fallback: try capturing from stdout
                whole = "".join(stdout_chunks).strip()
                try:
                    data = json.loads(whole)
                except Exception:
                    raise RuntimeError(f"Could not parse JSON for {bench_name}. See {task_stdout_log}")

            return data

        except KeyboardInterrupt:
            # Try to terminate gracefully
            try:
                proc.terminate()
            except Exception:
                pass
            raise

def choose_primary_metric(metrics: Dict[str, Any]) -> Optional[str]:
    """Pick a sensible primary metric key from a metrics dict."""
    if not metrics or not isinstance(metrics, dict):
        return None
    # Common keys in OB/Inspect tasks
    preferred = ["acc", "accuracy", "pass@1", "pass1", "score", "auc", "exact_match"]
    for k in preferred:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k
    # Fallback: first numeric
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k
    return None

def summarize_for_table(bench_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a minimal summary row for markdown."""
    metrics = result.get("metrics", {})
    primary = choose_primary_metric(metrics)
    return {
        "benchmark": bench_name,
        "primary_metric": primary or "",
        "value": (metrics.get(primary) if primary else None),
        "n": result.get("samples", {}).get("count") or result.get("num_samples") or None,
    }

def write_markdown_table(rows: List[Dict[str, Any]], path: Path):
    # basic GitHub Markdown table
    lines = []
    lines.append("| Benchmark | Metric | Value | N |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        v = r["value"]
        vstr = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")
        nstr = str(r["n"]) if r["n"] is not None else ""
        lines.append(f"| `{r['benchmark']}` | {r['primary_metric']} | {vstr} | {nstr} |")
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Inspect/OpenBench model string, e.g. vllm/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--base-url", default=os.environ.get("BENCH_MODEL_BASE_URL"),
                        help="Base URL for model API (vLLM OpenAI server), e.g. http://localhost:8000/v1")
    parser.add_argument("--out-dir", default="runs/default", help="Output directory for artifacts")
    parser.add_argument("--batch", type=int, default=32, help="Concurrent API calls (drives vLLM batching via Inspect/OpenBench)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", default=None, help="Limit samples: integer or 'start,end'")
    parser.add_argument("--timeout", type=int, default=10000, help="Per-request timeout (seconds)")
    parser.add_argument("--extra-model-arg", "-M", action="append", default=[], help="Pass-through extra model args (repeatable)")
    parser.add_argument("--only", default=None, help="Comma-separated subset of tasks to run")
    parser.add_argument("--skip", default=None, help="Comma-separated tasks to skip")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with an existing JSON result")
    args = parser.parse_args()

    # Install signal handlers
    signal.signal(signal.SIGINT, sigint_handler)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine task list
    tasks = DEFAULT_TASKS[:]
    if args.only:
        keep = {t.strip() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t in keep]
    if args.skip:
        drop = {t.strip() for t in args.skip.split(",") if t.strip()}
        tasks = [t for t in tasks if t not in drop]

    # Optional per-task overrides placeholder (e.g., custom grader or task flags)
    per_task_overrides: Dict[str, List[str]] = {}

    consolidated: Dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "batch": args.batch,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "limit": args.limit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {}
    }

    table_rows = []

    try:
        for bench in tasks:
            if CANCEL_REQUESTED:
                print("[runner] Cancellation requested: skipping remaining tasks.")
                break

            task_file = out_dir / f"{bench}.json"
            if args.resume and task_file.exists():
                print(f"[runner] Skipping {bench} (resume: found {task_file})")
                with open(task_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                consolidated["results"][bench] = data
                table_rows.append(summarize_for_table(bench, data))
                continue

            if bench in GRADER_HEAVY and not os.environ.get("OPENAI_API_KEY"):
                print(f"[warn] {bench} typically uses an OpenAI grader by default — OPENAI_API_KEY not set. "
                      f"Consider exporting it or overriding the grader; continuing anyway.")

            data = run_one_bench(
                bench_name=bench,
                model=args.model,
                base_url=args.base_url,
                out_dir=out_dir,
                batch=args.batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                limit=args.limit,
                extra_model_args=args.extra_model_arg,
                per_task_overrides=per_task_overrides,
                timeout_s=args.timeout,
            )
            consolidated["results"][bench] = data
            table_rows.append(summarize_for_table(bench, data))

    except KeyboardInterrupt:
        print("[runner] Interrupted by user.")
    finally:
        # Write consolidated artifacts
        consolidated_path = out_dir / "consolidated.json"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, ensure_ascii=False, indent=2)

        md_path = out_dir / "results.md"
        write_markdown_table(table_rows, md_path)

        print(f"\n[runner] Wrote consolidated JSON: {consolidated_path}")
        print(f"[runner] Wrote Markdown table:     {md_path}\n")

        if CANCEL_REQUESTED:
            print("[runner] Note: run was cancelled mid-stream; some tasks may be missing or partial.")
        sys.exit(0)

if __name__ == "__main__":
    main()
