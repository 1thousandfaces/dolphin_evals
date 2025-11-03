#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner_diverse.py
Thin wrapper around lm-evaluation-harness for "diverse evals".

- Uses vLLM's OpenAI-compatible endpoint via lm-eval's `openai-chat` model.
- Sets OPENAI_API_BASE / OPENAI_API_KEY envs for you.
- Batching: --batch_size auto (vLLM handles real concurrency).
- Writes one summary JSON per run, plus per-benchmark result rows.

Usage:
  python runner_diverse.py --config config.toml --set quick
  python runner_diverse.py --config config.toml --benchmarks gsm8k hellaswag --limit 100
"""

import argparse, json, logging, os, shlex, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # py311+
except Exception:
    import tomli as tomllib  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("runner_diverse")

# ---------------- Bench registry ----------------

BENCHMARKS = {
    # Math & Reasoning
    "gsm8k":        {"task": "gsm8k",        "metric": "exact_match", "fewshot": 5},
    "arc_challenge":{"task": "arc_challenge","metric": "acc_norm",    "fewshot": 25},
    "arc_easy":     {"task": "arc_easy",     "metric": "acc_norm",    "fewshot": 25},
    # Commonsense
    "hellaswag":    {"task": "hellaswag",    "metric": "acc_norm",    "fewshot": 10},
    "winogrande":   {"task": "winogrande",   "metric": "acc",         "fewshot": 5},
    "piqa":         {"task": "piqa",         "metric": "acc_norm",    "fewshot": 5},
    # Truth / factuality
    "truthfulqa_mc2":{"task":"truthfulqa_mc2","metric":"acc",         "fewshot": 0},
    # Reading comp
    "boolq":        {"task": "boolq",        "metric": "acc",         "fewshot": 0},
    "squad_v2":     {"task": "squad_v2",     "metric": "exact",       "fewshot": 0},
    # BBH subset (few-shot CoT variant in harness)
    "bbh":          {"task": "bbh",          "metric": "acc",         "fewshot": 3},
}

BENCHMARK_SETS = {
    "quick":        ["gsm8k", "arc_challenge", "hellaswag", "truthfulqa_mc2"],
    "reasoning":    ["gsm8k", "arc_challenge", "arc_easy", "bbh"],
    "commonsense":  ["hellaswag", "winogrande", "piqa", "boolq"],
    "comprehensive":["gsm8k","arc_challenge","hellaswag","winogrande","truthfulqa_mc2","boolq","piqa"],
    "all":          list(BENCHMARKS.keys()),
}

# ---------------- Config ----------------

@dataclass
class Backend:
    endpoint: str
    endpoint_api_key: str = ""

@dataclass
class Model:
    name: str
    endpoint_model_name: str = ""

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    b = data["backend"]; m = data["model"]
    return {
        "backend": Backend(endpoint=b["endpoint"], endpoint_api_key=b.get("endpoint_api_key","")),
        "model":   Model(name=m["name"], endpoint_model_name=m.get("endpoint_model_name", m["name"])),
    }

# ---------------- lm-eval integration ----------------

def ensure_lm_eval():
    try:
        subprocess.run(["lm_eval", "--help"], capture_output=True, check=False, timeout=5)
    except FileNotFoundError:
        log.error("lm-evaluation-harness CLI 'lm_eval' not found. Install: pip install 'lm-eval[api]'")
        sys.exit(1)

def run_one(cfg: Dict[str,Any], bench: str, out_root: Path, limit: Optional[int]) -> Dict[str,Any]:
    info = BENCHMARKS[bench]
    task = info["task"]; few = info["fewshot"]

    outdir = out_root / f"{int(time.time())}-{bench}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Env for OpenAI-compatible backend
    model_name = "vllm"

    # vLLM engine args (tweak to your box)
    # pretrained MUST be a local HF id or folder with weights (yours is ./models/24b-kto)
    model_args = (
        f"pretrained={cfg['model'].endpoint_model_name},"  # e.g. ./models/24b-kto
        "dtype=float16,"                                    # or bfloat16
        "tensor_parallel_size=1,"                           # >1 if multi-GPU
        "gpu_memory_utilization=0.9,"                       # pack batches
        "max_model_len=32768"                               # adjust if needed
    )
    cmd = [
        "lm_eval",
        "--model", model_name,
        "--model_args", model_args,
        "--tasks", task,
        "--num_fewshot", str(few),
        "--batch_size", "auto",              # vllm supports auto batching
        "--output_path", str(outdir),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    log.info("lm-eval: %s", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error("lm-eval failed for %s", bench)
        if p.stderr:
            log.error("stderr: %s", p.stderr.strip())
        raise RuntimeError(f"lm-eval failed ({bench})")

    results_json = next((outdir.glob("results_*.json"))).__str__()
    with open(results_json, "r") as f:
        data = json.load(f)

    task_block = data["results"].get(task, {})
    metric = info["metric"]
    score = task_block.get(metric)
    if score is None:
        for k, v in task_block.items():
            if isinstance(v, (int, float)):
                score, metric = v, k
                break

    n = data.get("n-shot", {}).get(task) or data.get("versions",{}).get(task,{}).get("num_samp")
    return {
        "benchmark": bench,
        "task": task,
        "metric": metric,
        "score": score,
        "num_fewshot": few,
        "samples": n,
        "output_dir": outdir.as_posix(),
    }


# ---------------- Runner ----------------

def run_all(cfg: Dict[str,Any], benches: List[str], limit: Optional[int], out_root: Path) -> Dict[str,Any]:
    ensure_lm_eval()
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for b in benches:
        try:
            results.append(run_one(cfg, b, out_root, limit))
        except Exception as e:
            results.append({"benchmark": b, "task": BENCHMARKS[b]["task"], "metric": BENCHMARKS[b]["metric"],
                            "score": None, "error": str(e)})

    return {
        "model": cfg["model"].name,
        "endpoint_model_name": cfg["model"].endpoint_model_name,
        "timestamp": int(time.time()),
        "results": results,
    }

def save_summary(cfg: Dict[str,Any], summary: Dict[str,Any], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    model_safe = cfg["model"].name.replace("/", "_")
    out_file = out_root / f"{summary['timestamp']}-{model_safe}-diverse.json"
    with open(out_file, "w") as f: json.dump(summary, f, indent=2)
    log.info("Saved summary: %s", out_file)
    return out_file

# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Run diverse benchmarks via lm-eval")
    p.add_argument("--config", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--benchmarks", nargs="+")
    g.add_argument("--set", choices=BENCHMARK_SETS.keys())
    p.add_argument("--limit", type=int)
    p.add_argument("--output-dir", default="results_diverse")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    benches = BENCHMARK_SETS[args.set] if args.set else args.benchmarks
    invalid = [b for b in benches if b not in BENCHMARKS]
    if invalid:
        log.error("Invalid benchmarks: %s", invalid); return 1

    cfg = load_cfg(args.config)
    out_root = Path(args.output_dir)

    print("Diverse Eval Runner")
    print("="*60)
    print(f"Benchmarks: {', '.join(benches)}")
    if args.limit: print(f"Limit per task: {args.limit}")
    print("="*60)

    summary = run_all(cfg, benches, args.limit, out_root)
    out_file = save_summary(cfg, summary, out_root)

    # concise console summary
    scores = []
    for r in summary["results"]:
        s = r.get("score")
        if s is None:
            print(f"{r['benchmark']:20s}: FAILED ({r.get('error','')})")
        else:
            scores.append(s)
            print(f"{r['benchmark']:20s}: {s*100:6.2f}%  [{r['metric']}]")
    if scores:
        avg = sum(scores)/len(scores)
        print("-"*60)
        print(f"{'Average':20s}: {avg*100:6.2f}%")
    print(f"\nSummary saved: {out_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
