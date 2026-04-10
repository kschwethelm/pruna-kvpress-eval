"""Experiment 2: Downstream task evaluation for KVPress via Pruna.

Validates that KV cache compression through Pruna's smash() interface
does not degrade model quality on long-context tasks, by running
lm-eval-harness on both baseline and compressed models.

Two sub-experiments:
1. Press-type comparison: all 5 press types at a fixed ratio (0.5).
2. KnormPress ratio sweep: 5 compression ratios with the default press.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruna import SmashConfig, smash
from _utils import select_attn_implementation

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TASKS = [
    "longbench_narrativeqa",
    "longbench_qasper",
    "longbench_hotpotqa",
]

# Sub-experiment 1: compare press types at a fixed compression ratio.
PRESS_COMPARISON_TYPES = [
    "SnapKVPress",
    "ExpectedAttentionPress",
    "StreamingLLMPress",
    "KnormPress",
    "RandomPress",
]
PRESS_COMPARISON_RATIO = 0.5

# Sub-experiment 2: ratio sweep for the main press types.
RATIO_SWEEP_PRESSES = ["KnormPress", "SnapKVPress", "ExpectedAttentionPress"]
RATIO_SWEEP_RATIOS = [0.2, 0.4, 0.6, 0.8]  # 0.5 already covered by sub-experiment 1

EVAL_LIMIT = None  # None = use all samples (tasks have ~200 each)


def cleanup():
    """Force full GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()


def load_model():
    """Load model."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=select_attn_implementation()
    ).to("cuda")


def run_eval(model, tokenizer, tasks, limit):
    """Run lm-eval-harness on a model and return results dict."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        batch_size=1,
    )
    # Extract task-level metrics
    task_results = {}
    for task_name, task_data in results["results"].items():
        # Get the primary metric for each task
        metrics = {}
        for key, value in task_data.items():
            if key.endswith(",none") and not key.startswith("alias"):
                metric_name = key.replace(",none", "")
                if isinstance(value, (int, float)):
                    metrics[metric_name] = round(value, 4)
        task_results[task_name] = metrics
    return task_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/downstream_eval.json")
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    def save():
        """Persist current results so partial progress survives crashes."""
        output_path.write_text(json.dumps(results, indent=2))

    save()  # Clear any stale file from a previous run.

    # Baseline
    print("=== Baseline (no compression) ===")
    cfg_start = time.perf_counter()
    model = load_model()
    task_results = run_eval(model, tokenizer, args.tasks, args.limit)
    elapsed = time.perf_counter() - cfg_start
    print(f"  {task_results}  [timing] {elapsed:.1f}s")
    results.append({
        "press_type": "none",
        "compression_ratio": 0.0,
        "tasks": task_results,
        "elapsed_s": round(elapsed, 2),
    })
    save()
    del model
    cleanup()

    def run_config(press_type: str, ratio: float):
        print(f"=== {press_type} @ ratio={ratio} ===")
        cfg_start = time.perf_counter()
        model = load_model()
        config = SmashConfig()
        config.add({"kvpress": {"press_type": press_type, "compression_ratio": ratio}})
        smashed = smash(model, smash_config=config)

        task_results = run_eval(smashed, tokenizer, args.tasks, args.limit)
        elapsed = time.perf_counter() - cfg_start
        print(f"  {task_results}  [timing] {elapsed:.1f}s")
        results.append({
            "press_type": press_type,
            "compression_ratio": ratio,
            "tasks": task_results,
            "elapsed_s": round(elapsed, 2),
        })
        save()
        del smashed, model
        cleanup()

    # Sub-experiment 1: press-type comparison at fixed ratio
    print(f"\n### Sub-experiment 1: press comparison @ ratio={PRESS_COMPARISON_RATIO} ###")
    for press_type in PRESS_COMPARISON_TYPES:
        run_config(press_type, PRESS_COMPARISON_RATIO)

    # Sub-experiment 2: ratio sweep (0.5 already covered above)
    for press_type in RATIO_SWEEP_PRESSES:
        print(f"\n### Sub-experiment 2: {press_type} ratio sweep ###")
        for ratio in RATIO_SWEEP_RATIOS:
            run_config(press_type, ratio)

    save()
    total = time.perf_counter() - script_start
    print(f"\nResults saved to {output_path}")
    print(f"[timing] downstream_eval total: {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
