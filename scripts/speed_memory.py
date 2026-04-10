"""Experiment 1: Speed and memory benchmark for KVPress via Pruna."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from pruna import SmashConfig, smash
from _utils import select_attn_implementation

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PRESS_TYPE = "KnormPress"  # only one press; all press types give identical memory at a fixed ratio
COMPRESSION_RATIO = 0.5
CONTEXT_LENGTHS = [8192, 16384, 32768, 65536, 131072]
# Ratio sweep at longer contexts; 0.5 is already covered by the main sweep.
RATIO_SWEEP_CONTEXTS = [32768, 65536, 131072]
RATIO_SWEEP_RATIOS = [0.2, 0.4, 0.6, 0.8]
GENERATE_TOKENS = 100
NUM_WARMUP = 2
NUM_RUNS = 5

QUANTIZED_CACHE_CONFIG = {
    "backend": "hqq",
    "nbits": 4,
}


def cleanup():
    """Force full GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_input(tokenizer: AutoTokenizer, context_length: int) -> dict:
    """Create a dummy input of approximately context_length tokens."""
    text = "The quick brown fox jumps over the lazy dog. " * (context_length // 8)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=context_length,
    )
    return {k: v.cuda() for k, v in inputs.items()}


def load_model():
    """Load model without accelerate's device_map to avoid memory leaks."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=select_attn_implementation()
    ).to("cuda")


def make_smash_config(kvpress: dict | None = None, hqq: dict | None = None) -> SmashConfig:
    """Build a SmashConfig using the recommended nested-dict API."""
    config = SmashConfig()
    if kvpress is not None:
        config.add({"kvpress": kvpress})
    if hqq is not None:
        config.add({"hqq": hqq})
    return config


KVPRESS_KWARGS = {"press_type": PRESS_TYPE, "compression_ratio": COMPRESSION_RATIO}
HQQ_KWARGS = {"weight_bits": 4, "compute_dtype": "torch.bfloat16"}


def measure_generation(
    model,
    inputs: dict,
    generate_tokens: int,
    num_warmup: int,
    num_runs: int,
    extra_generate_kwargs: dict | None = None,
) -> dict:
    """Measure peak memory and generation throughput."""
    gen_kwargs = dict(max_new_tokens=generate_tokens, do_sample=False)
    if extra_generate_kwargs:
        gen_kwargs.update(extra_generate_kwargs)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]
    avg_time = sum(times) / len(times)
    throughput = tokens_generated / avg_time

    return {
        "peak_memory_gb": round(peak_mem_gb, 3),
        "avg_time_s": round(avg_time, 4),
        "throughput_tok_s": round(throughput, 2),
        "tokens_generated": tokens_generated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/speed_memory.json")
    args = parser.parse_args()

    script_start = time.perf_counter()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    def save():
        """Persist current results so partial progress survives crashes."""
        output_path.write_text(json.dumps(results, indent=2))

    save()  # Clear any stale file from a previous run.

    quantized_gen_kwargs = {
        "cache_implementation": "quantized",
        "generation_config": GenerationConfig(
            cache_config=QUANTIZED_CACHE_CONFIG,
            max_new_tokens=GENERATE_TOKENS,
            do_sample=False,
        ),
    }

    for context_length in CONTEXT_LENGTHS:
        print(f"\n=== Context length: {context_length} ===")

        # Baseline (no compression) for this context length
        print(f"  -- Baseline (no compression) --")
        model = load_model()
        inputs = build_input(tokenizer, context_length)
        result = measure_generation(model, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS)
        baseline = {
            "context_length": context_length,
            "press_type": "none",
            "compression_ratio": 0.0,
            **result,
        }
        print(f"  {baseline}")
        results.append(baseline)
        save()
        del model, inputs
        cleanup()

        # Quantized cache only (no KVPress)
        print(f"  -- Quantized cache (4-bit, no press) --")
        model = load_model()
        inputs = build_input(tokenizer, context_length)
        result = measure_generation(
            model, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS,
            extra_generate_kwargs=quantized_gen_kwargs,
        )
        entry = {
            "context_length": context_length,
            "press_type": "quantized_only",
            "compression_ratio": 0.0,
            **result,
        }
        print(f"  {entry}")
        results.append(entry)
        save()
        del model, inputs
        cleanup()

        # KVPress only
        print(f"  -- {PRESS_TYPE} @ ratio={COMPRESSION_RATIO} --")
        model = load_model()
        smashed = smash(model, smash_config=make_smash_config(kvpress=KVPRESS_KWARGS))

        inputs = build_input(tokenizer, context_length)
        result = measure_generation(smashed, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS)
        entry = {
            "context_length": context_length,
            "press_type": PRESS_TYPE,
            "compression_ratio": COMPRESSION_RATIO,
            **result,
        }
        print(f"  {entry}")
        results.append(entry)
        save()
        del smashed, model, inputs
        cleanup()

        # KVPress + Quantized cache
        print(f"  -- {PRESS_TYPE} + Quantized @ ratio={COMPRESSION_RATIO} --")
        model = load_model()
        smashed = smash(model, smash_config=make_smash_config(kvpress=KVPRESS_KWARGS))

        inputs = build_input(tokenizer, context_length)
        result = measure_generation(
            smashed, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS,
            extra_generate_kwargs=quantized_gen_kwargs,
        )
        entry = {
            "context_length": context_length,
            "press_type": f"{PRESS_TYPE}+Quantized",
            "compression_ratio": COMPRESSION_RATIO,
            **result,
        }
        print(f"  {entry}")
        results.append(entry)
        save()
        del smashed, model, inputs
        cleanup()

        # Pruna HQQ weight quantization only (no press)
        print(f"  -- HQQ (4-bit weights, no press) --")
        model = load_model()
        smashed = smash(model, smash_config=make_smash_config(hqq=HQQ_KWARGS))

        inputs = build_input(tokenizer, context_length)
        result = measure_generation(smashed, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS)
        entry = {
            "context_length": context_length,
            "press_type": "hqq_only",
            "compression_ratio": 0.0,
            **result,
        }
        print(f"  {entry}")
        results.append(entry)
        save()
        del smashed, model, inputs
        cleanup()

        # Pruna composition: KVPress + HQQ
        print(f"  -- {PRESS_TYPE} + HQQ @ ratio={COMPRESSION_RATIO} --")
        model = load_model()
        smashed = smash(
            model,
            smash_config=make_smash_config(kvpress=KVPRESS_KWARGS, hqq=HQQ_KWARGS),
        )

        inputs = build_input(tokenizer, context_length)
        result = measure_generation(smashed, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS)
        entry = {
            "context_length": context_length,
            "press_type": f"{PRESS_TYPE}+HQQ",
            "compression_ratio": COMPRESSION_RATIO,
            **result,
        }
        print(f"  {entry}")
        results.append(entry)
        save()
        del smashed, model, inputs
        cleanup()

    # Compression ratio sweep at longer contexts (KVPress only).
    # Shows how memory and throughput scale with compression ratio.
    for context_length in RATIO_SWEEP_CONTEXTS:
        print(f"\n=== Ratio sweep @ context={context_length} ===")
        for ratio in RATIO_SWEEP_RATIOS:
            print(f"  -- {PRESS_TYPE} @ ratio={ratio} --")
            model = load_model()
            smashed = smash(
                model,
                smash_config=make_smash_config(
                    kvpress={"press_type": PRESS_TYPE, "compression_ratio": ratio},
                ),
            )

            inputs = build_input(tokenizer, context_length)
            result = measure_generation(smashed, inputs, GENERATE_TOKENS, NUM_WARMUP, NUM_RUNS)
            entry = {
                "context_length": context_length,
                "press_type": PRESS_TYPE,
                "compression_ratio": ratio,
                **result,
            }
            print(f"  {entry}")
            results.append(entry)
            save()
            del smashed, model, inputs
            cleanup()

    save()
    elapsed = time.perf_counter() - script_start
    print(f"\nResults saved to {output_path}")
    print(f"[timing] speed_memory total: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
