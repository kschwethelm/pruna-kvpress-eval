"""Generate plots from experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

PRESS_COLORS = {
    "SnapKVPress": "#1f77b4",
    "ExpectedAttentionPress": "#ff7f0e",
    "StreamingLLMPress": "#2ca02c",
    "KnormPress": "#d62728",
    "RandomPress": "#7f7f7f",
    "quantized_only": "#9467bd",
    "KnormPress+Quantized": "#e377c2",
    "hqq_only": "#8c564b",
    "KnormPress+HQQ": "#17becf",
}

# Display names for plot legends (internal key -> readable label)
PRESS_LABELS = {
    "none": "Baseline",
    "quantized_only": "Cache Quant (4-bit)",
    "hqq_only": "Weight Quant (HQQ 4-bit)",
    "KnormPress+Quantized": "KnormPress + Cache Quant",
    "KnormPress+HQQ": "KnormPress + Weight Quant",
}

CONTEXT_SWEEP_RATIO = 0.5
RATIO_SWEEP_CONTEXT = 131072


def plot_speed_memory_context_sweep(data: list[dict], output_dir: Path):
    """Line plots: memory and throughput vs context length at ratio 0.5."""
    # Filter to entries that are part of the context sweep:
    # baseline (ratio=0.0) and compressed configs at the context-sweep ratio.
    sweep = [
        r for r in data
        if r["press_type"] == "none"
        or r["compression_ratio"] == CONTEXT_SWEEP_RATIO
        or r["press_type"] in ("quantized_only", "hqq_only")
    ]
    context_lengths = sorted(set(r["context_length"] for r in sweep))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = sorted(set(r["press_type"] for r in sweep))
    # Plot baseline first with dashed line
    for cfg in configs:
        pts = sorted(
            [r for r in sweep if r["press_type"] == cfg],
            key=lambda r: r["context_length"],
        )
        ctx = [r["context_length"] for r in pts]
        mem = [r["peak_memory_gb"] for r in pts]
        thr = [r["throughput_tok_s"] for r in pts]
        label = PRESS_LABELS.get(cfg, cfg)
        if cfg == "none":
            ax1.plot(ctx, mem, "k--", label=label, linewidth=2)
            ax2.plot(ctx, thr, "k--", label=label, linewidth=2)
        else:
            color = PRESS_COLORS.get(cfg, "#333333")
            ax1.plot(ctx, mem, marker="o", label=label, color=color)
            ax2.plot(ctx, thr, marker="o", label=label, color=color)

    ctx_labels = [f"{c // 1024}k" for c in context_lengths]
    for ax in (ax1, ax2):
        ax.set_xticks(context_lengths)
        ax.set_xticklabels(ctx_labels)
        ax.set_xlabel("Context Length (tokens)")

    ax1.set_ylabel("Peak Memory (GB)")
    ax1.set_title(f"Peak GPU Memory (compression_ratio={CONTEXT_SWEEP_RATIO})")
    ax1.legend(fontsize=8)

    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title(f"Decode Throughput (compression_ratio={CONTEXT_SWEEP_RATIO})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "speed_memory_context.pdf", bbox_inches="tight")
    print(f"Saved {output_dir / 'speed_memory_context.pdf'}")
    plt.close(fig)


def plot_speed_memory_ratio_sweep(data: list[dict], output_dir: Path):
    """Line plots: memory and throughput vs compression ratio at 128k context."""
    # Baseline at 128k
    baseline = next(
        (r for r in data
         if r["press_type"] == "none" and r["context_length"] == RATIO_SWEEP_CONTEXT),
        None,
    )
    # All KnormPress entries at the sweep context
    knorm = sorted(
        [r for r in data
         if r["press_type"] == "KnormPress" and r["context_length"] == RATIO_SWEEP_CONTEXT],
        key=lambda r: r["compression_ratio"],
    )
    if not knorm:
        print("Skipping ratio sweep plot: no KnormPress entries at 128k")
        return

    ratios = [r["compression_ratio"] for r in knorm]
    mem = [r["peak_memory_gb"] for r in knorm]
    thr = [r["throughput_tok_s"] for r in knorm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    color = PRESS_COLORS["KnormPress"]
    ax1.plot(ratios, mem, marker="o", color=color, label="KnormPress")
    ax2.plot(ratios, thr, marker="o", color=color, label="KnormPress")

    if baseline:
        ax1.axhline(baseline["peak_memory_gb"], color="black", linestyle="--", label="Baseline")
        ax2.axhline(baseline["throughput_tok_s"], color="black", linestyle="--", label="Baseline")

    for ax in (ax1, ax2):
        ax.set_xlabel("Compression Ratio")
        ax.legend(fontsize=8)

    ax1.set_ylabel("Peak Memory (GB)")
    ax1.set_title(f"Peak GPU Memory (context={RATIO_SWEEP_CONTEXT // 1024}k)")
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title(f"Decode Throughput (context={RATIO_SWEEP_CONTEXT // 1024}k)")

    fig.tight_layout()
    fig.savefig(output_dir / "speed_memory_ratio.pdf", bbox_inches="tight")
    print(f"Saved {output_dir / 'speed_memory_ratio.pdf'}")
    plt.close(fig)


def _get_metric(task_data: dict) -> tuple[float, str]:
    """Return the primary metric value and name for a task."""
    for key in ["qa_f1_score", "exact_match", "acc_norm", "acc"]:
        if key in task_data:
            return task_data[key], key
    for key, val in task_data.items():
        if isinstance(val, (int, float)):
            return val, key
    return 0.0, "unknown"


def plot_downstream_press_comparison(data: list[dict], output_dir: Path):
    """Grouped bar chart: baseline + all press types and quantization configs."""
    baseline = next(r for r in data if r["press_type"] == "none")
    # Press types at ratio 0.5, plus standalone quantization configs (ratio 0.0)
    QUANT_ONLY_TYPES = {"hqq_only", "quantized_only"}
    compressed = [
        r for r in data
        if r["press_type"] != "none"
        and (r["compression_ratio"] == CONTEXT_SWEEP_RATIO
             or r["press_type"] in QUANT_ONLY_TYPES)
    ]
    if not compressed:
        print("Skipping downstream press comparison: no entries at ratio 0.5")
        return

    tasks = sorted(baseline["tasks"].keys())
    fig, ax = plt.subplots(figsize=(12, 5))

    n_groups = len(tasks)
    n_bars = 1 + len(compressed)
    bar_width = 0.8 / n_bars
    x = list(range(n_groups))

    bl_vals = [_get_metric(baseline["tasks"][t])[0] for t in tasks]
    offsets = [xi - (n_bars - 1) * bar_width / 2 for xi in x]
    ax.bar(offsets, bl_vals, bar_width, label="Baseline", color="black", alpha=0.7)

    for j, entry in enumerate(compressed):
        vals = [_get_metric(entry["tasks"][t])[0] for t in tasks]
        color = PRESS_COLORS.get(entry["press_type"], "#333333")
        bar_offsets = [xi - (n_bars - 1) * bar_width / 2 + (j + 1) * bar_width for xi in x]
        label = PRESS_LABELS.get(entry["press_type"], entry["press_type"])
        ax.bar(bar_offsets, vals, bar_width, label=label, color=color, alpha=0.85)

    metric_labels = [_get_metric(baseline["tasks"][t])[1] for t in tasks]
    tick_labels = [f"{t}\n({m})" for t, m in zip(tasks, metric_labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Downstream Task Performance (compression_ratio={CONTEXT_SWEEP_RATIO})")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout()
    fig.savefig(output_dir / "downstream_press_comparison.pdf", bbox_inches="tight")
    print(f"Saved {output_dir / 'downstream_press_comparison.pdf'}")
    plt.close(fig)


def plot_downstream_ratio_sweep(data: list[dict], output_dir: Path):
    """Line plot: avg F1 vs compression ratio for KnormPress and SnapKVPress."""
    baseline = next(r for r in data if r["press_type"] == "none")
    tasks = sorted(baseline["tasks"].keys())
    bl_avg = sum(_get_metric(baseline["tasks"][t])[0] for t in tasks) / len(tasks)

    fig, ax = plt.subplots(figsize=(8, 5))

    for press_type in ["SnapKVPress", "ExpectedAttentionPress", "KnormPress"]:
        pts = sorted(
            [r for r in data if r["press_type"] == press_type],
            key=lambda r: r["compression_ratio"],
        )
        if len(pts) < 2:
            continue
        ratios = [r["compression_ratio"] for r in pts]
        avgs = [
            sum(_get_metric(r["tasks"][t])[0] for t in tasks) / len(tasks)
            for r in pts
        ]
        color = PRESS_COLORS.get(press_type, "#333333")
        ax.plot(ratios, avgs, marker="o", label=press_type, color=color)

    ax.axhline(bl_avg, color="black", linestyle="--", linewidth=1.5, label="Baseline")
    ax.set_xlabel("Compression Ratio (fraction of KV pairs removed)")
    ax.set_ylabel("Average F1 Score")
    ax.set_title("Task Quality vs Compression Ratio")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "downstream_ratio_sweep.pdf", bbox_inches="tight")
    print(f"Saved {output_dir / 'downstream_ratio_sweep.pdf'}")
    plt.close(fig)


def _load_and_merge(*paths: Path) -> list[dict]:
    """Load and merge JSON result files."""
    merged = []
    for p in paths:
        if p.exists():
            merged.extend(json.loads(p.read_text()))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sm_data = _load_and_merge(Path("results/speed_memory.json"))
    ds_data = _load_and_merge(Path("results/downstream_eval.json"))

    if sm_data:
        print(f"Speed/memory: {len(sm_data)} entries")
        plot_speed_memory_context_sweep(sm_data, output_dir)
        plot_speed_memory_ratio_sweep(sm_data, output_dir)
    else:
        print("Skipping speed/memory plots: no data found")

    if ds_data:
        print(f"Downstream: {len(ds_data)} entries")
        plot_downstream_press_comparison(ds_data, output_dir)
        plot_downstream_ratio_sweep(ds_data, output_dir)
    else:
        print("Skipping downstream eval plots: no data found")


if __name__ == "__main__":
    main()
