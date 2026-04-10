# pruna-kvpress-eval

Evaluation scripts and benchmarks for the [KVPress](https://github.com/NVIDIA/kvpress) integration into [Pruna](https://github.com/PrunaAI/pruna). The integration lives on the [`feat/kvpress`](https://github.com/kschwethelm/pruna/tree/feat/kvpress) branch of [kschwethelm/pruna](https://github.com/kschwethelm/pruna).

## Results

See [eval_results.pdf](eval_results.pdf) for the full evaluation report.

Key findings on Llama-3.1-8B-Instruct (H100 GPU):

- **Memory**: At 128k context, removing 50% of KV pairs saves 8.6 GB (48.9 → 40.3 GB). Composing KV cache pruning with HQQ weight quantization via Pruna's `SmashConfig` yields 38% total savings (48.9 → 30.3 GB).
- **Quality**: SnapKVPress nearly matches baseline on LongBench tasks at 50% compression (avg F1: 0.303 vs 0.304). ExpectedAttentionPress follows closely. With default settings, quantization (cache and weight) degrades quality more than pruning.
- **Throughput**: KV cache pruning has negligible throughput impact (<5%) and improves throughput at long contexts (+12–16% at 32k–128k). Quantization (HQQ backend) adds 30–75% overhead, though alternative backends may perform differently.

## Setup

```bash
uv sync
```

## Experiments

All experiments are submitted via SLURM:

```bash
./shells/_submit.sh shells/speed_memory.sh       # Speed and memory profiling
./shells/_submit.sh shells/downstream_eval.sh     # LongBench downstream tasks
```

### Speed and Memory (`scripts/speed_memory.py`)

Profiles peak GPU memory and decode throughput across context lengths (8k–128k) for six configurations: baseline, KnormPress, cache quant (4-bit), KnormPress + cache quant, weight quant (HQQ 4-bit), and KnormPress + weight quant. Includes a compression ratio sweep (0.2–0.8) at 32k, 64k, and 128k context.

### Downstream Evaluation (`scripts/downstream_eval.py`)

Evaluates quality on three LongBench tasks (HotpotQA, NarrativeQA, Qasper) via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). Two sub-experiments: (1) all five press types plus quantization configs at ratio 0.5, and (2) ratio sweep (0.2–0.8) for SnapKVPress, ExpectedAttentionPress, and KnormPress.

### Plotting (`scripts/plot_results.py`)

Generates four figures from the result JSON files:
- `speed_memory_context.pdf` — memory and throughput vs context length
- `speed_memory_ratio.pdf` — memory and throughput vs compression ratio at 128k
- `downstream_press_comparison.pdf` — bar chart comparing all configs at ratio 0.5
- `downstream_ratio_sweep.pdf` — quality vs compression ratio for three press types

## Integration Design

KVPress is exposed as a single Pruna algorithm class with a `press_type` hyperparameter:

```python
from pruna import smash, SmashConfig

config = SmashConfig()
config.add({"kvpress": {"press_type": "SnapKVPress", "compression_ratio": 0.4}})
model = smash(model, smash_config=config)
```

Composable with other Pruna algorithms (e.g., HQQ weight quantization):

```python
config = SmashConfig()
config.add({"kvpress": {"press_type": "KnormPress", "compression_ratio": 0.5}})
config.add({"hqq": {"weight_bits": 4, "compute_dtype": "torch.bfloat16"}})
model = smash(model, smash_config=config)
```

## Related

- Integration PR branch: [kschwethelm/pruna@feat/kvpress](https://github.com/kschwethelm/pruna/tree/feat/kvpress)
- Upstream issue: [PrunaAI/pruna#366](https://github.com/PrunaAI/pruna/issues/366)
- KVPress: [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
- Pruna: [PrunaAI/pruna](https://github.com/PrunaAI/pruna)
