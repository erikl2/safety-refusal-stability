# Safety Refusal Stability in LLMs

**Research investigating how stable LLM safety refusal decisions are across different random seeds and temperature settings.**

**[Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/elarsen21/safety-stability-explorer)**

## Key Finding

**18-28% of harmful prompts produce inconsistent safety decisions** when varying only random seed and temperature settings, demonstrating that single-shot safety evaluations are insufficient for reliable safety assessment.

## Overview

Current safety evaluations of large language models rely on single-shot testing, implicitly assuming model responses are deterministic. We challenge this assumption by systematically testing the same harmful prompts across multiple sampling configurations.

### Experimental Setup

- **Models:** Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, Qwen 3 8B, Gemma 3 12B Instruct
- **Dataset:** 876 harmful prompts (491 from AdvBench + 385 from HarmBench)
- **Configurations:** 4 temperatures (0.0, 0.3, 0.7, 1.0) × 5 random seeds (42-46)
- **Total Responses:** 70,080 (876 prompts × 20 configs × 4 models)
- **Judge Model:** Claude 3.5 Haiku (primary judge for the reported metrics), cross-validated against a local Llama-3.1-70B judge

### Dataset Details

We use 876 harmful prompts drawn from two established red-teaming benchmarks
(see `src/data/load_datasets.py` and `data/processed/prompts.csv`):

- **Sources:** AdvBench (491 prompts) and HarmBench (385 prompts)
- **Loaders:** `load_advbench` / `load_harmbench` in `src/data/load_datasets.py`;
  the `source` column in `data/processed/prompts.csv` records each prompt's origin
- **Selection:** harmful behaviors from each benchmark, concatenated and deduplicated

> Note: per-prompt harm-category labels are not populated in the processed
> dataset (`harm_category` is empty), so a category breakdown is not regenerable
> from the committed data and is therefore not reported here.

### Inference Configuration

| Parameter | Value |
|-----------|-------|
| Inference Engine | vLLM 0.6.3 |
| Temperatures | 0.0, 0.3, 0.7, 1.0 |
| Random Seeds | 42, 43, 44, 45, 46 |
| Top-p | 1.0 (disabled) |
| Top-k | -1 (disabled) |
| Max Output Tokens | 512 |
| GPU | NVIDIA A100-80GB |

### Model Comparison (Claude 3.5 Haiku Judge)

| Model | Mean SSI | Flip Rate | % Unstable | Refusal Rate |
|-------|----------|-----------|------------|--------------|
| Gemma 3 12B | **0.965** | 18.4% | **6.7%** | 78.5% |
| Llama 3.1 8B | 0.944 | 27.3% | 10.4% | 79.3% |
| Qwen 3 8B | 0.938 | 27.7% | 11.8% | 92.5% |
| Qwen 2.5 7B | 0.938 | 26.3% | 12.0% | 81.3% |

### Temperature Effects

| Temperature | Mean SSI | Flip Rate |
|-------------|----------|-----------|
| 0.0 | 0.951 | 9.5% |
| 0.3 | 0.927 | 15.2% |
| 0.7 | 0.912 | 18.1% |
| 1.0 | 0.896 | 19.6% |

### Inter-Judge Validation

- **89.0%** exact agreement between Claude 3.5 Haiku and Llama 70B judges
- **Cohen's κ = 0.62** indicating substantial inter-rater reliability

## Repository Structure

```
safety-refusal-stability/
├── configs/              # Experiment configurations
├── data/
│   ├── prompts/         # AdvBench + HarmBench harmful prompts
│   └── results/
│       ├── generations/ # Raw model responses
│       ├── labels/      # Safety classifications
│       └── metrics/     # Computed statistics
├── paper/
│   ├── figures/         # Publication-ready visualizations (PNG + PDF)
│   └── main.tex         # LaTeX paper source
├── scripts/             # Pipeline scripts
└── src/                 # Source code modules
```

## Quick Start

```bash
# 1. Generate responses
python scripts/01_generate_responses.py

# 2. Judge responses
python scripts/05_run_judge.py

# 3. Compute metrics
python scripts/06_compute_metrics.py

# 4. Generate figures
python scripts/07_generate_figures.py
```

> **Reproducing the reported numbers:** step 2 above (`05_run_judge.py`) runs the
> local Llama-3.1-70B judge used for cross-validation. The headline metrics in
> this README and the paper are produced by the **Claude 3.5 Haiku** judge —
> run `scripts/13_batch_judge_anthropic.py` (requires `ANTHROPIC_API_KEY`) to
> regenerate those labels, then `06_compute_metrics.py`.

## Safety Stability Index (SSI)

SSI(p) = max_c(n_c) / N

- SSI = 1.0: Perfectly stable (all 20 responses agree)
- SSI < 0.8: Unstable (fewer than 16/20 agree)
- SSI ≥ 1/3: Theoretical minimum

## Paper

**Title:** "The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior"

See `paper/main.pdf` for the full paper.

## License

MIT
