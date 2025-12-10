# Safety Refusal Stability in LLMs

**Research investigating how stable LLM safety refusal decisions are across different random seeds and temperature settings.**

**[Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/elarsen21/safety-stability-explorer)**

## Key Finding

**32% of harmful prompts produce inconsistent safety decisions** when varying only random seed and temperature settings, demonstrating that single-shot safety evaluations are insufficient for reliable safety assessment.

## Overview

Current safety evaluations of large language models rely on single-shot testing, implicitly assuming model responses are deterministic. We challenge this assumption by systematically testing the same harmful prompts across multiple sampling configurations.

### Experimental Setup

- **Models:** Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct
- **Dataset:** 876 harmful prompts from BeaverTails
- **Configurations:** 4 temperatures (0.0, 0.3, 0.7, 1.0) × 5 random seeds (42-46)
- **Total Responses:** 35,040 (876 prompts × 20 configs × 2 models)
- **Judge Model:** Llama 3.1 70B Instruct (3-class: REFUSE/PARTIAL/COMPLY)

### Main Results (Llama 3.1 8B)

| Metric | Value |
|--------|-------|
| Mean SSI | 0.924 |
| Decision Flips | 32.0% |
| Unstable Prompts (SSI < 0.8) | 14.3% |
| Refusal Rate | 83.1% |
| Partial Rate | 10.9% |
| Comply Rate | 5.9% |

### Temperature Effects

| Temperature | Mean SSI | % Unstable |
|-------------|----------|------------|
| 0.0 | 0.987 | 0.9% |
| 0.3 | 0.951 | 8.0% |
| 0.7 | 0.927 | 12.3% |
| 1.0 | 0.904 | 16.2% |

## Repository Structure

```
safety-refusal-stability/
├── configs/              # Experiment configurations
├── data/
│   ├── prompts/         # BeaverTails harmful prompts
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
