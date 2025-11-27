# Safety Refusal Stability in LLMs

**Research investigating how stable LLM safety refusal decisions are across different random seeds and temperature settings.**

## Key Finding

**32% of harmful prompts produce inconsistent safety decisions** when varying only random seed and temperature settings, demonstrating that single-shot safety evaluations are insufficient for reliable safety assessment.

## Overview

Current safety evaluations of large language models rely on single-shot testing, implicitly assuming model responses are deterministic. We challenge this assumption by systematically testing the same harmful prompts across multiple sampling configurations.

### Experimental Setup

- **Model:** Llama 3.1 8B Instruct
- **Dataset:** 876 harmful prompts from BeaverTails
- **Configurations:** 4 temperatures (0.0, 0.3, 0.7, 1.0) × 5 random seeds (42-46)
- **Total Responses:** 17,520 (876 prompts × 20 configs)
- **Judge Model:** Llama 3.1 8B Instruct (3-class: REFUSE/PARTIAL/COMPLY)

### Main Results

| Metric | Value |
|--------|-------|
| Mean SSI | 0.799 |
| Unstable Prompts (SSI < 0.8) | 32.0% |
| Overall Refusal Rate | 83.1% |
| Overall Partial Rate | 10.9% |
| Overall Comply Rate | 5.9% |

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
python scripts/02_judge_responses.py

# 3. Compute metrics
python scripts/06_compute_metrics.py

# 4. Generate figures
python scripts/07_generate_figures.py
```

## Safety Stability Index (SSI)

SSI(p) = max_c(n_c) / N

- SSI = 1.0: Perfectly stable
- SSI < 0.8: Unstable
- SSI ≥ 1/3: Theoretical minimum

## Paper

**Title:** "The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior"

Full paper in `paper/paper_v5.pdf`

## License

MIT
