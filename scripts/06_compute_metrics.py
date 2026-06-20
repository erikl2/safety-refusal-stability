#!/usr/bin/env python3
"""
Compute stability metrics from labeled responses.

Calculates:
- Per-prompt stability index (SSI)
- Flip rates
- Aggregate statistics by model and temperature

Usage:
    python scripts/06_compute_metrics.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.metrics import compute_all_metrics


def main():
    print("=" * 60)
    print("Safety Refusal Stability - Metrics Computation")
    print("=" * 60)
    print()

    try:
        metrics = compute_all_metrics()

        print("\nMetrics saved to data/results/metrics/")
        print("  - per_prompt_metrics.csv")
        print("  - aggregate_by_model.csv")
        print("  - aggregate_by_temperature.csv")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run scripts/05_run_judge.py first.")
        return 1

    except Exception as e:
        print(f"\nError computing metrics: {e}")
        raise


if __name__ == "__main__":
    exit(main())
