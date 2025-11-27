#!/usr/bin/env python3
"""
Generate paper-ready figures from computed metrics.

Creates:
- ssi_distribution.png: Histogram of stability index
- temperature_effect.png: Line plot of temperature vs compliance
- model_comparison.png: Bar chart comparing models
- stability_vs_compliance.png: Scatter plot
- flip_rate.png: Bar chart of consistency

Usage:
    python scripts/07_generate_figures.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.figures import create_all_figures


def main():
    print("=" * 60)
    print("Safety Refusal Stability - Figure Generation")
    print("=" * 60)
    print()

    try:
        figures = create_all_figures()

        print()
        print("=" * 60)
        print("Figures Generated")
        print("=" * 60)
        for name in figures:
            print(f"  - {name}.png")
            print(f"  - {name}.pdf")

        print("\nFigures saved to paper/figures/")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run scripts/06_compute_metrics.py first.")
        return 1

    except Exception as e:
        print(f"\nError generating figures: {e}")
        raise


if __name__ == "__main__":
    exit(main())
