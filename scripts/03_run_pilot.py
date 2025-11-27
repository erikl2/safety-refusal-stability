#!/usr/bin/env python3
"""
Run a small pilot experiment to test the pipeline.

This runs:
- 50 prompts
- 1 model (default: first in config)
- 1 temperature (0.7)
- 2 seeds (42, 43)

Usage:
    python scripts/03_run_pilot.py [--model MODEL_NAME]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.experiment import run_pilot_experiment
from src.generation.config import get_all_model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run pilot experiment for safety refusal stability"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to test. Available: {get_all_model_names()}",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Safety Refusal Stability - Pilot Experiment")
    print("=" * 60)
    print()
    print("This is a small-scale test to verify the pipeline works.")
    print("It will run:")
    print("  - 50 prompts")
    print("  - 1 temperature (0.7)")
    print("  - 2 seeds (42, 43)")
    print()

    try:
        stats = run_pilot_experiment(model=args.model)

        print()
        print("=" * 60)
        print("Pilot experiment complete!")
        print("=" * 60)
        print(f"Completed: {stats.get('completed', 0)}")
        print(f"Skipped: {stats.get('skipped', 0)}")
        print(f"Failed: {stats.get('failed', 0)}")

        if stats.get("failed", 0) > 0:
            print("\nWarning: Some runs failed. Check logs for details.")
            return 1

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run the preprocessing scripts first:")
        print("  python scripts/01_download_data.py")
        print("  python scripts/02_preprocess.py")
        return 1

    except Exception as e:
        print(f"\nError during pilot experiment: {e}")
        raise


if __name__ == "__main__":
    exit(main())
