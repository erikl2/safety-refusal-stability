#!/usr/bin/env python3
"""
Preprocess raw datasets into unified prompt file.

Usage:
    python scripts/02_preprocess.py [--threshold 0.9] [--sample-size 50]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import preprocess_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess harmful prompt datasets"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for fuzzy deduplication (default: 0.9)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of prompts for sample file (default: 50)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Safety Refusal Stability - Data Preprocessing")
    print("=" * 60)
    print()

    try:
        df_full, df_sample = preprocess_prompts(
            similarity_threshold=args.threshold,
            sample_size=args.sample_size,
        )
        print("\nPreprocessing complete!")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run scripts/01_download_data.py first.")
        return 1

    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        raise


if __name__ == "__main__":
    exit(main())
