#!/usr/bin/env python3
"""
Run judge to classify all generated responses.

Classifies each response as:
- REFUSE: Clear refusal to help
- PARTIAL: Partial refusal or hedged response
- COMPLY: Substantive compliance with harmful request

Usage:
    # Judge all generation files
    python scripts/05_run_judge.py

    # Judge a specific file
    python scripts/05_run_judge.py --file data/results/generations/llama_temp0.7_seed42.csv

    # Use a different judge model
    python scripts/05_run_judge.py --judge-model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.judge import judge_all_generations, Judge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run judge to classify model responses"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Specific generation file to judge (default: all files)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model to use for judging (default: from config)",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["default", "detailed", "simple"],
        default="default",
        help="Judge prompt template (default: default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for judge inference (default: 16)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-judge files even if labels exist",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Safety Refusal Stability - Response Judge")
    print("=" * 60)
    print()

    try:
        if args.file:
            # Judge specific file
            if not args.file.exists():
                print(f"Error: File not found: {args.file}")
                return 1

            print(f"Judging single file: {args.file}")

            judge = Judge(model_name=args.judge_model)
            df = judge.judge_file(
                input_path=args.file,
                template=args.template,
                batch_size=args.batch_size,
            )

            print(f"\nJudged {len(df)} responses")
            print(f"Label distribution:")
            print(df["label"].value_counts().to_string())

        else:
            # Judge all files
            print("Judging all generation files...")
            print()

            stats = judge_all_generations(
                judge_model=args.judge_model,
                template=args.template,
                batch_size=args.batch_size,
                skip_existing=not args.no_skip,
            )

            print()
            print("=" * 60)
            print("Judge Summary")
            print("=" * 60)
            print(f"Processed: {stats['processed']}")
            print(f"Skipped (already done): {stats['skipped']}")
            print(f"Failed: {stats['failed']}")

            if stats["failed"] > 0:
                print("\nWarning: Some files failed. Check logs for details.")
                return 1

        print("\nLabels saved to data/results/labels/")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure generation files exist in data/results/generations/")
        return 1

    except Exception as e:
        print(f"\nError during judging: {e}")
        raise


if __name__ == "__main__":
    exit(main())
