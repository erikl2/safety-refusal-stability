#!/usr/bin/env python3
"""
Run full safety refusal stability experiments.

This runs generation for all configured models, temperatures, and seeds.
Supports checkpointing - will skip already-completed configurations.

Usage:
    # Run all models
    python scripts/04_run_experiments.py

    # Run specific model
    python scripts/04_run_experiments.py --model llama-3.1-8b-instruct

    # Dry run (show what would be run)
    python scripts/04_run_experiments.py --dry-run

    # Pilot mode (50 prompts only)
    python scripts/04_run_experiments.py --pilot
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.experiment import run_full_experiment, run_pilot_experiment
from src.generation.config import get_all_model_names, load_sampling_config


def main():
    available_models = get_all_model_names()
    sampling_config = load_sampling_config()

    parser = argparse.ArgumentParser(
        description="Run safety refusal stability experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models: {', '.join(available_models)}

Examples:
    # Run all configurations for all models
    python scripts/04_run_experiments.py

    # Run only Llama model
    python scripts/04_run_experiments.py --model llama-3.1-8b-instruct

    # Preview what would run (no actual generation)
    python scripts/04_run_experiments.py --dry-run

    # Quick test with 50 prompts
    python scripts/04_run_experiments.py --pilot

    # Run with custom number of prompts
    python scripts/04_run_experiments.py --num-prompts 100
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models,
        default=None,
        help="Run only this model (default: all models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot experiment (50 prompts, 1 temp, 2 seeds)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit number of prompts (default: all)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already-completed configurations",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Safety Refusal Stability - Experiment Runner")
    print("=" * 60)
    print()

    # Show configuration
    models = [args.model] if args.model else available_models
    print(f"Models: {models}")
    print(f"Temperatures: {sampling_config.temperatures}")
    print(f"Seeds: {sampling_config.seeds}")
    print(f"Max new tokens: {sampling_config.max_new_tokens}")
    print(f"Batch size: {sampling_config.batch_size}")
    print()

    total_configs = len(models) * len(sampling_config.temperatures) * len(sampling_config.seeds)
    print(f"Total configurations: {total_configs}")
    print()

    try:
        if args.pilot:
            print("Running PILOT experiment (50 prompts, limited configs)")
            print()
            stats = run_pilot_experiment(model=args.model)
        else:
            stats = run_full_experiment(
                models=models,
                num_prompts=args.num_prompts,
                dry_run=args.dry_run,
                skip_existing=not args.no_skip,
            )

        if args.dry_run:
            print(f"\n[DRY RUN] Would run {stats.get('would_run', 0)} configurations")
            return 0

        print()
        print("=" * 60)
        print("Experiment Summary")
        print("=" * 60)
        print(f"Completed: {stats.get('completed', 0)}")
        print(f"Skipped (already done): {stats.get('skipped', 0)}")
        print(f"Failed: {stats.get('failed', 0)}")

        if stats.get("failed", 0) > 0:
            print("\nWarning: Some configurations failed. Check logs for details.")
            return 1

        print("\nExperiment complete! Results saved to data/results/generations/")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run the preprocessing scripts first:")
        print("  python scripts/01_download_data.py")
        print("  python scripts/02_preprocess.py")
        return 1

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Progress has been saved. Re-run to continue from where you left off.")
        return 130

    except Exception as e:
        print(f"\nError during experiment: {e}")
        raise


if __name__ == "__main__":
    exit(main())
