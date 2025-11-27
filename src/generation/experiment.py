"""
Experiment runner for safety refusal stability experiments.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .config import (
    load_sampling_config,
    load_model_config,
    get_all_model_names,
    load_pilot_config,
    SamplingConfig,
    ModelConfig,
)
from .vllm_runner import VLLMRunner, results_to_dataframe

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GENERATIONS_DIR = PROJECT_ROOT / "data" / "results" / "generations"
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_output_filename(model_name: str, temperature: float, seed: int) -> str:
    """Generate output filename for a specific configuration."""
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    return f"{safe_model_name}_temp{temperature}_seed{seed}.csv"


def check_existing_runs(
    models: list[str],
    temperatures: list[float],
    seeds: list[int],
    output_dir: Path,
) -> list[tuple[str, float, int]]:
    """
    Check which configurations have already been run.

    Returns:
        List of (model, temperature, seed) tuples that still need to be run
    """
    remaining = []

    for model in models:
        model_config = load_model_config(model)
        for temp in temperatures:
            for seed in seeds:
                filename = get_output_filename(model_config.hf_name, temp, seed)
                output_path = output_dir / filename

                if output_path.exists():
                    # Verify the file is complete (has data)
                    try:
                        df = pd.read_csv(output_path)
                        if len(df) > 0:
                            continue  # Already done
                    except Exception:
                        pass  # File is corrupt, re-run

                remaining.append((model, temp, seed))

    return remaining


def load_prompts(
    prompts_file: Optional[Path] = None,
    sample: bool = False,
    num_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load prompts for generation.

    Args:
        prompts_file: Path to prompts CSV file
        sample: If True, use the sample file
        num_prompts: If specified, limit to this many prompts

    Returns:
        DataFrame with prompts
    """
    if prompts_file is None:
        if sample:
            prompts_file = PROCESSED_DATA_DIR / "prompts_sample.csv"
        else:
            prompts_file = PROCESSED_DATA_DIR / "prompts.csv"

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_file}. "
            "Run scripts/02_preprocess.py first."
        )

    df = pd.read_csv(prompts_file)

    if num_prompts is not None and num_prompts < len(df):
        df = df.head(num_prompts)

    return df


def run_experiment(
    model_name: str,
    temperatures: list[float],
    seeds: list[int],
    prompts_df: pd.DataFrame,
    output_dir: Path,
    sampling_config: SamplingConfig,
    logger: logging.Logger,
    skip_existing: bool = True,
) -> dict:
    """
    Run generation experiment for a single model.

    Args:
        model_name: Model key name from config
        temperatures: List of temperatures to test
        seeds: List of seeds to test
        prompts_df: DataFrame with prompts
        output_dir: Directory to save results
        sampling_config: Sampling configuration
        logger: Logger instance
        skip_existing: Whether to skip already-completed runs

    Returns:
        Dict with statistics about the run
    """
    model_config = load_model_config(model_name)
    logger.info(f"Running experiments for model: {model_config.hf_name}")

    # Initialize runner (loads model once)
    runner = VLLMRunner(
        model_name=model_config.hf_name,
        model_config=model_config,
    )

    # Prepare prompts list
    prompts_list = [
        {"id": row["id"], "prompt": row["prompt"]}
        for _, row in prompts_df.iterrows()
    ]

    stats = {
        "model": model_name,
        "total_configs": len(temperatures) * len(seeds),
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    # Run for each temperature and seed
    for temp in temperatures:
        for seed in seeds:
            filename = get_output_filename(model_config.hf_name, temp, seed)
            output_path = output_dir / filename

            # Check if already done
            if skip_existing and output_path.exists():
                try:
                    df = pd.read_csv(output_path)
                    if len(df) >= len(prompts_list):
                        logger.info(f"Skipping {filename} (already complete)")
                        stats["skipped"] += 1
                        continue
                except Exception:
                    pass

            logger.info(f"Running: temperature={temp}, seed={seed}")

            try:
                results = runner.generate(
                    prompts=prompts_list,
                    temperature=temp,
                    seed=seed,
                    max_new_tokens=sampling_config.max_new_tokens,
                    top_p=sampling_config.top_p,
                    batch_size=sampling_config.batch_size,
                    show_progress=True,
                )

                # Save results
                df = results_to_dataframe(results)
                output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)

                logger.info(f"Saved {len(df)} results to {output_path}")
                stats["completed"] += 1

            except Exception as e:
                logger.error(f"Failed: {e}")
                stats["failed"] += 1

    return stats


def run_full_experiment(
    models: Optional[list[str]] = None,
    temperatures: Optional[list[float]] = None,
    seeds: Optional[list[int]] = None,
    num_prompts: Optional[int] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """
    Run the full experiment suite.

    Args:
        models: List of model names (uses all models from config if not specified)
        temperatures: List of temperatures (uses config if not specified)
        seeds: List of seeds (uses config if not specified)
        num_prompts: Limit number of prompts (uses all if not specified)
        output_dir: Output directory for generations
        dry_run: If True, print what would be run without running
        skip_existing: Whether to skip already-completed runs

    Returns:
        Dict with overall statistics
    """
    # Load configurations
    sampling_config = load_sampling_config()

    if models is None:
        models = get_all_model_names()
    if temperatures is None:
        temperatures = sampling_config.temperatures
    if seeds is None:
        seeds = sampling_config.seeds
    if output_dir is None:
        output_dir = GENERATIONS_DIR

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"experiment_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Safety Refusal Stability - Experiment Runner")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Temperatures: {temperatures}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total configurations: {len(models) * len(temperatures) * len(seeds)}")

    # Check what needs to be run
    remaining = check_existing_runs(models, temperatures, seeds, output_dir)
    logger.info(f"Remaining configurations: {len(remaining)}")

    if dry_run:
        logger.info("\n[DRY RUN] Would run the following:")
        for model, temp, seed in remaining:
            logger.info(f"  - {model}, temp={temp}, seed={seed}")
        return {"dry_run": True, "would_run": len(remaining)}

    # Load prompts
    prompts_df = load_prompts(num_prompts=num_prompts)
    logger.info(f"Loaded {len(prompts_df)} prompts")

    # Run experiments
    all_stats = []

    for model in models:
        # Filter to just this model's remaining configs
        model_temps = sorted(set(t for m, t, s in remaining if m == model))
        model_seeds = sorted(set(s for m, t, s in remaining if m == model))

        if not model_temps or not model_seeds:
            logger.info(f"Skipping {model} (all configurations complete)")
            continue

        stats = run_experiment(
            model_name=model,
            temperatures=model_temps,
            seeds=model_seeds,
            prompts_df=prompts_df,
            output_dir=output_dir,
            sampling_config=sampling_config,
            logger=logger,
            skip_existing=skip_existing,
        )
        all_stats.append(stats)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Complete")
    logger.info("=" * 60)

    total_completed = sum(s["completed"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)

    logger.info(f"Completed: {total_completed}")
    logger.info(f"Skipped: {total_skipped}")
    logger.info(f"Failed: {total_failed}")

    return {
        "completed": total_completed,
        "skipped": total_skipped,
        "failed": total_failed,
        "stats_by_model": all_stats,
    }


def run_pilot_experiment(
    model: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run a small pilot experiment for testing.

    Args:
        model: Model to test (uses first model from config if not specified)
        output_dir: Output directory

    Returns:
        Dict with statistics
    """
    pilot_config = load_pilot_config()

    if model is None:
        model = get_all_model_names()[0]

    return run_full_experiment(
        models=[model],
        temperatures=pilot_config["temperatures"],
        seeds=pilot_config["seeds"],
        num_prompts=pilot_config["num_prompts"],
        output_dir=output_dir,
        skip_existing=True,
    )
