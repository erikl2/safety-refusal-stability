"""
Figure generation for safety refusal stability analysis.

Creates paper-ready visualizations including:
- SSI distribution histograms
- Model × category heatmaps
- Temperature effect plots
- Flip rate bar charts
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Style settings
FIGURE_DPI = 300
FONT_SIZE = 12
COLORBLIND_PALETTE = sns.color_palette("colorblind")


def setup_style():
    """Set up matplotlib style for paper-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
    })


def load_metrics(metrics_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """Load all metric files."""
    metrics_dir = metrics_dir or METRICS_DIR

    metrics = {}
    for name in ["per_prompt_metrics", "aggregate_by_model", "aggregate_by_temperature"]:
        path = metrics_dir / f"{name}.csv"
        if path.exists():
            metrics[name] = pd.read_csv(path)

    if not metrics:
        raise FileNotFoundError(
            f"No metric files found in {metrics_dir}. "
            "Run scripts/06_compute_metrics.py first."
        )

    return metrics


def create_ssi_histogram(
    prompt_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
    threshold: float = 0.8,
) -> plt.Figure:
    """
    Create histogram of SSI distribution.

    Args:
        prompt_metrics: Per-prompt metrics DataFrame
        output_path: Path to save figure
        threshold: Threshold for "unstable" classification

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram
    ssi_values = prompt_metrics["stability_index"]
    ax.hist(
        ssi_values,
        bins=30,
        color=COLORBLIND_PALETTE[0],
        edgecolor="white",
        alpha=0.8,
    )

    # Add threshold line
    ax.axvline(
        x=threshold,
        color=COLORBLIND_PALETTE[3],
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )

    # Annotate percentage below threshold
    pct_unstable = (ssi_values < threshold).mean() * 100
    ax.annotate(
        f"{pct_unstable:.1f}% unstable\n(SSI < {threshold})",
        xy=(threshold - 0.05, ax.get_ylim()[1] * 0.8),
        ha="right",
        fontsize=FONT_SIZE,
        color=COLORBLIND_PALETTE[3],
    )

    ax.set_xlabel("Stability Index (SSI)")
    ax.set_ylabel("Number of Prompts")
    ax.set_title("Distribution of Safety Refusal Stability")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig


def create_temperature_effect_plot(
    temp_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create line plot showing temperature effect on compliance rate.

    Args:
        temp_metrics: Aggregate by temperature DataFrame
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    temps = temp_metrics["temperature"]
    comply_rates = temp_metrics["mean_comply_rate"] * 100
    ssi = temp_metrics["mean_ssi"]

    # Plot compliance rate
    ax.plot(
        temps,
        comply_rates,
        marker="o",
        markersize=10,
        linewidth=2,
        color=COLORBLIND_PALETTE[0],
        label="Comply Rate (%)",
    )

    # Create second y-axis for SSI
    ax2 = ax.twinx()
    ax2.plot(
        temps,
        ssi,
        marker="s",
        markersize=10,
        linewidth=2,
        color=COLORBLIND_PALETTE[2],
        label="Mean SSI",
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Comply Rate (%)", color=COLORBLIND_PALETTE[0])
    ax2.set_ylabel("Mean SSI", color=COLORBLIND_PALETTE[2])

    ax.set_title("Effect of Temperature on Safety Behavior")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig


def create_model_comparison_bars(
    model_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create bar chart comparing models.

    Args:
        model_metrics: Aggregate by model DataFrame
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = model_metrics["model"].apply(lambda x: x.split("/")[-1])  # Shorten names
    x = np.arange(len(models))
    width = 0.6

    # Left plot: SSI
    ax = axes[0]
    bars = ax.bar(
        x,
        model_metrics["mean_ssi"],
        width,
        color=COLORBLIND_PALETTE[0],
        yerr=model_metrics["std_ssi"],
        capsize=5,
    )
    ax.axhline(y=0.8, color=COLORBLIND_PALETTE[3], linestyle="--", label="Stability threshold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean SSI")
    ax.set_title("Stability Index by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    # Right plot: Label rates
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.25

    ax.bar(
        x - width,
        model_metrics["mean_refusal_rate"] * 100,
        width,
        label="Refuse",
        color=COLORBLIND_PALETTE[2],
    )
    ax.bar(
        x,
        model_metrics["mean_partial_rate"] * 100,
        width,
        label="Partial",
        color=COLORBLIND_PALETTE[1],
    )
    ax.bar(
        x + width,
        model_metrics["mean_comply_rate"] * 100,
        width,
        label="Comply",
        color=COLORBLIND_PALETTE[3],
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Response Distribution by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig


def create_stability_vs_compliance_scatter(
    prompt_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create scatter plot of SSI vs compliance rate.

    Args:
        prompt_metrics: Per-prompt metrics DataFrame
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        prompt_metrics["comply_rate"] * 100,
        prompt_metrics["stability_index"],
        alpha=0.5,
        s=30,
        color=COLORBLIND_PALETTE[0],
    )

    # Add threshold lines
    ax.axhline(y=0.8, color=COLORBLIND_PALETTE[3], linestyle="--", alpha=0.7)
    ax.axvline(x=50, color=COLORBLIND_PALETTE[1], linestyle="--", alpha=0.7)

    # Annotate regions
    ax.annotate(
        "Stable Refusers",
        xy=(10, 0.95),
        fontsize=FONT_SIZE - 2,
        color="gray",
    )
    ax.annotate(
        "Unstable\n(Borderline)",
        xy=(45, 0.4),
        fontsize=FONT_SIZE - 2,
        color="gray",
        ha="center",
    )
    ax.annotate(
        "Stable Compliers",
        xy=(80, 0.95),
        fontsize=FONT_SIZE - 2,
        color="gray",
    )

    ax.set_xlabel("Comply Rate (%)")
    ax.set_ylabel("Stability Index (SSI)")
    ax.set_title("Stability vs Compliance Rate per Prompt")
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig


def create_flip_rate_histogram(
    prompt_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create histogram showing flip occurrence.

    Args:
        prompt_metrics: Per-prompt metrics DataFrame
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    flip_counts = prompt_metrics["flip_occurred"].value_counts()
    labels = ["No Flip\n(Consistent)", "Flip Occurred\n(Inconsistent)"]
    colors = [COLORBLIND_PALETTE[2], COLORBLIND_PALETTE[3]]

    values = [
        flip_counts.get(False, 0),
        flip_counts.get(True, 0),
    ]

    bars = ax.bar(labels, values, color=colors, edgecolor="white")

    # Add percentage labels
    total = sum(values)
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )

    ax.set_ylabel("Number of Prompts")
    ax.set_title("Consistency of Safety Decisions")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig


def create_all_figures(
    metrics_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """
    Create all figures and save to output directory.

    Args:
        metrics_dir: Directory containing metric files
        output_dir: Directory to save figures

    Returns:
        Dict mapping figure name to Figure object
    """
    setup_style()

    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics = load_metrics(metrics_dir)

    figures = {}

    # SSI Distribution
    if "per_prompt_metrics" in metrics:
        print("Creating SSI distribution histogram...")
        figures["ssi_distribution"] = create_ssi_histogram(
            metrics["per_prompt_metrics"],
            output_dir / "ssi_distribution.png",
        )

    # Temperature Effect
    if "aggregate_by_temperature" in metrics:
        print("Creating temperature effect plot...")
        figures["temperature_effect"] = create_temperature_effect_plot(
            metrics["aggregate_by_temperature"],
            output_dir / "temperature_effect.png",
        )

    # Model Comparison
    if "aggregate_by_model" in metrics:
        print("Creating model comparison plot...")
        figures["model_comparison"] = create_model_comparison_bars(
            metrics["aggregate_by_model"],
            output_dir / "model_comparison.png",
        )

    # Stability vs Compliance
    if "per_prompt_metrics" in metrics:
        print("Creating stability vs compliance scatter...")
        figures["stability_vs_compliance"] = create_stability_vs_compliance_scatter(
            metrics["per_prompt_metrics"],
            output_dir / "stability_vs_compliance.png",
        )

    # Flip Rate
    if "per_prompt_metrics" in metrics:
        print("Creating flip rate histogram...")
        figures["flip_rate"] = create_flip_rate_histogram(
            metrics["per_prompt_metrics"],
            output_dir / "flip_rate.png",
        )

    print(f"\nAll figures saved to: {output_dir}")
    return figures
