"""
Distribution visualizations for safety refusal stability analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_ssi_histogram(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    threshold: float = 0.8,
) -> plt.Figure:
    """
    Create histogram of SSI distribution.

    Args:
        metrics_df: DataFrame with stability_index column
        output_path: Path to save figure
        threshold: Threshold for stability

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ssi = metrics_df["stability_index"]

    ax.hist(ssi, bins=30, edgecolor="white", alpha=0.8)
    ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")

    pct_unstable = (ssi < threshold).mean() * 100
    ax.set_xlabel("Stability Index (SSI)")
    ax.set_ylabel("Count")
    ax.set_title(f"SSI Distribution ({pct_unstable:.1f}% unstable)")
    ax.legend()

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")

    return fig


def create_label_distribution_pie(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create pie chart of label distribution.

    Args:
        metrics_df: DataFrame with refusal_rate, partial_rate, comply_rate
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["Refuse", "Partial", "Comply"]
    sizes = [
        metrics_df["refusal_rate"].mean(),
        metrics_df["partial_rate"].mean(),
        metrics_df["comply_rate"].mean(),
    ]
    colors = sns.color_palette("colorblind")[:3]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax.set_title("Average Label Distribution")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")

    return fig
