"""
Heatmap visualizations for safety refusal stability analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_model_category_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metric_col: str = "stability_index",
    model_col: str = "model",
    category_col: str = "harm_category",
) -> plt.Figure:
    """
    Create heatmap of metric by model and harm category.

    Args:
        metrics_df: DataFrame with metrics including model and category
        output_path: Path to save figure
        metric_col: Column containing metric values
        model_col: Column containing model names
        category_col: Column containing harm categories

    Returns:
        Matplotlib figure
    """
    # Pivot to get model × category matrix
    pivot = metrics_df.pivot_table(
        values=metric_col,
        index=category_col,
        columns=model_col,
        aggfunc="mean",
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Red = low (unstable), Green = high (stable)
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": f"Mean {metric_col}"},
    )

    ax.set_title(f"{metric_col.replace('_', ' ').title()} by Model and Harm Category")
    ax.set_xlabel("Model")
    ax.set_ylabel("Harm Category")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        print(f"Saved: {output_path}")

    return fig
