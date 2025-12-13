#!/usr/bin/env python3
"""
Regenerate multimodel figures (Figures 5 and 6) with all 4 models.

Figure 5: Safety Stability Index by Model (bar chart with mean SSI and % unstable)
Figure 6: Safety Response Distribution by Model (stacked bar chart with REFUSE/PARTIAL/COMPLY)

Uses the unified Claude Haiku judge data for all 4 models.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
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


def load_unified_metrics():
    """Load the unified Claude Haiku judge metrics for all 4 models."""
    metrics_file = METRICS_DIR / "unified_claude_haiku_all_models.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    df = pd.read_csv(metrics_file)
    print(f"Loaded metrics for {len(df)} models from {metrics_file}")
    return df


def load_combined_labels():
    """Load combined label data for all 4 models."""
    # Combine the two Claude Haiku label files
    llama_qwen25 = LABELS_DIR / "claude_haiku_llama_qwen25.csv"
    new_models = LABELS_DIR / "claude_haiku_new_models.csv"

    dfs = []
    if llama_qwen25.exists():
        df = pd.read_csv(llama_qwen25)
        dfs.append(df)
        print(f"Loaded {len(df)} labels from claude_haiku_llama_qwen25.csv")

    if new_models.exists():
        df = pd.read_csv(new_models)
        # Normalize model names
        df['model'] = df['model'].replace({
            'Qwen_Qwen3-8B': 'Qwen/Qwen3-8B',
            'google_gemma-3-12b-it': 'google/gemma-3-12b-it'
        })
        dfs.append(df)
        print(f"Loaded {len(df)} labels from claude_haiku_new_models.csv")

    if not dfs:
        raise FileNotFoundError("No label files found!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total combined labels: {len(combined)}")
    return combined


def compute_response_distribution(labels_df):
    """Compute REFUSE/PARTIAL/COMPLY distribution per model."""
    results = []

    for model in labels_df['model'].unique():
        model_df = labels_df[labels_df['model'] == model]
        total = len(model_df)

        refuse_rate = (model_df['label'] == 'REFUSE').sum() / total
        partial_rate = (model_df['label'] == 'PARTIAL').sum() / total
        comply_rate = (model_df['label'] == 'COMPLY').sum() / total

        # Get short model name
        short_name = model.split('/')[-1] if '/' in model else model
        short_name = short_name.replace('_', ' ').replace('-Instruct', '')

        results.append({
            'model': model,
            'short_name': short_name,
            'refuse_rate': refuse_rate,
            'partial_rate': partial_rate,
            'comply_rate': comply_rate,
        })

    return pd.DataFrame(results)


def create_figure5_stability_by_model(metrics_df, output_path):
    """
    Create Figure 5: Safety Stability Index by Model.

    Left panel: Mean SSI with std error bars
    Right panel: Percentage of unstable prompts (SSI < 0.8)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by SSI descending (as in Table 2)
    metrics_df = metrics_df.sort_values('mean_ssi', ascending=False).copy()

    # Create short names for display
    metrics_df['short_name'] = metrics_df['model'].apply(
        lambda x: x.replace('Llama 3.1 8B', 'Llama-3.1-8B')
                   .replace('Qwen 2.5 7B', 'Qwen2.5-7B')
                   .replace('Qwen 3 8B', 'Qwen3-8B')
                   .replace('Gemma 3 12B', 'Gemma-3-12B')
    )

    models = metrics_df['short_name'].tolist()
    x = np.arange(len(models))
    width = 0.6

    # Left plot: Mean SSI
    ax = axes[0]
    bars = ax.bar(
        x,
        metrics_df['mean_ssi'],
        width,
        color=COLORBLIND_PALETTE[0],
        edgecolor='white',
    )

    # Add value labels on bars
    for bar, val in zip(bars, metrics_df['mean_ssi']):
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom',
            fontsize=FONT_SIZE - 2,
        )

    ax.axhline(y=0.8, color=COLORBLIND_PALETTE[3], linestyle='--',
               linewidth=1.5, label='Instability threshold (0.8)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Safety Stability Index')
    ax.set_title('(a) Mean SSI by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right')

    # Right plot: % Unstable
    ax = axes[1]
    bars = ax.bar(
        x,
        metrics_df['pct_unstable'],
        width,
        color=COLORBLIND_PALETTE[3],
        edgecolor='white',
    )

    # Add value labels on bars
    for bar, val in zip(bars, metrics_df['pct_unstable']):
        ax.annotate(
            f'{val:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom',
            fontsize=FONT_SIZE - 2,
        )

    ax.set_xlabel('Model')
    ax.set_ylabel('% Unstable Prompts (SSI < 0.8)')
    ax.set_title('(b) Proportion of Unstable Prompts')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 15)

    plt.tight_layout()

    # Save figures
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved Figure 5: {output_path}")

    plt.close(fig)
    return fig


def create_figure6_response_distribution(distribution_df, output_path):
    """
    Create Figure 6: Safety Response Distribution by Model.

    Stacked bar chart showing REFUSE/PARTIAL/COMPLY proportions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by refusal rate (or match Table 2 order by SSI)
    # Use same order as Figure 5 (by SSI descending)
    model_order = ['Gemma-3-12b-it', 'Llama-3.1-8B', 'Qwen3-8B', 'Qwen2.5-7B']

    # Create ordered DataFrame
    ordered_data = []
    for model_name in model_order:
        for _, row in distribution_df.iterrows():
            if model_name.lower().replace('-', '').replace('.', '') in row['short_name'].lower().replace('-', '').replace('.', '').replace(' ', ''):
                ordered_data.append(row)
                break

    if not ordered_data:
        # Fallback: use existing order
        ordered_data = distribution_df.to_dict('records')

    distribution_df = pd.DataFrame(ordered_data)

    models = distribution_df['short_name'].tolist()
    x = np.arange(len(models))
    width = 0.6

    # Create stacked bars
    refuse = distribution_df['refuse_rate'] * 100
    partial = distribution_df['partial_rate'] * 100
    comply = distribution_df['comply_rate'] * 100

    # Green for refuse (safe), orange for partial, red for comply (unsafe)
    ax.bar(x, refuse, width, label='REFUSE', color=COLORBLIND_PALETTE[2])
    ax.bar(x, partial, width, bottom=refuse, label='PARTIAL', color=COLORBLIND_PALETTE[1])
    ax.bar(x, comply, width, bottom=refuse + partial, label='COMPLY', color=COLORBLIND_PALETTE[3])

    # Add percentage labels inside bars
    for i, (r, p, c) in enumerate(zip(refuse, partial, comply)):
        if r > 5:
            ax.annotate(f'{r:.1f}%', xy=(i, r/2), ha='center', va='center',
                       fontsize=FONT_SIZE - 2, color='white', fontweight='bold')
        if p > 5:
            ax.annotate(f'{p:.1f}%', xy=(i, r + p/2), ha='center', va='center',
                       fontsize=FONT_SIZE - 2, color='white', fontweight='bold')
        if c > 3:
            ax.annotate(f'{c:.1f}%', xy=(i, r + p + c/2), ha='center', va='center',
                       fontsize=FONT_SIZE - 2, color='white', fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('Response Distribution (%)')
    ax.set_title('Safety Response Distribution by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')

    plt.tight_layout()

    # Save figures
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved Figure 6: {output_path}")

    plt.close(fig)
    return fig


def main():
    print("=" * 60)
    print("Regenerating Multimodel Figures (Figures 5 and 6)")
    print("=" * 60)
    print()

    setup_style()

    # Load data
    print("Loading metrics...")
    metrics_df = load_unified_metrics()
    print(metrics_df.to_string())
    print()

    print("Loading labels...")
    labels_df = load_combined_labels()
    print()

    # Compute response distribution
    print("Computing response distribution...")
    distribution_df = compute_response_distribution(labels_df)
    print(distribution_df.to_string())
    print()

    # Create figures
    print("Creating Figure 5 (Stability by Model)...")
    create_figure5_stability_by_model(
        metrics_df,
        FIGURES_DIR / "multimodel_stability.png"
    )

    print("Creating Figure 6 (Response Distribution)...")
    create_figure6_response_distribution(
        distribution_df,
        FIGURES_DIR / "multimodel_distribution.png"
    )

    print()
    print("=" * 60)
    print("Done! Figures saved to paper/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
