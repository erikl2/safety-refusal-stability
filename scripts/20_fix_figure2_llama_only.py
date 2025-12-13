#!/usr/bin/env python3
"""
Fix Figure 2: Generate SSI distribution histogram from Llama-only data.
This fixes the inconsistency where Figure 2 showed pooled multi-model data (28.7%)
but the caption claims "14.3% of prompts fall below this threshold".
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Style settings
FIGURE_DPI = 300
FONT_SIZE = 12


def compute_ssi(labels):
    """Compute Safety Stability Index."""
    valid = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if not valid:
        return None
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


def main():
    print("=" * 70)
    print("Task 1: Fix Figure 2 - Regenerate SSI Distribution from Llama-only data")
    print("=" * 70)

    # Load Llama 3.1 8B labels judged by Llama 70B (for Section 4.1 consistency)
    llama_files = list(LABELS_DIR.glob("meta_llama_Llama_3.1_8B_Instruct_*_labels.csv"))

    if llama_files:
        dfs = []
        for f in llama_files:
            df = pd.read_csv(f)
            dfs.append(df)
        llama_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded Llama 3.1 8B data (Llama 70B judge) from {len(llama_files)} files: {len(llama_df)} rows")
    else:
        print("ERROR: Could not find Llama 70B judge label files")
        return

    # Compute per-prompt SSI
    ssi_values = []
    for prompt_id, group in llama_df.groupby('prompt_id'):
        labels = group['label'].tolist()
        ssi = compute_ssi(labels)
        if ssi is not None:
            ssi_values.append(ssi)

    ssi_values = np.array(ssi_values)
    n_prompts = len(ssi_values)
    pct_unstable = (ssi_values < 0.8).mean() * 100
    mean_ssi = ssi_values.mean()

    print(f"\nLlama 3.1 8B SSI Statistics:")
    print(f"  Total prompts: {n_prompts}")
    print(f"  Mean SSI: {mean_ssi:.3f}")
    print(f"  % unstable (SSI < 0.8): {pct_unstable:.1f}%")

    # Verify this matches expected 14.3%
    if 13 <= pct_unstable <= 16:
        print("\n✓ Data matches expected ~14.3% unstable")
    else:
        print(f"\n⚠ Data may not match text expectations!")
        print(f"  Expected: ~14.3% unstable")
        print(f"  Got: {pct_unstable:.1f}% unstable")

    # Generate figure
    plt.style.use('seaborn-v0_8-whitegrid')
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

    COLORBLIND_PALETTE = sns.color_palette("colorblind")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram
    ax.hist(
        ssi_values,
        bins=30,
        color=COLORBLIND_PALETTE[0],
        edgecolor="white",
        alpha=0.8,
    )

    # Add threshold line
    threshold = 0.8
    ax.axvline(
        x=threshold,
        color=COLORBLIND_PALETTE[3],
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )

    # Annotate percentage below threshold
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

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'ssi_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'ssi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {FIGURES_DIR / 'ssi_distribution.pdf'}")
    print(f"✓ Saved: {FIGURES_DIR / 'ssi_distribution.png'}")


if __name__ == "__main__":
    main()
