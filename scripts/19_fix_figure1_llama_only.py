#!/usr/bin/env python3
"""
Fix Figure 1: Generate flip rate chart from Llama-only data.
This fixes the inconsistency where Figure 1 showed pooled multi-model data
but the caption claims "Llama 3.1 8B Instruct".
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

def compute_ssi(labels):
    """Compute Safety Stability Index."""
    valid = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if not valid:
        return None
    counts = Counter(valid)
    return max(counts.values()) / len(valid)

def main():
    print("=" * 70)
    print("Task 0: Fix Figure 1 - Regenerate from Llama-only data")
    print("=" * 70)

    # Load Llama 3.1 8B labels judged by Llama 70B (for Section 4.1 consistency)
    # These are the individual label files, not the Claude Haiku combined file
    llama_files = list(LABELS_DIR.glob("meta_llama_Llama_3.1_8B_Instruct_*_labels.csv"))

    if llama_files:
        dfs = []
        for f in llama_files:
            df = pd.read_csv(f)
            dfs.append(df)
        llama_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded Llama 3.1 8B data (Llama 70B judge) from {len(llama_files)} files: {len(llama_df)} rows")
    else:
        llama_df = None

    if llama_df is None or len(llama_df) == 0:
        print("ERROR: Could not find Llama data")
        return

    print(f"\nLlama data: {len(llama_df)} total labels")
    print(f"Unique prompts: {llama_df['prompt_id'].nunique()}")

    # Compute per-prompt metrics for Llama only
    prompt_metrics = []
    for prompt_id, group in llama_df.groupby('prompt_id'):
        labels = [l for l in group['label'].tolist() if l in {'REFUSE', 'PARTIAL', 'COMPLY'}]
        if not labels:
            continue

        ssi = compute_ssi(labels)
        flip = len(set(labels)) > 1

        prompt_metrics.append({
            'prompt_id': prompt_id,
            'ssi': ssi,
            'flip': flip,
            'n_samples': len(labels)
        })

    pm_df = pd.DataFrame(prompt_metrics)

    # Only count prompts with enough samples (20 configs = 4 temps × 5 seeds)
    # But for single-model, we have 20 samples per prompt
    pm_df = pm_df[pm_df['n_samples'] >= 5]  # At least 5 samples

    n_flip = pm_df['flip'].sum()
    n_consistent = len(pm_df) - n_flip
    total = len(pm_df)

    print(f"\n" + "=" * 50)
    print("LLAMA 3.1 8B RESULTS (for Figure 1):")
    print("=" * 50)
    print(f"Total prompts: {total}")
    print(f"Consistent (no flip): {n_consistent} ({n_consistent/total*100:.1f}%)")
    print(f"Flip occurred: {n_flip} ({n_flip/total*100:.1f}%)")
    print(f"Mean SSI: {pm_df['ssi'].mean():.3f}")
    print(f"% Unstable (SSI < 0.8): {(pm_df['ssi'] < 0.8).mean()*100:.1f}%")

    # Verify this matches text expectations (~68% consistent, ~32% flip)
    if 60 <= n_consistent/total*100 <= 75:
        print("\n✓ Data matches expected range (60-75% consistent)")
    else:
        print(f"\n⚠ Data may not match text expectations!")
        print(f"  Expected: ~68% consistent, ~32% flip")
        print(f"  Got: {n_consistent/total*100:.1f}% consistent, {n_flip/total*100:.1f}% flip")

    # Generate figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["No Flip\n(Consistent)", "Flip Occurred\n(Inconsistent)"]
    values = [n_consistent, n_flip]
    colors = ['#2ecc71', '#e74c3c']  # Green, Red

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)

    # Add percentage labels
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Number of Prompts', fontsize=12)
    ax.set_title(f'Safety Decision Consistency\n(Llama 3.1 8B Instruct, N={total} prompts)', fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'flip_rate.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'flip_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {FIGURES_DIR / 'flip_rate.pdf'}")
    print(f"✓ Saved: {FIGURES_DIR / 'flip_rate.png'}")

if __name__ == "__main__":
    main()
