#!/usr/bin/env python3
"""
Within-Temperature Stability Analysis

This script computes SSI separately for each temperature level, using only
the 5 seeds at that temperature. This separates temperature effects from
seed effects.

Creates:
- Table showing within-temperature SSI by temperature level
- Sample size reliability analysis at fixed temperature (temp=0.7)
- New figure for within-temperature stability
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
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


def load_combined_labels():
    """Load combined label data for all 4 models."""
    llama_qwen25 = LABELS_DIR / "claude_haiku_llama_qwen25.csv"
    new_models = LABELS_DIR / "claude_haiku_new_models.csv"

    dfs = []
    if llama_qwen25.exists():
        df = pd.read_csv(llama_qwen25)
        dfs.append(df)

    if new_models.exists():
        df = pd.read_csv(new_models)
        df['model'] = df['model'].replace({
            'Qwen_Qwen3-8B': 'Qwen/Qwen3-8B',
            'google_gemma-3-12b-it': 'google/gemma-3-12b-it'
        })
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} total labels")
    return combined


def compute_stability_index(labels):
    """Compute SSI for a list of labels."""
    valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if len(valid_labels) == 0:
        return 0.0
    counts = Counter(valid_labels)
    return max(counts.values()) / len(valid_labels)


def compute_flip_occurred(labels):
    """Check if labels vary."""
    valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    return len(set(valid_labels)) > 1


def compute_within_temperature_ssi(labels_df):
    """
    Compute SSI separately for each temperature level.

    For each prompt at each temperature, uses only the 5 seeds.
    This measures pure seed variance (within-temperature stability).
    """
    results = []

    for temp in sorted(labels_df['temperature'].unique()):
        temp_df = labels_df[labels_df['temperature'] == temp]

        prompt_metrics = []
        for prompt_id, group in temp_df.groupby('prompt_id'):
            labels = group['label'].tolist()
            ssi = compute_stability_index(labels)
            flip = compute_flip_occurred(labels)
            prompt_metrics.append({
                'prompt_id': prompt_id,
                'ssi': ssi,
                'flip': flip,
                'n_samples': len(labels)
            })

        pm_df = pd.DataFrame(prompt_metrics)

        results.append({
            'temperature': temp,
            'mean_ssi': pm_df['ssi'].mean(),
            'std_ssi': pm_df['ssi'].std(),
            'median_ssi': pm_df['ssi'].median(),
            'pct_unstable': (pm_df['ssi'] < 0.8).mean() * 100,
            'flip_rate': pm_df['flip'].mean() * 100,
            'n_prompts': len(pm_df),
        })

    return pd.DataFrame(results)


def compute_within_temperature_by_model(labels_df):
    """
    Compute within-temperature SSI for each model separately.
    """
    results = []

    for model in labels_df['model'].unique():
        model_df = labels_df[labels_df['model'] == model]
        short_name = model.split('/')[-1] if '/' in model else model

        for temp in sorted(model_df['temperature'].unique()):
            temp_df = model_df[model_df['temperature'] == temp]

            prompt_metrics = []
            for prompt_id, group in temp_df.groupby('prompt_id'):
                labels = group['label'].tolist()
                ssi = compute_stability_index(labels)
                flip = compute_flip_occurred(labels)
                prompt_metrics.append({'ssi': ssi, 'flip': flip})

            pm_df = pd.DataFrame(prompt_metrics)

            results.append({
                'model': short_name,
                'temperature': temp,
                'mean_ssi': pm_df['ssi'].mean(),
                'pct_unstable': (pm_df['ssi'] < 0.8).mean() * 100,
                'flip_rate': pm_df['flip'].mean() * 100,
            })

    return pd.DataFrame(results)


def compute_sample_size_reliability_fixed_temp(labels_df, target_temp=0.7, n_bootstrap=1000):
    """
    Compute sample size reliability at a fixed temperature.

    This answers: "If I run a benchmark at temp=0.7, how many seeds do I need?"
    """
    # Filter to target temperature
    temp_df = labels_df[labels_df['temperature'] == target_temp].copy()

    # Get ground truth (majority label using all 5 seeds)
    ground_truth = {}
    for prompt_id, group in temp_df.groupby('prompt_id'):
        labels = group['label'].tolist()
        counts = Counter(labels)
        ground_truth[prompt_id] = counts.most_common(1)[0][0]

    results = []
    seeds = sorted(temp_df['seed'].unique())
    max_n = len(seeds)  # 5 seeds

    for n_samples in range(1, max_n + 1):
        agreements = []

        for _ in range(n_bootstrap):
            sampled_seeds = np.random.choice(seeds, size=n_samples, replace=False)
            sampled_df = temp_df[temp_df['seed'].isin(sampled_seeds)]

            prompt_agreements = []
            for prompt_id, group in sampled_df.groupby('prompt_id'):
                if prompt_id not in ground_truth:
                    continue
                labels = group['label'].tolist()
                counts = Counter(labels)
                sampled_majority = counts.most_common(1)[0][0]
                prompt_agreements.append(sampled_majority == ground_truth[prompt_id])

            agreements.append(np.mean(prompt_agreements) * 100)

        results.append({
            'n_samples': n_samples,
            'mean_agreement': np.mean(agreements),
            'std_agreement': np.std(agreements),
            'ci_lower': np.percentile(agreements, 2.5),
            'ci_upper': np.percentile(agreements, 97.5),
        })

    return pd.DataFrame(results)


def create_within_temp_figure(within_temp_df, output_path):
    """Create figure showing within-temperature stability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    temps = within_temp_df['temperature']

    # Left: Mean SSI by temperature
    ax = axes[0]
    ax.bar(temps.astype(str), within_temp_df['mean_ssi'],
           color=COLORBLIND_PALETTE[0], edgecolor='white')
    ax.axhline(y=0.8, color=COLORBLIND_PALETTE[3], linestyle='--',
               label='Instability threshold')

    for i, (t, ssi) in enumerate(zip(temps, within_temp_df['mean_ssi'])):
        ax.annotate(f'{ssi:.3f}', xy=(i, ssi + 0.005), ha='center',
                   fontsize=FONT_SIZE - 2)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mean Within-Temperature SSI')
    ax.set_title('(a) Seed Variance Only (5 seeds per temperature)')
    ax.set_ylim(0.9, 1.0)
    ax.legend()

    # Right: Flip rate by temperature
    ax = axes[1]
    ax.bar(temps.astype(str), within_temp_df['flip_rate'],
           color=COLORBLIND_PALETTE[1], edgecolor='white')

    for i, (t, fr) in enumerate(zip(temps, within_temp_df['flip_rate'])):
        ax.annotate(f'{fr:.1f}%', xy=(i, fr + 0.5), ha='center',
                   fontsize=FONT_SIZE - 2)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Flip Rate (%)')
    ax.set_title('(b) % of Prompts with Different Labels Across Seeds')
    ax.set_ylim(0, max(within_temp_df['flip_rate']) * 1.2)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved: {output_path}")

    plt.close(fig)


def create_sample_size_fixed_temp_figure(reliability_df, temp, output_path):
    """Create sample size reliability figure for fixed temperature."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = reliability_df['n_samples']
    mean_agr = reliability_df['mean_agreement']
    ci_lower = reliability_df['ci_lower']
    ci_upper = reliability_df['ci_upper']

    ax.plot(n, mean_agr, 'o-', markersize=10, linewidth=2,
            color=COLORBLIND_PALETTE[0])
    ax.fill_between(n, ci_lower, ci_upper, alpha=0.3, color=COLORBLIND_PALETTE[0])

    # Add reference lines
    ax.axhline(y=95, color=COLORBLIND_PALETTE[2], linestyle='--',
               label='95% reliability')
    ax.axhline(y=99, color=COLORBLIND_PALETTE[1], linestyle=':',
               label='99% reliability')

    # Annotate values
    for i, (ns, agr) in enumerate(zip(n, mean_agr)):
        ax.annotate(f'{agr:.1f}%', xy=(ns, agr + 1), ha='center',
                   fontsize=FONT_SIZE - 2)

    ax.set_xlabel('Number of Seeds')
    ax.set_ylabel('Agreement with Ground Truth (%)')
    ax.set_title(f'Sample Size Reliability at Temperature {temp}')
    ax.set_xticks(n)
    ax.set_ylim(85, 101)
    ax.legend(loc='lower right')

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved: {output_path}")

    plt.close(fig)


def main():
    print("=" * 60)
    print("Within-Temperature Stability Analysis")
    print("=" * 60)
    print()

    setup_style()

    # Load data
    print("Loading labels...")
    labels_df = load_combined_labels()
    print()

    # Compute within-temperature SSI (all models combined)
    print("Computing within-temperature SSI (all models)...")
    within_temp_df = compute_within_temperature_ssi(labels_df)
    print("\nWithin-Temperature Stability (seed variance only):")
    print(within_temp_df.to_string(index=False))
    print()

    # Save to CSV
    within_temp_df.to_csv(METRICS_DIR / "within_temperature_ssi.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'within_temperature_ssi.csv'}")

    # Compute within-temperature by model
    print("\nComputing within-temperature SSI by model...")
    by_model_df = compute_within_temperature_by_model(labels_df)

    # Pivot for nice display
    pivot_ssi = by_model_df.pivot(index='model', columns='temperature', values='mean_ssi')
    print("\nWithin-Temperature SSI by Model:")
    print(pivot_ssi.round(3).to_string())
    print()

    pivot_flip = by_model_df.pivot(index='model', columns='temperature', values='flip_rate')
    print("Within-Temperature Flip Rate (%) by Model:")
    print(pivot_flip.round(1).to_string())
    print()

    by_model_df.to_csv(METRICS_DIR / "within_temperature_by_model.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'within_temperature_by_model.csv'}")

    # Sample size reliability at fixed temperature
    print("\nComputing sample size reliability at temp=0.7...")
    reliability_df = compute_sample_size_reliability_fixed_temp(labels_df, target_temp=0.7)
    print("\nSample Size Reliability at Temperature 0.7:")
    print(reliability_df.to_string(index=False))
    print()

    reliability_df.to_csv(METRICS_DIR / "sample_size_reliability_temp07.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'sample_size_reliability_temp07.csv'}")

    # Create figures
    print("\nCreating figures...")
    create_within_temp_figure(within_temp_df, FIGURES_DIR / "within_temperature_stability.png")
    create_sample_size_fixed_temp_figure(reliability_df, 0.7,
                                         FIGURES_DIR / "sample_size_reliability_temp07.png")

    # Print LaTeX table for paper
    print("\n" + "=" * 60)
    print("LaTeX Table for Paper (Within-Temperature Stability)")
    print("=" * 60)
    print(r"""
\begin{table}[h]
\centering
\caption{Within-Temperature Stability (seed variance only). Each cell uses N=5 seeds at the specified temperature.}
\label{tab:within_temp}
\begin{tabular}{cccc}
\toprule
\textbf{Temperature} & \textbf{Mean SSI} & \textbf{\% Unstable} & \textbf{Flip Rate (\%)} \\
\midrule""")
    for _, row in within_temp_df.iterrows():
        print(f"{row['temperature']:.1f} & {row['mean_ssi']:.3f} & {row['pct_unstable']:.1f}\\% & {row['flip_rate']:.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
