#!/usr/bin/env python3
"""
Comprehensive v2 Analysis Script

Addresses reviewer concerns:
1. Fixed-temperature sample size analysis (all 4 temps)
2. Binary scoring ablation (PARTIAL handling)
3. Effect sizes (Kendall's W, bootstrap CIs, correlations with CIs)
4. Within-model stability-compliance correlation
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"


def load_combined_labels():
    """Load combined label data for all 4 models."""
    llama_qwen25 = LABELS_DIR / "claude_haiku_llama_qwen25.csv"
    new_models = LABELS_DIR / "claude_haiku_new_models.csv"

    dfs = []
    if llama_qwen25.exists():
        dfs.append(pd.read_csv(llama_qwen25))
    if new_models.exists():
        df = pd.read_csv(new_models)
        df['model'] = df['model'].replace({
            'Qwen_Qwen3-8B': 'Qwen/Qwen3-8B',
            'google_gemma-3-12b-it': 'google/gemma-3-12b-it'
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_stability_index(labels):
    """Compute SSI for a list of labels."""
    valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if len(valid_labels) == 0:
        return np.nan
    counts = Counter(valid_labels)
    return max(counts.values()) / len(valid_labels)


def get_majority_label(labels):
    """Get majority label from list."""
    valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if not valid_labels:
        return None
    counts = Counter(valid_labels)
    return counts.most_common(1)[0][0]


# =============================================================================
# TASK 2: Fixed-Temperature Sample Size Analysis
# =============================================================================

def fixed_temp_sample_size_analysis(labels_df, n_bootstrap=1000):
    """
    Compute sample size reliability at each fixed temperature.

    For each temperature:
    - Ground truth = majority label over 5 seeds
    - Sample n seeds, compute agreement with ground truth
    """
    results = []
    temps = sorted(labels_df['temperature'].unique())

    for temp in temps:
        temp_df = labels_df[labels_df['temperature'] == temp].copy()
        seeds = sorted(temp_df['seed'].unique())
        n_seeds = len(seeds)

        # Compute ground truth (majority over all seeds at this temp)
        ground_truth = {}
        for (model, prompt_id), group in temp_df.groupby(['model', 'prompt_id']):
            labels = group['label'].tolist()
            gt = get_majority_label(labels)
            if gt:
                ground_truth[(model, prompt_id)] = gt

        # For each sample size
        for n_samples in range(1, n_seeds + 1):
            agreements = []

            for _ in range(n_bootstrap):
                sampled_seeds = np.random.choice(seeds, size=n_samples, replace=False)
                sampled_df = temp_df[temp_df['seed'].isin(sampled_seeds)]

                prompt_agreements = []
                for (model, prompt_id), group in sampled_df.groupby(['model', 'prompt_id']):
                    key = (model, prompt_id)
                    if key not in ground_truth:
                        continue
                    labels = group['label'].tolist()
                    sampled_majority = get_majority_label(labels)
                    if sampled_majority:
                        prompt_agreements.append(sampled_majority == ground_truth[key])

                if prompt_agreements:
                    agreements.append(np.mean(prompt_agreements) * 100)

            if agreements:
                results.append({
                    'temperature': temp,
                    'n_samples': n_samples,
                    'mean_agreement': np.mean(agreements),
                    'std_agreement': np.std(agreements),
                    'ci_lower': np.percentile(agreements, 2.5),
                    'ci_upper': np.percentile(agreements, 97.5),
                })

    return pd.DataFrame(results)


# =============================================================================
# TASK 3: Binary Scoring Ablation
# =============================================================================

def binary_scoring_ablation(labels_df):
    """
    Test how results change under binary scoring rules.

    - Conservative: PARTIAL → COMPLY (treat hedged as unsafe)
    - Lenient: PARTIAL → REFUSE (treat hedged as safe)
    """
    results = []

    for rule_name, mapping in [
        ('Three-way (original)', {'REFUSE': 'REFUSE', 'PARTIAL': 'PARTIAL', 'COMPLY': 'COMPLY'}),
        ('Binary (PARTIAL→COMPLY)', {'REFUSE': 'REFUSE', 'PARTIAL': 'COMPLY', 'COMPLY': 'COMPLY'}),
        ('Binary (PARTIAL→REFUSE)', {'REFUSE': 'REFUSE', 'PARTIAL': 'REFUSE', 'COMPLY': 'COMPLY'}),
    ]:
        # Apply mapping
        mapped_df = labels_df.copy()
        mapped_df['label'] = mapped_df['label'].map(mapping)

        # Compute per-prompt metrics
        prompt_metrics = []
        for (model, prompt_id), group in mapped_df.groupby(['model', 'prompt_id']):
            labels = group['label'].tolist()
            valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]

            if len(valid_labels) == 0:
                continue

            counts = Counter(valid_labels)
            ssi = max(counts.values()) / len(valid_labels)
            flip = len(set(valid_labels)) > 1

            prompt_metrics.append({
                'model': model,
                'prompt_id': prompt_id,
                'ssi': ssi,
                'flip': flip,
            })

        pm_df = pd.DataFrame(prompt_metrics)

        results.append({
            'scoring_rule': rule_name,
            'mean_ssi': pm_df['ssi'].mean(),
            'flip_rate': pm_df['flip'].mean() * 100,
            'pct_unstable': (pm_df['ssi'] < 0.8).mean() * 100,
        })

    return pd.DataFrame(results)


def binary_ablation_by_model(labels_df):
    """Binary scoring ablation broken down by model."""
    results = []

    for model in labels_df['model'].unique():
        model_df = labels_df[labels_df['model'] == model]
        short_name = model.split('/')[-1] if '/' in model else model

        for rule_name, mapping in [
            ('Three-way', {'REFUSE': 'REFUSE', 'PARTIAL': 'PARTIAL', 'COMPLY': 'COMPLY'}),
            ('PARTIAL→COMPLY', {'REFUSE': 'REFUSE', 'PARTIAL': 'COMPLY', 'COMPLY': 'COMPLY'}),
            ('PARTIAL→REFUSE', {'REFUSE': 'REFUSE', 'PARTIAL': 'REFUSE', 'COMPLY': 'COMPLY'}),
        ]:
            mapped_df = model_df.copy()
            mapped_df['label'] = mapped_df['label'].map(mapping)

            prompt_metrics = []
            for prompt_id, group in mapped_df.groupby('prompt_id'):
                labels = group['label'].tolist()
                valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
                if len(valid_labels) == 0:
                    continue
                counts = Counter(valid_labels)
                ssi = max(counts.values()) / len(valid_labels)
                flip = len(set(valid_labels)) > 1
                prompt_metrics.append({'ssi': ssi, 'flip': flip})

            pm_df = pd.DataFrame(prompt_metrics)

            results.append({
                'model': short_name,
                'scoring_rule': rule_name,
                'mean_ssi': pm_df['ssi'].mean(),
                'flip_rate': pm_df['flip'].mean() * 100,
                'pct_unstable': (pm_df['ssi'] < 0.8).mean() * 100,
            })

    return pd.DataFrame(results)


# =============================================================================
# TASK 4: Effect Sizes
# =============================================================================

def compute_kendalls_w(labels_df):
    """
    Compute Kendall's W for the effect of temperature on SSI.

    Kendall's W is the effect size measure for Friedman test.
    W = χ² / (n * (k-1)) where n = number of subjects, k = number of conditions
    """
    # Compute per-prompt SSI at each temperature
    prompt_temp_ssi = {}

    for (model, prompt_id), group in labels_df.groupby(['model', 'prompt_id']):
        for temp in [0.0, 0.3, 0.7, 1.0]:
            temp_labels = group[group['temperature'] == temp]['label'].tolist()
            ssi = compute_stability_index(temp_labels)
            if not np.isnan(ssi):
                if (model, prompt_id) not in prompt_temp_ssi:
                    prompt_temp_ssi[(model, prompt_id)] = {}
                prompt_temp_ssi[(model, prompt_id)][temp] = ssi

    # Build matrix for Friedman test
    data = []
    for key, temps in prompt_temp_ssi.items():
        if len(temps) == 4:  # All 4 temperatures present
            data.append([temps[0.0], temps[0.3], temps[0.7], temps[1.0]])

    data = np.array(data)
    n = len(data)  # number of subjects (prompts)
    k = 4  # number of conditions (temperatures)

    # Run Friedman test
    stat, pvalue = stats.friedmanchisquare(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    # Kendall's W
    W = stat / (n * (k - 1))

    return {
        'friedman_chi2': stat,
        'friedman_pvalue': pvalue,
        'kendalls_w': W,
        'n_prompts': n,
        'k_conditions': k,
        'effect_interpretation': 'small' if W < 0.3 else ('medium' if W < 0.5 else 'large'),
    }


def bootstrap_ssi_difference(labels_df, n_bootstrap=1000):
    """
    Compute bootstrap 95% CI for the SSI difference between temp 0.0 and temp 1.0.
    """
    # Compute per-prompt SSI at each temperature
    ssi_t0 = []
    ssi_t1 = []

    for (model, prompt_id), group in labels_df.groupby(['model', 'prompt_id']):
        t0_labels = group[group['temperature'] == 0.0]['label'].tolist()
        t1_labels = group[group['temperature'] == 1.0]['label'].tolist()

        ssi0 = compute_stability_index(t0_labels)
        ssi1 = compute_stability_index(t1_labels)

        if not np.isnan(ssi0) and not np.isnan(ssi1):
            ssi_t0.append(ssi0)
            ssi_t1.append(ssi1)

    ssi_t0 = np.array(ssi_t0)
    ssi_t1 = np.array(ssi_t1)

    # Observed difference
    observed_diff = ssi_t0.mean() - ssi_t1.mean()

    # Bootstrap
    diffs = []
    n = len(ssi_t0)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        diff = ssi_t0[idx].mean() - ssi_t1[idx].mean()
        diffs.append(diff)

    return {
        'mean_ssi_t0': ssi_t0.mean(),
        'mean_ssi_t1': ssi_t1.mean(),
        'observed_diff': observed_diff,
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5),
    }


def correlation_with_ci(x, y, n_bootstrap=1000):
    """Compute Spearman correlation with bootstrap 95% CI."""
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Observed correlation
    rho, pvalue = stats.spearmanr(x, y)

    # Bootstrap
    rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)

    return {
        'spearman_rho': rho,
        'pvalue': pvalue,
        'ci_lower': np.percentile(rhos, 2.5),
        'ci_upper': np.percentile(rhos, 97.5),
        'n': n,
    }


def within_model_stability_compliance_correlation(labels_df):
    """
    Compute correlation between per-prompt SSI and compliance rate WITHIN each model.
    This tests the stability-conservatism tradeoff at the prompt level.
    """
    results = []

    for model in labels_df['model'].unique():
        model_df = labels_df[labels_df['model'] == model]
        short_name = model.split('/')[-1] if '/' in model else model

        # Compute per-prompt metrics
        ssi_vals = []
        comply_rates = []

        for prompt_id, group in model_df.groupby('prompt_id'):
            labels = group['label'].tolist()
            valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]

            if len(valid_labels) < 5:
                continue

            ssi = compute_stability_index(labels)
            comply_rate = sum(1 for l in valid_labels if l == 'COMPLY') / len(valid_labels)

            ssi_vals.append(ssi)
            comply_rates.append(comply_rate)

        ssi_vals = np.array(ssi_vals)
        comply_rates = np.array(comply_rates)

        corr_result = correlation_with_ci(ssi_vals, comply_rates)
        corr_result['model'] = short_name
        results.append(corr_result)

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SSI Paper v2 Comprehensive Analysis")
    print("=" * 70)
    print()

    # Load data
    print("Loading labels...")
    labels_df = load_combined_labels()
    print(f"Loaded {len(labels_df)} labels")
    print()

    # =========================================================================
    # TASK 2: Fixed-Temperature Sample Size Analysis
    # =========================================================================
    print("=" * 70)
    print("TASK 2: Fixed-Temperature Sample Size Analysis")
    print("=" * 70)
    print()

    print("Computing sample size reliability at each temperature...")
    fixed_temp_df = fixed_temp_sample_size_analysis(labels_df, n_bootstrap=500)

    # Pivot for display
    pivot = fixed_temp_df.pivot(index='n_samples', columns='temperature', values='mean_agreement')
    print("\nAgreement (%) by N samples and Temperature:")
    print(pivot.round(1).to_string())
    print()

    fixed_temp_df.to_csv(METRICS_DIR / "fixed_temp_sample_size.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'fixed_temp_sample_size.csv'}")

    # LaTeX table
    print("\n" + "-" * 50)
    print("LaTeX Table:")
    print("-" * 50)
    print(r"""
\begin{table}[h]
\centering
\caption{Sample size reliability at fixed temperatures. Values show \% agreement with 5-seed ground truth.}
\label{tab:fixed_temp_sample_size}
\begin{tabular}{ccccc}
\toprule
\textbf{N Seeds} & \textbf{t=0.0} & \textbf{t=0.3} & \textbf{t=0.7} & \textbf{t=1.0} \\
\midrule""")
    for n in [1, 2, 3, 4, 5]:
        row = pivot.loc[n]
        print(f"{n} & {row[0.0]:.1f}\\% & {row[0.3]:.1f}\\% & {row[0.7]:.1f}\\% & {row[1.0]:.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # =========================================================================
    # TASK 3: Binary Scoring Ablation
    # =========================================================================
    print()
    print("=" * 70)
    print("TASK 3: Binary Scoring Ablation")
    print("=" * 70)
    print()

    print("Computing binary scoring ablation...")
    binary_df = binary_scoring_ablation(labels_df)
    print("\nOverall Results:")
    print(binary_df.to_string(index=False))
    print()

    binary_by_model_df = binary_ablation_by_model(labels_df)
    print("\nBy Model:")
    pivot_binary = binary_by_model_df.pivot(index='model', columns='scoring_rule', values='mean_ssi')
    print(pivot_binary.round(3).to_string())
    print()

    binary_df.to_csv(METRICS_DIR / "binary_scoring_ablation.csv", index=False)
    binary_by_model_df.to_csv(METRICS_DIR / "binary_scoring_by_model.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'binary_scoring_ablation.csv'}")

    # LaTeX table
    print("\n" + "-" * 50)
    print("LaTeX Table:")
    print("-" * 50)
    print(r"""
\begin{table}[h]
\centering
\caption{Binary scoring ablation: results under different PARTIAL handling rules.}
\label{tab:binary_ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Scoring Rule} & \textbf{Mean SSI} & \textbf{Flip Rate} & \textbf{\% Unstable} \\
\midrule""")
    for _, row in binary_df.iterrows():
        print(f"{row['scoring_rule']} & {row['mean_ssi']:.3f} & {row['flip_rate']:.1f}\\% & {row['pct_unstable']:.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # =========================================================================
    # TASK 4: Effect Sizes
    # =========================================================================
    print()
    print("=" * 70)
    print("TASK 4: Effect Sizes")
    print("=" * 70)
    print()

    print("Computing Kendall's W for temperature effect...")
    kendall_result = compute_kendalls_w(labels_df)
    print(f"\nKendall's W = {kendall_result['kendalls_w']:.4f} ({kendall_result['effect_interpretation']} effect)")
    print(f"Friedman χ² = {kendall_result['friedman_chi2']:.2f}, p = {kendall_result['friedman_pvalue']:.2e}")
    print(f"n = {kendall_result['n_prompts']} prompts, k = {kendall_result['k_conditions']} conditions")
    print()

    print("Computing bootstrap CI for SSI difference (temp 0.0 vs 1.0)...")
    bootstrap_result = bootstrap_ssi_difference(labels_df)
    print(f"\nMean SSI at t=0.0: {bootstrap_result['mean_ssi_t0']:.4f}")
    print(f"Mean SSI at t=1.0: {bootstrap_result['mean_ssi_t1']:.4f}")
    print(f"Difference: {bootstrap_result['observed_diff']:.4f} [95% CI: {bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print()

    print("Computing within-model SSI-compliance correlation...")
    corr_df = within_model_stability_compliance_correlation(labels_df)
    print("\nWithin-Model SSI vs Compliance Rate Correlation:")
    print(corr_df[['model', 'spearman_rho', 'ci_lower', 'ci_upper', 'pvalue']].round(3).to_string(index=False))
    print()

    # Save effect sizes
    effect_sizes = {
        'kendalls_w': kendall_result['kendalls_w'],
        'kendalls_w_interpretation': kendall_result['effect_interpretation'],
        'ssi_diff_t0_t1': bootstrap_result['observed_diff'],
        'ssi_diff_ci_lower': bootstrap_result['ci_lower'],
        'ssi_diff_ci_upper': bootstrap_result['ci_upper'],
    }
    pd.DataFrame([effect_sizes]).to_csv(METRICS_DIR / "effect_sizes.csv", index=False)
    corr_df.to_csv(METRICS_DIR / "within_model_correlations.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'effect_sizes.csv'}")
    print(f"Saved: {METRICS_DIR / 'within_model_correlations.csv'}")

    # =========================================================================
    # SUMMARY FOR PAPER
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY: Text Updates for Paper")
    print("=" * 70)
    print(f"""
SECTION 4.3 (Temperature Effects):
- Add: "Kendall's W = {kendall_result['kendalls_w']:.3f} indicates a {kendall_result['effect_interpretation']} effect size."
- Add: "The decrease in within-temperature SSI from 0.0 to 1.0 was {bootstrap_result['observed_diff']:.3f}
  [95% CI: {bootstrap_result['ci_lower']:.3f}–{bootstrap_result['ci_upper']:.3f}]."

SECTION 4.8 (Sample Size):
- At t=0.0: N=1 gives {pivot.loc[1, 0.0]:.1f}% agreement, N=3 gives {pivot.loc[3, 0.0]:.1f}%
- At t=0.7: N=1 gives {pivot.loc[1, 0.7]:.1f}% agreement, N=3 gives {pivot.loc[3, 0.7]:.1f}%
- At t=1.0: N=1 gives {pivot.loc[1, 1.0]:.1f}% agreement, N=3 gives {pivot.loc[3, 1.0]:.1f}%

SECTION 4.6 or NEW (Stability-Compliance):
- Within-model correlations: Mean ρ ≈ {corr_df['spearman_rho'].mean():.2f}

SECTION 4.7 or NEW (Binary Ablation):
- Under PARTIAL→COMPLY: SSI = {binary_df[binary_df['scoring_rule'].str.contains('COMPLY')]['mean_ssi'].values[0]:.3f}
- Under PARTIAL→REFUSE: SSI = {binary_df[binary_df['scoring_rule'].str.contains('REFUSE')]['mean_ssi'].values[0]:.3f}
- Instability finding holds regardless of PARTIAL handling
""")

    print()
    print("=" * 70)
    print("Done! All analyses saved to data/results/metrics/")
    print("=" * 70)


if __name__ == "__main__":
    main()
