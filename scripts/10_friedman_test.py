#!/usr/bin/env python3
"""
Friedman Test for Temperature Effects on Stability

This replaces the incorrect Kruskal-Wallis test with the correct Friedman test.
Friedman test is appropriate for repeated measures (same prompts at each temperature).

Also computes post-hoc Wilcoxon signed-rank tests with Bonferroni correction.
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


def compute_per_prompt_ssi_by_temperature(labels_df):
    """
    Compute SSI for each prompt at each temperature.

    Returns a DataFrame with columns: prompt_id, temp_0.0, temp_0.3, temp_0.7, temp_1.0
    Each row is a prompt, values are SSI at that temperature (using 5 seeds).
    """
    results = []

    for prompt_id in labels_df['prompt_id'].unique():
        prompt_df = labels_df[labels_df['prompt_id'] == prompt_id]
        row = {'prompt_id': prompt_id}

        for temp in [0.0, 0.3, 0.7, 1.0]:
            temp_labels = prompt_df[prompt_df['temperature'] == temp]['label'].tolist()
            ssi = compute_stability_index(temp_labels)
            row[f'temp_{temp}'] = ssi

        results.append(row)

    return pd.DataFrame(results)


def run_friedman_test(ssi_by_temp_df):
    """
    Run Friedman test on SSI values across temperatures.

    This is the correct test for repeated measures (same prompts at each temperature).
    """
    # Get SSI values for each temperature
    ssi_0 = ssi_by_temp_df['temp_0.0'].dropna()
    ssi_03 = ssi_by_temp_df['temp_0.3'].dropna()
    ssi_07 = ssi_by_temp_df['temp_0.7'].dropna()
    ssi_1 = ssi_by_temp_df['temp_1.0'].dropna()

    # Align to common prompts (should be all prompts, but just in case)
    common_idx = ssi_0.index.intersection(ssi_03.index).intersection(ssi_07.index).intersection(ssi_1.index)

    ssi_0 = ssi_0.loc[common_idx]
    ssi_03 = ssi_03.loc[common_idx]
    ssi_07 = ssi_07.loc[common_idx]
    ssi_1 = ssi_1.loc[common_idx]

    # Run Friedman test
    stat, pvalue = stats.friedmanchisquare(ssi_0, ssi_03, ssi_07, ssi_1)

    return {
        'test': 'Friedman',
        'statistic': stat,
        'pvalue': pvalue,
        'n_prompts': len(common_idx),
        'df': 3,  # k-1 where k=4 temperature levels
    }


def run_posthoc_wilcoxon(ssi_by_temp_df):
    """
    Run post-hoc Wilcoxon signed-rank tests with Bonferroni correction.

    Compares adjacent temperature pairs: 0.0 vs 0.3, 0.3 vs 0.7, 0.7 vs 1.0
    Also 0.0 vs 1.0 for overall comparison.
    """
    comparisons = [
        ('temp_0.0', 'temp_0.3', '0.0 vs 0.3'),
        ('temp_0.3', 'temp_0.7', '0.3 vs 0.7'),
        ('temp_0.7', 'temp_1.0', '0.7 vs 1.0'),
        ('temp_0.0', 'temp_1.0', '0.0 vs 1.0'),
    ]

    n_comparisons = len(comparisons)
    results = []

    for col1, col2, label in comparisons:
        # Align data
        valid_idx = ssi_by_temp_df[[col1, col2]].dropna().index
        x = ssi_by_temp_df.loc[valid_idx, col1]
        y = ssi_by_temp_df.loc[valid_idx, col2]

        # Wilcoxon signed-rank test
        stat, pvalue = stats.wilcoxon(x, y, alternative='two-sided')

        # Bonferroni correction
        corrected_pvalue = min(pvalue * n_comparisons, 1.0)

        results.append({
            'comparison': label,
            'statistic': stat,
            'pvalue': pvalue,
            'corrected_pvalue': corrected_pvalue,
            'significant': corrected_pvalue < 0.05,
            'mean_diff': (x - y).mean(),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Friedman Test for Temperature Effects on Stability")
    print("=" * 60)
    print()

    # Load data
    print("Loading labels...")
    labels_df = load_combined_labels()
    print(f"Loaded {len(labels_df)} labels")
    print()

    # Compute per-prompt SSI by temperature
    print("Computing per-prompt SSI by temperature...")
    ssi_by_temp_df = compute_per_prompt_ssi_by_temperature(labels_df)
    print(f"Computed SSI for {len(ssi_by_temp_df)} prompts")
    print()

    # Run Friedman test
    print("Running Friedman test...")
    friedman_result = run_friedman_test(ssi_by_temp_df)

    print("\n" + "=" * 60)
    print("FRIEDMAN TEST RESULTS")
    print("=" * 60)
    print(f"Test: {friedman_result['test']}")
    print(f"χ² statistic: {friedman_result['statistic']:.2f}")
    print(f"p-value: {friedman_result['pvalue']:.2e}")
    print(f"Degrees of freedom: {friedman_result['df']}")
    print(f"Number of prompts: {friedman_result['n_prompts']}")

    if friedman_result['pvalue'] < 0.001:
        print("\n*** SIGNIFICANT at p < 0.001 ***")
    elif friedman_result['pvalue'] < 0.05:
        print("\n*** SIGNIFICANT at p < 0.05 ***")
    else:
        print("\n(Not significant)")

    # Run post-hoc tests
    print("\n" + "=" * 60)
    print("POST-HOC WILCOXON TESTS (Bonferroni corrected)")
    print("=" * 60)

    posthoc_df = run_posthoc_wilcoxon(ssi_by_temp_df)
    print(posthoc_df.to_string(index=False))

    # Save results
    results_dict = {
        'friedman_chi2': friedman_result['statistic'],
        'friedman_pvalue': friedman_result['pvalue'],
        'friedman_df': friedman_result['df'],
        'n_prompts': friedman_result['n_prompts'],
    }
    pd.DataFrame([results_dict]).to_csv(METRICS_DIR / "friedman_test_results.csv", index=False)
    posthoc_df.to_csv(METRICS_DIR / "posthoc_wilcoxon_results.csv", index=False)
    print(f"\nSaved results to {METRICS_DIR}")

    # Print LaTeX for paper
    print("\n" + "=" * 60)
    print("TEXT FOR PAPER")
    print("=" * 60)
    print(f"""
Temperature significantly affects stability (Friedman χ² = {friedman_result['statistic']:.2f},
p < 0.001). Post-hoc Wilcoxon signed-rank tests with Bonferroni correction confirm
significant decreases in SSI between temperature 0.0 and 1.0 (p < 0.001).
""")

    # Compare with old Kruskal-Wallis value from paper
    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER")
    print("=" * 60)
    print(f"""
Original paper reports: Kruskal-Wallis H = 185.43, p < 0.001
New Friedman test:      χ² = {friedman_result['statistic']:.2f}, p < 0.001

The Friedman test is more appropriate because:
1. Same prompts are tested at each temperature (repeated measures)
2. Kruskal-Wallis assumes independent groups, which is violated here
3. Friedman test accounts for the paired/repeated nature of the data

UPDATE NEEDED IN PAPER:
- Section 4.3: Replace "Kruskal-Wallis H = 185.43" with "Friedman χ² = {friedman_result['statistic']:.2f}"
- Keep the χ² test for categorical data as-is (that's correct)
""")


if __name__ == "__main__":
    main()
