"""
Statistical tests for safety refusal stability analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def compute_significance_tests(
    group1_metrics: pd.DataFrame,
    group2_metrics: pd.DataFrame,
    metric_col: str = "stability_index",
) -> dict:
    """
    Compute statistical significance tests between two groups.

    Args:
        group1_metrics: First group of metrics
        group2_metrics: Second group of metrics
        metric_col: Column to compare

    Returns:
        Dict with test results
    """
    vals1 = group1_metrics[metric_col].dropna()
    vals2 = group2_metrics[metric_col].dropna()

    # t-test
    t_stat, t_pval = stats.ttest_ind(vals1, vals2)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
    cohens_d = (vals1.mean() - vals2.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "u_statistic": u_stat,
        "u_pvalue": u_pval,
        "cohens_d": cohens_d,
        "group1_mean": vals1.mean(),
        "group2_mean": vals2.mean(),
        "group1_std": vals1.std(),
        "group2_std": vals2.std(),
    }


def compute_correlation(
    metrics_df: pd.DataFrame,
    col1: str,
    col2: str,
) -> dict:
    """
    Compute correlation between two metric columns.

    Args:
        metrics_df: DataFrame with metrics
        col1: First column name
        col2: Second column name

    Returns:
        Dict with correlation results
    """
    vals1 = metrics_df[col1].dropna()
    vals2 = metrics_df[col2].dropna()

    # Align indices
    common_idx = vals1.index.intersection(vals2.index)
    vals1 = vals1.loc[common_idx]
    vals2 = vals2.loc[common_idx]

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(vals1, vals2)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(vals1, vals2)

    return {
        "pearson_r": pearson_r,
        "pearson_pvalue": pearson_p,
        "spearman_r": spearman_r,
        "spearman_pvalue": spearman_p,
        "n_samples": len(common_idx),
    }
