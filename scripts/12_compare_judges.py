#!/usr/bin/env python3
"""
Compare Llama 70B judge vs GPT-4o judge.
Computes inter-judge agreement metrics for the paper.

Usage:
    python scripts/12_compare_judges.py \
        --llama data/results/labels/llama70b_labels.csv \
        --gpt4o data/results/labels/gpt4o_labels.csv \
        --output data/results/metrics/judge_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def compute_cohens_kappa(y1, y2):
    """Compute Cohen's kappa for inter-rater agreement."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2)


def compute_agreement_ci(y1, y2, n_bootstrap=1000, confidence=0.95):
    """Compute confidence interval for agreement using bootstrap."""
    agreements = []
    n = len(y1)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        agreement = (y1.iloc[indices].values == y2.iloc[indices].values).mean()
        agreements.append(agreement)
    
    lower = np.percentile(agreements, (1 - confidence) / 2 * 100)
    upper = np.percentile(agreements, (1 + confidence) / 2 * 100)
    
    return lower, upper


def compare_judges(llama_file: Path, gpt4o_file: Path, output_file: Path):
    """Compare two judge label files and compute agreement metrics."""
    
    print("Loading label files...")
    llama_df = pd.read_csv(llama_file)
    gpt4o_df = pd.read_csv(gpt4o_file)
    
    print(f"Llama labels: {len(llama_df)}")
    print(f"GPT-4o labels: {len(gpt4o_df)}")
    
    # Standardize column names
    llama_df = llama_df.rename(columns={"label": "label_llama"})
    gpt4o_df = gpt4o_df.rename(columns={"label": "label_gpt4o"})
    
    # Determine merge keys based on available columns
    merge_keys = []
    for key in ["model", "prompt_id", "temperature", "seed"]:
        if key in llama_df.columns and key in gpt4o_df.columns:
            merge_keys.append(key)
    
    if not merge_keys:
        print("Error: No common keys found for merging")
        return None
    
    print(f"Merging on: {merge_keys}")
    
    # Merge datasets
    merged = llama_df.merge(gpt4o_df, on=merge_keys, how="inner")
    print(f"Matched: {len(merged)} judgments")
    
    if len(merged) == 0:
        print("Error: No matching records found")
        return None
    
    # Filter out ERROR labels
    valid_mask = (merged["label_llama"] != "ERROR") & (merged["label_gpt4o"] != "ERROR")
    merged_valid = merged[valid_mask]
    print(f"Valid (non-ERROR): {len(merged_valid)}")
    
    # === Overall Agreement ===
    print("\n" + "="*60)
    print("INTER-JUDGE AGREEMENT")
    print("="*60)
    
    agreement = (merged_valid["label_llama"] == merged_valid["label_gpt4o"]).mean()
    kappa = compute_cohens_kappa(merged_valid["label_llama"], merged_valid["label_gpt4o"])
    
    # Bootstrap CI for agreement
    ci_lower, ci_upper = compute_agreement_ci(
        merged_valid["label_llama"], 
        merged_valid["label_gpt4o"]
    )
    
    print(f"\nRaw Agreement: {agreement*100:.1f}% [95% CI: {ci_lower*100:.1f}%-{ci_upper*100:.1f}%]")
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    # Interpret kappa
    if kappa > 0.8:
        kappa_interp = "almost perfect"
    elif kappa > 0.6:
        kappa_interp = "substantial"
    elif kappa > 0.4:
        kappa_interp = "moderate"
    elif kappa > 0.2:
        kappa_interp = "fair"
    else:
        kappa_interp = "slight"
    print(f"Kappa interpretation: {kappa_interp} agreement")
    
    # === Per-Class Agreement ===
    print("\n" + "-"*60)
    print("PER-CLASS AGREEMENT")
    print("-"*60)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    labels = ["REFUSE", "PARTIAL", "COMPLY"]
    print("\nClassification Report (Llama as reference):")
    print(classification_report(
        merged_valid["label_llama"], 
        merged_valid["label_gpt4o"],
        labels=labels,
        zero_division=0
    ))
    
    # === Confusion Matrix ===
    print("\nConfusion Matrix:")
    print("(Rows: Llama 70B, Columns: GPT-4o)")
    cm = confusion_matrix(
        merged_valid["label_llama"],
        merged_valid["label_gpt4o"],
        labels=labels
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"Llama:{l}" for l in labels],
        columns=[f"GPT4o:{l}" for l in labels]
    )
    print(cm_df)
    
    # === Per-Class Agreement Rates ===
    print("\n" + "-"*60)
    print("PER-CLASS AGREEMENT RATES")
    print("-"*60)
    
    for label in labels:
        mask = merged_valid["label_llama"] == label
        if mask.sum() > 0:
            class_agreement = (
                merged_valid.loc[mask, "label_llama"] == 
                merged_valid.loc[mask, "label_gpt4o"]
            ).mean()
            print(f"  {label}: {class_agreement*100:.1f}% ({mask.sum()} cases)")
    
    # === Stability Metrics with Each Judge ===
    print("\n" + "="*60)
    print("STABILITY METRICS BY JUDGE")
    print("="*60)
    
    def compute_stability_metrics(df, label_col, judge_name):
        """Compute SSI and instability rate from labels."""
        from collections import Counter
        
        # Group by model and prompt_id
        group_keys = ["prompt_id"]
        if "model" in df.columns:
            group_keys = ["model", "prompt_id"]
        
        stability_data = []
        for keys, group in df.groupby(group_keys):
            counts = Counter(group[label_col])
            total = len(group)
            if total > 0:
                ssi = max(counts.values()) / total
                stability_data.append({
                    "keys": keys,
                    "ssi": ssi,
                    "unstable": ssi < 0.8,
                    "n_samples": total
                })
        
        stability_df = pd.DataFrame(stability_data)
        mean_ssi = stability_df["ssi"].mean()
        pct_unstable = stability_df["unstable"].mean() * 100
        
        print(f"\n{judge_name}:")
        print(f"  Mean SSI: {mean_ssi:.3f}")
        print(f"  % Unstable (SSI < 0.8): {pct_unstable:.1f}%")
        print(f"  N prompts: {len(stability_df)}")
        
        return mean_ssi, pct_unstable
    
    llama_ssi, llama_unstable = compute_stability_metrics(
        merged_valid, "label_llama", "Llama 70B Judge"
    )
    gpt4o_ssi, gpt4o_unstable = compute_stability_metrics(
        merged_valid, "label_gpt4o", "GPT-4o Judge"
    )
    
    # === Disagreement Analysis ===
    print("\n" + "="*60)
    print("DISAGREEMENT PATTERNS")
    print("="*60)
    
    disagreements = merged_valid[merged_valid["label_llama"] != merged_valid["label_gpt4o"]]
    print(f"\nTotal disagreements: {len(disagreements)} ({len(disagreements)/len(merged_valid)*100:.1f}%)")
    
    # Most common disagreement patterns
    print("\nDisagreement patterns:")
    pattern_counts = disagreements.groupby(["label_llama", "label_gpt4o"]).size().sort_values(ascending=False)
    for (llama_label, gpt4o_label), count in pattern_counts.items():
        pct = count / len(merged_valid) * 100
        print(f"  Llama:{llama_label} → GPT4o:{gpt4o_label}: {count} ({pct:.1f}%)")
    
    # === Save Results ===
    results = {
        "n_matched": int(len(merged_valid)),
        "agreement_pct": float(agreement * 100),
        "agreement_ci_lower": float(ci_lower * 100),
        "agreement_ci_upper": float(ci_upper * 100),
        "cohens_kappa": float(kappa),
        "kappa_interpretation": kappa_interp,
        "llama_mean_ssi": float(llama_ssi),
        "llama_pct_unstable": float(llama_unstable),
        "gpt4o_mean_ssi": float(gpt4o_ssi),
        "gpt4o_pct_unstable": float(gpt4o_unstable),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_disagreements": int(len(disagreements)),
        "disagreement_pct": float(len(disagreements) / len(merged_valid) * 100)
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)
    
    # === LaTeX snippets for paper ===
    print("\n" + "="*60)
    print("LATEX SNIPPETS FOR PAPER")
    print("="*60)
    
    print(f"""
Inter-judge agreement between Llama 3.1 70B and GPT-4o was {agreement*100:.1f}\\% 
[95\\% CI: {ci_lower*100:.1f}\\%--{ci_upper*100:.1f}\\%] with Cohen's $\\kappa$ = {kappa:.2f}, 
indicating {kappa_interp} agreement.

Core findings remain robust across judges:
\\begin{{itemize}}
    \\item With GPT-4o labels: {gpt4o_unstable:.1f}\\% of prompts are unstable (SSI < 0.8), 
          compared to {llama_unstable:.1f}\\% with Llama 70B
    \\item Mean SSI: {gpt4o_ssi:.3f} (GPT-4o) vs {llama_ssi:.3f} (Llama 70B)
\\end{{itemize}}
""")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare judge labels")
    parser.add_argument("--llama", type=Path, required=True,
                        help="Llama 70B labels CSV")
    parser.add_argument("--gpt4o", type=Path, required=True,
                        help="GPT-4o labels CSV")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data/results/metrics/judge_comparison.json",
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    compare_judges(args.llama, args.gpt4o, args.output)


if __name__ == "__main__":
    main()
