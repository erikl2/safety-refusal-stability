#!/usr/bin/env python3
"""
Sensitivity Analysis and Judge Confusion Matrix

Task 6: Compute % unstable at different SSI thresholds
Task 7: Compute confusion matrix between Llama 70B and Claude Haiku judges
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
METRICS_DIR = PROJECT_ROOT / "data" / "results" / "metrics"


def compute_stability_index(labels):
    """Compute SSI for a list of labels."""
    valid_labels = [l for l in labels if l in {"REFUSE", "PARTIAL", "COMPLY"}]
    if len(valid_labels) == 0:
        return np.nan
    counts = Counter(valid_labels)
    return max(counts.values()) / len(valid_labels)


def load_combined_labels():
    """Load combined label data for all 4 models (Claude Haiku judge)."""
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


def compute_per_prompt_ssi(labels_df):
    """Compute SSI for each prompt (all configs)."""
    results = []
    for (model, prompt_id), group in labels_df.groupby(['model', 'prompt_id']):
        labels = group['label'].tolist()
        ssi = compute_stability_index(labels)
        results.append({
            'model': model,
            'prompt_id': prompt_id,
            'ssi': ssi,
        })
    return pd.DataFrame(results)


def sensitivity_analysis(prompt_ssi_df, thresholds=[0.7, 0.75, 0.8, 0.85, 0.9]):
    """
    Compute % unstable at different thresholds for each model.
    """
    results = []

    for model in prompt_ssi_df['model'].unique():
        model_df = prompt_ssi_df[prompt_ssi_df['model'] == model]
        short_name = model.split('/')[-1] if '/' in model else model
        short_name = short_name.replace('_', ' ')

        for threshold in thresholds:
            pct_unstable = (model_df['ssi'] < threshold).mean() * 100
            results.append({
                'model': short_name,
                'threshold': threshold,
                'pct_unstable': pct_unstable,
            })

    return pd.DataFrame(results)


def load_llama_judge_labels():
    """Load labels from Llama 70B judge (original labels)."""
    # These are the original _labels.csv files
    label_files = list(LABELS_DIR.glob("meta_llama_Llama_3.1_8B*_labels.csv"))
    label_files += list(LABELS_DIR.glob("Qwen_Qwen2.5_7B*_labels.csv"))

    if not label_files:
        print("No Llama judge label files found")
        return None

    dfs = []
    for f in label_files:
        df = pd.read_csv(f)
        # Extract metadata from filename
        parts = f.stem.replace('_labels', '').split('_temp')
        if len(parts) >= 2:
            model = parts[0].replace('_', '/')
            temp_seed = parts[1].split('_seed')
            temp = float(temp_seed[0])
            seed = int(temp_seed[1]) if len(temp_seed) > 1 else 42
            df['model'] = model
            df['temperature'] = temp
            df['seed'] = seed
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def compute_confusion_matrix(claude_df, llama_df):
    """
    Compute confusion matrix between Claude Haiku and Llama 70B judges.

    Note: This only works for the models that were judged by both
    (Llama 3.1 8B and Qwen 2.5 7B).
    """
    # Filter to only models that appear in both
    claude_models = set(claude_df['model'].unique())
    llama_models = set(llama_df['model'].unique())

    # Normalize model names for comparison
    def normalize_model(m):
        return m.replace('/', '_').replace('-', '_').lower()

    # Create lookup from Claude Haiku labels
    claude_lookup = {}
    for _, row in claude_df.iterrows():
        key = (normalize_model(row['model']), row['prompt_id'], row['temperature'], row['seed'])
        claude_lookup[key] = row['label']

    # Match with Llama labels
    matches = []
    for _, row in llama_df.iterrows():
        key = (normalize_model(row['model']), row['prompt_id'], row['temperature'], row['seed'])
        if key in claude_lookup:
            matches.append({
                'llama_label': row['label'],
                'claude_label': claude_lookup[key],
            })

    if not matches:
        print("No matching labels found between judges")
        return None

    matches_df = pd.DataFrame(matches)
    print(f"Found {len(matches_df)} matching labels between judges")

    # Compute confusion matrix
    labels = ['REFUSE', 'PARTIAL', 'COMPLY']
    confusion = pd.crosstab(
        matches_df['llama_label'],
        matches_df['claude_label'],
        rownames=['Llama 70B'],
        colnames=['Claude Haiku'],
    )

    # Ensure all labels are present
    for label in labels:
        if label not in confusion.index:
            confusion.loc[label] = 0
        if label not in confusion.columns:
            confusion[label] = 0

    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)

    # Compute agreement
    total = len(matches_df)
    agree = (matches_df['llama_label'] == matches_df['claude_label']).sum()
    agreement_pct = agree / total * 100

    # Compute Cohen's kappa
    # Expected agreement by chance
    llama_dist = matches_df['llama_label'].value_counts(normalize=True)
    claude_dist = matches_df['claude_label'].value_counts(normalize=True)

    expected_agree = sum(
        llama_dist.get(label, 0) * claude_dist.get(label, 0)
        for label in labels
    )

    kappa = (agreement_pct/100 - expected_agree) / (1 - expected_agree) if expected_agree < 1 else 0

    return {
        'confusion_matrix': confusion,
        'total_samples': total,
        'agreement_count': agree,
        'agreement_pct': agreement_pct,
        'cohens_kappa': kappa,
    }


def main():
    print("=" * 60)
    print("Sensitivity Analysis and Judge Confusion Matrix")
    print("=" * 60)
    print()

    # TASK 6: Sensitivity Analysis
    print("=" * 60)
    print("TASK 6: SSI Threshold Sensitivity Analysis")
    print("=" * 60)
    print()

    print("Loading Claude Haiku labels...")
    claude_df = load_combined_labels()
    print(f"Loaded {len(claude_df)} labels")

    print("\nComputing per-prompt SSI...")
    prompt_ssi_df = compute_per_prompt_ssi(claude_df)

    print("\nComputing sensitivity analysis...")
    sensitivity_df = sensitivity_analysis(prompt_ssi_df)

    # Pivot for display
    sensitivity_pivot = sensitivity_df.pivot(index='model', columns='threshold', values='pct_unstable')
    print("\n% Unstable by Model and Threshold:")
    print(sensitivity_pivot.round(1).to_string())

    # Save
    sensitivity_df.to_csv(METRICS_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\nSaved: {METRICS_DIR / 'threshold_sensitivity.csv'}")

    # Print LaTeX table
    print("\n" + "-" * 60)
    print("LaTeX Table for Paper:")
    print("-" * 60)
    print(r"""
\begin{table}[h]
\centering
\caption{Sensitivity analysis: \% of prompts classified as unstable at different SSI thresholds.}
\label{tab:threshold_sensitivity}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{0.70} & \textbf{0.75} & \textbf{0.80} & \textbf{0.85} & \textbf{0.90} \\
\midrule""")
    for model in sensitivity_pivot.index:
        row = sensitivity_pivot.loc[model]
        print(f"{model} & {row[0.70]:.1f}\\% & {row[0.75]:.1f}\\% & {row[0.80]:.1f}\\% & {row[0.85]:.1f}\\% & {row[0.90]:.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # TASK 7: Judge Confusion Matrix
    print()
    print("=" * 60)
    print("TASK 7: Judge Confusion Matrix")
    print("=" * 60)
    print()

    print("Loading Llama 70B judge labels...")
    llama_df = load_llama_judge_labels()

    if llama_df is not None and len(llama_df) > 0:
        print(f"Loaded {len(llama_df)} Llama judge labels")

        print("\nComputing confusion matrix...")
        result = compute_confusion_matrix(claude_df, llama_df)

        if result:
            print("\nConfusion Matrix:")
            print(result['confusion_matrix'].to_string())
            print(f"\nTotal samples compared: {result['total_samples']}")
            print(f"Agreement: {result['agreement_count']} ({result['agreement_pct']:.1f}%)")
            print(f"Cohen's kappa: {result['cohens_kappa']:.3f}")

            # Save
            result['confusion_matrix'].to_csv(METRICS_DIR / "judge_confusion_matrix.csv")
            print(f"\nSaved: {METRICS_DIR / 'judge_confusion_matrix.csv'}")

            # Print LaTeX table
            print("\n" + "-" * 60)
            print("LaTeX Table for Paper:")
            print("-" * 60)
            cm = result['confusion_matrix']
            print(r"""
\begin{table}[h]
\centering
\caption{Confusion matrix comparing Llama 70B and Claude Haiku judges. Overall agreement: """ + f"{result['agreement_pct']:.1f}" + r"""\%, Cohen's $\kappa$ = """ + f"{result['cohens_kappa']:.3f}" + r""".}
\label{tab:confusion_matrix}
\begin{tabular}{lccc|c}
\toprule
& \multicolumn{3}{c}{\textbf{Claude Haiku}} & \\
\textbf{Llama 70B} & REFUSE & PARTIAL & COMPLY & Total \\
\midrule""")
            for label in ['REFUSE', 'PARTIAL', 'COMPLY']:
                row = cm.loc[label]
                total = row.sum()
                print(f"{label} & {row['REFUSE']} & {row['PARTIAL']} & {row['COMPLY']} & {total} \\\\")
            print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    else:
        print("Could not compute confusion matrix - Llama judge labels not available")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
