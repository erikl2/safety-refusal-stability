#!/usr/bin/env python3
"""
Compute bootstrap 95% CI for the logistic regression AUC predicting instability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import re

PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
PROMPTS_FILE = PROJECT_ROOT / "data" / "processed" / "prompts.csv"

SENSITIVE_TERMS = {'hack', 'bomb', 'kill', 'illegal', 'steal', 'weapon', 'drug', 'attack'}


def extract_features(text):
    words = text.lower().split()
    return {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': sum(len(w) for w in words) / max(1, len(words)),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'has_question': 1 if '?' in text else 0,
        'sensitive_term_count': sum(1 for w in words if w in SENSITIVE_TERMS),
        'has_code_markers': 1 if any(m in text.lower() for m in ['```', 'def ', 'function', 'import']) else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text))
    }


def main():
    print("=" * 70)
    print("Computing Bootstrap 95% CI for AUC")
    print("=" * 70)

    # Load labels
    llama_files = list(LABELS_DIR.glob("meta_llama_Llama_3.1_8B_Instruct_temp*_seed*_labels.csv"))
    if not llama_files:
        print("No Llama label files found")
        return

    dfs = [pd.read_csv(f) for f in llama_files]
    labels_df = pd.concat(dfs, ignore_index=True)

    # Compute per-prompt SSI
    prompt_ssi = {}
    for pid, group in labels_df.groupby('prompt_id'):
        labs = [l for l in group['label'].tolist() if l in {'REFUSE', 'PARTIAL', 'COMPLY'}]
        if labs:
            counts = Counter(labs)
            ssi = max(counts.values()) / len(labs)
            prompt_ssi[pid] = {'ssi': ssi, 'unstable': ssi < 0.8}

    # Load prompts
    prompts_df = pd.read_csv(PROMPTS_FILE)

    # Extract features
    features_list = []
    labels_list = []
    feature_names = []

    for _, row in prompts_df.iterrows():
        pid = row['id']
        if pid not in prompt_ssi:
            continue
        feats = extract_features(row['prompt'])
        if not feature_names:
            feature_names = list(feats.keys())
        features_list.append([feats[k] for k in feature_names])
        labels_list.append(1 if prompt_ssi[pid]['unstable'] else 0)

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"Dataset: {len(X)} prompts, {sum(y)} unstable ({sum(y)/len(y)*100:.1f}%)")
    print(f"Features: {feature_names}")

    # Point estimate with CV
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    point_auc = roc_auc_score(y, y_pred_proba)
    print(f"\nPoint estimate AUC: {point_auc:.3f}")

    # Bootstrap CI
    n_bootstrap = 1000
    np.random.seed(42)
    auc_boots = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        try:
            clf_boot = LogisticRegression(max_iter=1000, random_state=42)
            cv_boot = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_boot = cross_val_predict(clf_boot, X_boot, y_boot, cv=cv_boot, method='predict_proba')[:, 1]
            auc = roc_auc_score(y_boot, y_pred_boot)
            auc_boots.append(auc)
        except Exception:
            continue

    auc_boots = np.array(auc_boots)
    ci_lower = np.percentile(auc_boots, 2.5)
    ci_upper = np.percentile(auc_boots, 97.5)

    print(f"\nAUC: {point_auc:.2f} [95% CI: {ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"Bootstrap samples: {len(auc_boots)}")

    # For paper: formatted output
    print("\n" + "=" * 70)
    print("For paper text:")
    print(f"AUC = {point_auc:.2f} [95% CI: {ci_lower:.2f}--{ci_upper:.2f}]")
    print(f"Features: {', '.join(feature_names)}")


if __name__ == "__main__":
    main()
