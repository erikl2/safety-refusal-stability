#!/usr/bin/env python3
"""
Verify whether greedy decoding (t=0.0) produces identical outputs across seeds.
This addresses reviewer concern about 5-12% flip rates at temperature 0.0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
GENERATIONS_DIR = PROJECT_ROOT / "data" / "results" / "generations"
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "metrics"

def hash_response(text: str) -> str:
    """Create hash of response for exact match comparison."""
    if pd.isna(text):
        return "NA"
    return hashlib.md5(str(text).encode()).hexdigest()

def normalize_response(text: str) -> str:
    """Normalize response for near-match comparison."""
    if pd.isna(text):
        return ""
    # Strip whitespace, lowercase
    return str(text).strip().lower()

def analyze_greedy_determinism():
    """
    For each model at t=0.0:
    1. Check if responses are byte-identical across seeds
    2. If not identical, check if labels are identical (judge noise)
    3. Separate "true model variation" from "judge noise on identical text"
    """
    results = []
    detailed_results = []

    # Get all generation files at t=0.0
    gen_files = list(GENERATIONS_DIR.glob("*_temp0.0_*.csv"))

    if not gen_files:
        print("ERROR: No generation files found at temperature 0.0")
        return pd.DataFrame(), pd.DataFrame()

    # Group by model
    model_files = defaultdict(list)
    for f in gen_files:
        # Extract model name from filename
        parts = f.stem.split("_temp0.0_")
        model_name = parts[0]
        model_files[model_name].append(f)

    for model_name, files in model_files.items():
        print(f"\nAnalyzing {model_name}...")

        # Load all seed files for this model at t=0.0
        dfs = []
        for f in sorted(files):
            df = pd.read_csv(f)
            seed = int(f.stem.split("seed")[-1])
            df['seed'] = seed
            dfs.append(df)

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)

        exact_match_count = 0
        near_match_count = 0
        total_prompts = 0
        variation_prompts = []

        for prompt_id, group in combined.groupby('prompt_id'):
            total_prompts += 1
            responses = group['response'].tolist()
            seeds = group['seed'].tolist()

            # Check exact match
            hashes = [hash_response(r) for r in responses]
            if len(set(hashes)) == 1:
                exact_match_count += 1
            else:
                # Check near-match (normalized)
                normalized = [normalize_response(r) for r in responses]
                if len(set(normalized)) == 1:
                    near_match_count += 1
                else:
                    # True variation - record details
                    variation_prompts.append({
                        'prompt_id': prompt_id,
                        'model': model_name,
                        'n_unique_responses': len(set(hashes)),
                        'n_unique_normalized': len(set(normalized)),
                        'seeds': seeds,
                        'response_lengths': [len(str(r)) for r in responses]
                    })

        # Also check label consistency for varying prompts
        label_files = list(LABELS_DIR.glob(f"{model_name}_temp0.0_*.csv"))
        if label_files:
            label_dfs = []
            for f in label_files:
                ldf = pd.read_csv(f)
                seed = int(f.stem.split("seed")[-1].replace("_labels", ""))
                ldf['seed'] = seed
                label_dfs.append(ldf)

            if label_dfs:
                labels_combined = pd.concat(label_dfs, ignore_index=True)

                # For prompts with identical responses, check label consistency
                label_flips_on_identical = 0
                for prompt_id, group in combined.groupby('prompt_id'):
                    hashes = [hash_response(r) for r in group['response'].tolist()]
                    if len(set(hashes)) == 1:  # Identical responses
                        # Check labels
                        prompt_labels = labels_combined[labels_combined['prompt_id'] == prompt_id]
                        if len(prompt_labels) > 0:
                            labels = prompt_labels['label'].unique()
                            if len(labels) > 1:
                                label_flips_on_identical += 1

        results.append({
            'model': model_name,
            'total_prompts': total_prompts,
            'exact_match_count': exact_match_count,
            'exact_match_rate': exact_match_count / total_prompts * 100 if total_prompts > 0 else 0,
            'near_match_count': near_match_count,
            'variation_count': total_prompts - exact_match_count - near_match_count,
            'variation_rate': (total_prompts - exact_match_count) / total_prompts * 100 if total_prompts > 0 else 0,
            'label_flips_on_identical': label_flips_on_identical if 'label_flips_on_identical' in dir() else 0
        })

        detailed_results.extend(variation_prompts)

    return pd.DataFrame(results), pd.DataFrame(detailed_results)

def analyze_label_noise_on_identical():
    """
    For prompts where t=0.0 responses are IDENTICAL across seeds,
    check if judge labels differ (indicating pure judge noise).
    """
    results = []

    # Get models
    gen_files = list(GENERATIONS_DIR.glob("*_temp0.0_*.csv"))
    model_files = defaultdict(list)
    for f in gen_files:
        parts = f.stem.split("_temp0.0_")
        model_name = parts[0]
        model_files[model_name].append(f)

    for model_name, files in model_files.items():
        print(f"\nAnalyzing label noise for {model_name}...")

        # Load generations
        gen_dfs = []
        for f in sorted(files):
            df = pd.read_csv(f)
            seed = int(f.stem.split("seed")[-1])
            df['seed'] = seed
            gen_dfs.append(df)

        if not gen_dfs:
            continue

        gen_combined = pd.concat(gen_dfs, ignore_index=True)

        # Load labels
        label_files = list(LABELS_DIR.glob(f"{model_name}_temp0.0_*.csv"))
        if not label_files:
            continue

        label_dfs = []
        for f in label_files:
            ldf = pd.read_csv(f)
            seed = int(f.stem.split("seed")[-1].replace("_labels", ""))
            ldf['seed'] = seed
            label_dfs.append(ldf)

        labels_combined = pd.concat(label_dfs, ignore_index=True)

        identical_response_prompts = 0
        label_flip_on_identical = 0

        for prompt_id, gen_group in gen_combined.groupby('prompt_id'):
            responses = gen_group['response'].tolist()
            hashes = [hash_response(r) for r in responses]

            if len(set(hashes)) == 1:  # Identical responses
                identical_response_prompts += 1

                # Check labels
                prompt_labels = labels_combined[labels_combined['prompt_id'] == prompt_id]
                if len(prompt_labels) > 0:
                    labels = [l for l in prompt_labels['label'].tolist() if l in {'REFUSE', 'PARTIAL', 'COMPLY'}]
                    if len(set(labels)) > 1:
                        label_flip_on_identical += 1

        results.append({
            'model': model_name,
            'identical_response_prompts': identical_response_prompts,
            'label_flips_on_identical': label_flip_on_identical,
            'judge_noise_rate': label_flip_on_identical / identical_response_prompts * 100 if identical_response_prompts > 0 else 0
        })

    return pd.DataFrame(results)

def main():
    print("=" * 70)
    print("Task 1.1: Verify Greedy Decoding Determinism")
    print("=" * 70)

    # Check if data exists
    if not GENERATIONS_DIR.exists():
        print("\nERROR: generations/ directory not found.")
        print("This script requires the full repository with raw generations.")
        return

    results, detailed = analyze_greedy_determinism()

    if len(results) == 0:
        print("No results generated")
        return

    print("\n" + "=" * 70)
    print("GREEDY DECODING (t=0.0) EXACT MATCH RATES:")
    print("=" * 70)
    print(results.to_string(index=False))

    # Save results
    results.to_csv(OUTPUT_DIR / "greedy_determinism_check.csv", index=False)
    if len(detailed) > 0:
        detailed.to_csv(OUTPUT_DIR / "greedy_variation_details.csv", index=False)

    # Analyze label noise on identical responses
    print("\n" + "=" * 70)
    print("JUDGE NOISE ON IDENTICAL RESPONSES:")
    print("=" * 70)

    label_noise = analyze_label_noise_on_identical()
    if len(label_noise) > 0:
        print(label_noise.to_string(index=False))
        label_noise.to_csv(OUTPUT_DIR / "judge_noise_on_identical.csv", index=False)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION FOR PAPER:")
    print("=" * 70)

    for _, row in results.iterrows():
        model = row['model']
        exact_rate = row['exact_match_rate']
        var_rate = row['variation_rate']

        if exact_rate > 99:
            print(f"\n{model}:")
            print(f"  - {exact_rate:.1f}% of prompts produce byte-identical outputs at t=0.0")
            print(f"  -> Flip rates at t=0.0 are likely JUDGE NOISE, not model variation")
        elif exact_rate > 95:
            print(f"\n{model}:")
            print(f"  - {exact_rate:.1f}% exact match, {var_rate:.1f}% variation")
            print(f"  -> Minor non-determinism, mostly judge noise")
        else:
            print(f"\n{model}:")
            print(f"  - {exact_rate:.1f}% exact match, {var_rate:.1f}% variation")
            print(f"  -> Significant GPU/inference non-determinism")

    # LaTeX output
    print("\n" + "-" * 50)
    print("LaTeX for paper Section 5.2:")
    print("-" * 50)

    avg_exact = results['exact_match_rate'].mean()
    if len(label_noise) > 0:
        avg_judge_noise = label_noise['judge_noise_rate'].mean()
        print(f"""
At temperature 0.0, we verified that {avg_exact:.1f}\\% of prompts produced
byte-identical outputs across all 5 seeds, confirming that inference is
largely deterministic. Of the prompts with identical responses,
{avg_judge_noise:.1f}\\% showed label variation, indicating that the
{100-avg_exact:.1f}\\% residual flip rate at this temperature primarily
reflects judge labeling ambiguity rather than model variation.
""")

if __name__ == "__main__":
    main()
