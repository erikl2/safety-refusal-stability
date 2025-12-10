"""
Script to prepare demo data from experiment results.

Extracts a curated subset of prompts for the Safety Stability Explorer demo.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
METRICS_FILE = PROJECT_ROOT / "data" / "results" / "metrics" / "per_prompt_metrics.csv"
PROMPTS_FILE = PROJECT_ROOT / "data" / "processed" / "prompts.csv"
OUTPUT_FILE = Path(__file__).parent / "data" / "prompts.json"

# Config
TEMPS = [0.0, 0.3, 0.7, 1.0]
SEEDS = [42, 43, 44, 45, 46]
MODEL_PREFIX = "meta_llama_Llama_3.1_8B_Instruct"


def load_all_labels():
    """Load all Llama labels into a single DataFrame."""
    dfs = []
    for temp in TEMPS:
        for seed in SEEDS:
            filename = f"{MODEL_PREFIX}_temp{temp}_seed{seed}_labels.csv"
            filepath = LABELS_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                dfs.append(df)
            else:
                print(f"Warning: {filepath} not found")

    if not dfs:
        raise FileNotFoundError("No label files found")

    return pd.concat(dfs, ignore_index=True)


def categorize_prompt(prompt_text):
    """Simple categorization based on keywords."""
    prompt_lower = prompt_text.lower()

    categories = {
        "violence": ["kill", "murder", "assault", "attack", "harm", "hurt", "weapon", "bomb", "explosive", "gun", "shoot"],
        "hacking": ["hack", "exploit", "vulnerability", "malware", "virus", "ddos", "sql injection", "phishing", "ransomware"],
        "fraud": ["fraud", "scam", "steal", "identity theft", "credit card", "fake", "counterfeit"],
        "illegal_drugs": ["drug", "cocaine", "heroin", "meth", "synthesize", "manufacture"],
        "harassment": ["harass", "bully", "stalk", "threaten", "intimidate"],
        "misinformation": ["fake news", "misinformation", "conspiracy", "false information"],
        "hate_speech": ["racist", "sexist", "discrimination", "hate", "minority"],
        "self_harm": ["suicide", "self-harm", "eating disorder"],
        "privacy": ["private", "personal information", "surveillance", "spy"],
        "copyright": ["lyrics", "generate the", "passage from", "book"],
    }

    for category, keywords in categories.items():
        if any(kw in prompt_lower for kw in keywords):
            return category

    return "other"


def select_prompts(metrics_df, labels_df, num_prompts=40):
    """
    Select a curated subset of prompts.

    Selection criteria:
    - 10-15 stable refusers (SSI > 0.95, mostly REFUSE)
    - 10-15 unstable borderline (SSI 0.4-0.7, mixed decisions)
    - 5-10 stable compliers (SSI > 0.95, mostly COMPLY or PARTIAL)
    - 5-10 interesting edge cases (SSI 0.7-0.9)

    Also filter for readability (not too long).
    """
    # Filter for Llama model prompts (20 samples each)
    llama_metrics = metrics_df[metrics_df["num_samples"] == 20].copy()

    # Filter for reasonable length prompts
    llama_metrics["prompt_length"] = llama_metrics["prompt"].str.len()
    llama_metrics = llama_metrics[llama_metrics["prompt_length"] < 200]

    selected = []

    # 1. Stable refusers (SSI > 0.95, majority REFUSE)
    stable_refuse = llama_metrics[
        (llama_metrics["stability_index"] > 0.95) &
        (llama_metrics["majority_label"] == "REFUSE")
    ].sample(n=min(12, len(llama_metrics[(llama_metrics["stability_index"] > 0.95) & (llama_metrics["majority_label"] == "REFUSE")])), random_state=42)
    selected.append(stable_refuse)
    print(f"Selected {len(stable_refuse)} stable refusers")

    # 2. Unstable borderline (SSI 0.3-0.7)
    unstable = llama_metrics[
        (llama_metrics["stability_index"] >= 0.1) &
        (llama_metrics["stability_index"] <= 0.7) &
        (llama_metrics["flip_occurred"] == True)
    ].sample(n=min(15, len(llama_metrics[(llama_metrics["stability_index"] >= 0.1) & (llama_metrics["stability_index"] <= 0.7)])), random_state=42)
    selected.append(unstable)
    print(f"Selected {len(unstable)} unstable prompts")

    # 3. Stable compliers (SSI > 0.9, majority COMPLY or high comply rate)
    stable_comply = llama_metrics[
        (llama_metrics["stability_index"] > 0.5) &
        (llama_metrics["comply_rate"] > 0.3)
    ].sample(n=min(8, len(llama_metrics[(llama_metrics["stability_index"] > 0.5) & (llama_metrics["comply_rate"] > 0.3)])), random_state=42)
    selected.append(stable_comply)
    print(f"Selected {len(stable_comply)} stable compliers / high comply rate")

    # 4. Edge cases (SSI 0.7-0.95, flip occurred)
    edge_cases = llama_metrics[
        (llama_metrics["stability_index"] > 0.7) &
        (llama_metrics["stability_index"] <= 0.95) &
        (llama_metrics["flip_occurred"] == True)
    ].sample(n=min(8, len(llama_metrics[(llama_metrics["stability_index"] > 0.7) & (llama_metrics["stability_index"] <= 0.95)])), random_state=42)
    selected.append(edge_cases)
    print(f"Selected {len(edge_cases)} edge cases")

    # Combine and deduplicate
    selected_df = pd.concat(selected).drop_duplicates(subset=["prompt_id"])
    print(f"\nTotal selected: {len(selected_df)} prompts")

    return selected_df


def build_prompt_data(prompt_id, prompt_text, ssi, labels_df):
    """Build the response data for a single prompt."""
    prompt_labels = labels_df[labels_df["prompt_id"] == prompt_id]

    responses = []
    for temp in TEMPS:
        for seed in SEEDS:
            row = prompt_labels[
                (prompt_labels["temperature"] == temp) &
                (prompt_labels["seed"] == seed)
            ]
            if len(row) > 0:
                row = row.iloc[0]
                responses.append({
                    "temp": temp,
                    "seed": seed,
                    "label": row["label"],
                    "text": row["response"][:1000] if pd.notna(row["response"]) else ""  # Truncate long responses
                })

    return {
        "id": prompt_id,
        "text": prompt_text,
        "category": categorize_prompt(prompt_text),
        "ssi": round(ssi, 3),
        "responses": responses
    }


def main():
    print("Loading data...")
    metrics_df = pd.read_csv(METRICS_FILE)
    labels_df = load_all_labels()

    print(f"Loaded {len(metrics_df)} prompts with metrics")
    print(f"Loaded {len(labels_df)} labeled responses")

    print("\nSelecting prompts...")
    selected = select_prompts(metrics_df, labels_df)

    print("\nBuilding demo data...")
    prompts = []
    for _, row in selected.iterrows():
        prompt_data = build_prompt_data(
            row["prompt_id"],
            row["prompt"],
            row["stability_index"],
            labels_df
        )
        if len(prompt_data["responses"]) == 20:  # Only include complete data
            prompts.append(prompt_data)

    # Sort by SSI ascending (most unstable first)
    prompts.sort(key=lambda x: x["ssi"])

    # Build output
    output = {
        "prompts": prompts,
        "metadata": {
            "model": "Llama-3.1-8B-Instruct",
            "temps": TEMPS,
            "seeds": SEEDS,
            "total_prompts": len(metrics_df),
            "subset_description": "Curated subset showing range of stability scores"
        }
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {OUTPUT_FILE}")

    # Print summary
    print("\nSSI distribution in selected prompts:")
    ssi_values = [p["ssi"] for p in prompts]
    print(f"  Min: {min(ssi_values):.3f}")
    print(f"  Max: {max(ssi_values):.3f}")
    print(f"  Mean: {sum(ssi_values)/len(ssi_values):.3f}")

    print("\nCategory distribution:")
    categories = defaultdict(int)
    for p in prompts:
        categories[p["category"]] += 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
