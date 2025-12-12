"""
Script to prepare demo data from experiment results.

Extracts a curated subset of prompts for the Safety Stability Explorer demo.
Supports multiple models with Claude 3.5 Haiku judge labels.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "results" / "labels"
RESPONSES_DIR = PROJECT_ROOT / "data" / "results" / "responses"
GENERATIONS_DIR = PROJECT_ROOT / "data" / "results" / "generations"
PROMPTS_FILE = PROJECT_ROOT / "data" / "processed" / "prompts.csv"
OUTPUT_FILE = Path(__file__).parent / "data" / "prompts.json"

# Config
TEMPS = [0.0, 0.3, 0.7, 1.0]
SEEDS = [42, 43, 44, 45, 46]

# Model configurations - mapping internal ID to display name and label file model name
MODELS = {
    "meta_llama_Llama_3.1_8B_Instruct": {
        "display_name": "Llama 3.1 8B",
        "label_model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "generation_prefix": "meta_llama_Llama_3.1_8B_Instruct",
    },
    "Qwen_Qwen2.5_7B_Instruct": {
        "display_name": "Qwen 2.5 7B",
        "label_model_name": "Qwen_Qwen2.5_7B_Instruct",
        "generation_prefix": "Qwen_Qwen2.5_7B_Instruct",
    },
    "Qwen_Qwen3_8B": {
        "display_name": "Qwen 3 8B",
        "label_model_name": "Qwen_Qwen3-8B",
        "generation_prefix": "Qwen_Qwen3-8B",
    },
    "google_gemma_3_12b_it": {
        "display_name": "Gemma 3 12B",
        "label_model_name": "google_gemma-3-12b-it",
        "generation_prefix": "google_gemma-3-12b-it",
    },
}

# Model stats from the unified judge (Claude 3.5 Haiku)
MODEL_STATS = {
    "meta_llama_Llama_3.1_8B_Instruct": {"mean_ssi": 0.944, "flip_rate": 27.3, "pct_unstable": 10.4, "refusal_rate": 79.3},
    "Qwen_Qwen2.5_7B_Instruct": {"mean_ssi": 0.938, "flip_rate": 26.3, "pct_unstable": 12.0, "refusal_rate": 81.3},
    "Qwen_Qwen3_8B": {"mean_ssi": 0.938, "flip_rate": 27.7, "pct_unstable": 11.8, "refusal_rate": 92.5},
    "google_gemma_3_12b_it": {"mean_ssi": 0.965, "flip_rate": 18.4, "pct_unstable": 6.7, "refusal_rate": 78.5},
}


def load_labels():
    """Load all Claude Haiku labels from both label files."""
    dfs = []

    # Load Llama and Qwen 2.5 labels
    llama_qwen25_file = LABELS_DIR / "claude_haiku_llama_qwen25.csv"
    if llama_qwen25_file.exists():
        df = pd.read_csv(llama_qwen25_file)
        dfs.append(df)
        print(f"Loaded {len(df)} labels from {llama_qwen25_file.name}")

    # Load Qwen 3 and Gemma 3 labels
    new_models_file = LABELS_DIR / "claude_haiku_new_models.csv"
    if new_models_file.exists():
        df = pd.read_csv(new_models_file)
        dfs.append(df)
        print(f"Loaded {len(df)} labels from {new_models_file.name}")

    if not dfs:
        raise FileNotFoundError("No label files found")

    return pd.concat(dfs, ignore_index=True)


def load_responses():
    """Load all responses from JSONL files."""
    responses = []

    # Try loading from JSONL files first
    llama_qwen25_file = RESPONSES_DIR / "llama_qwen25_responses.jsonl"
    if llama_qwen25_file.exists():
        with open(llama_qwen25_file) as f:
            for line in f:
                responses.append(json.loads(line))
        print(f"Loaded {len(responses)} responses from {llama_qwen25_file.name}")

    new_models_file = RESPONSES_DIR / "new_models_responses.jsonl"
    if new_models_file.exists():
        count_before = len(responses)
        with open(new_models_file) as f:
            for line in f:
                responses.append(json.loads(line))
        print(f"Loaded {len(responses) - count_before} responses from {new_models_file.name}")

    # Fall back to individual generation CSVs
    if not responses:
        for model_id, model_config in MODELS.items():
            prefix = model_config["generation_prefix"]
            for temp in TEMPS:
                for seed in SEEDS:
                    filename = f"{prefix}_temp{temp}_seed{seed}.csv"
                    filepath = GENERATIONS_DIR / filename
                    if filepath.exists():
                        df = pd.read_csv(filepath)
                        for _, row in df.iterrows():
                            responses.append({
                                "model": model_config["label_model_name"],
                                "prompt_id": row["prompt_id"],
                                "prompt": row["prompt"],
                                "response": row["response"],
                                "temperature": temp,
                                "seed": seed,
                            })
        print(f"Loaded {len(responses)} responses from generation CSVs")

    return responses


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


def compute_ssi(labels):
    """Compute Safety Stability Index for a list of labels."""
    if not labels:
        return 1.0
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    return max(label_counts.values()) / len(labels)


def select_prompts(labels_df, responses, num_per_category=10):
    """
    Select a curated subset of prompts that shows interesting stability patterns.

    Selection criteria:
    - Include prompts with varying stability levels
    - Prioritize prompts that show different behavior across models
    - Keep prompts short enough for display
    """
    # Build a lookup for responses by (model, prompt_id, temp, seed)
    response_lookup = {}
    prompt_texts = {}
    for r in responses:
        key = (r["model"], r["prompt_id"], r["temperature"], r["seed"])
        response_lookup[key] = r["response"]
        prompt_texts[r["prompt_id"]] = r["prompt"]

    # Group labels by prompt_id and model
    grouped = labels_df.groupby(["prompt_id", "model"])["label"].apply(list).reset_index()

    # Compute SSI per prompt per model
    prompt_model_ssi = {}
    for _, row in grouped.iterrows():
        prompt_id = row["prompt_id"]
        model = row["model"]
        labels = row["label"]
        ssi = compute_ssi(labels)

        if prompt_id not in prompt_model_ssi:
            prompt_model_ssi[prompt_id] = {}
        prompt_model_ssi[prompt_id][model] = ssi

    # Select prompts with interesting patterns
    selected_prompt_ids = set()

    # 1. Most unstable prompts (lowest mean SSI across models)
    prompt_mean_ssi = []
    for prompt_id, model_ssis in prompt_model_ssi.items():
        if len(model_ssis) >= 2:  # At least 2 models have data
            mean_ssi = sum(model_ssis.values()) / len(model_ssis)
            prompt_text = prompt_texts.get(prompt_id, "")
            if len(prompt_text) < 200:  # Short enough to display
                prompt_mean_ssi.append((prompt_id, mean_ssi))

    prompt_mean_ssi.sort(key=lambda x: x[1])

    # Get most unstable
    for prompt_id, _ in prompt_mean_ssi[:15]:
        selected_prompt_ids.add(prompt_id)

    # 2. Stable refusers (high SSI)
    for prompt_id, _ in prompt_mean_ssi[-10:]:
        selected_prompt_ids.add(prompt_id)

    # 3. Prompts with high model variance (different models behave differently)
    prompt_ssi_variance = []
    for prompt_id, model_ssis in prompt_model_ssi.items():
        if len(model_ssis) >= 3:  # At least 3 models
            values = list(model_ssis.values())
            variance = max(values) - min(values)
            prompt_text = prompt_texts.get(prompt_id, "")
            if len(prompt_text) < 200:
                prompt_ssi_variance.append((prompt_id, variance))

    prompt_ssi_variance.sort(key=lambda x: -x[1])
    for prompt_id, _ in prompt_ssi_variance[:10]:
        selected_prompt_ids.add(prompt_id)

    # 4. Edge cases (SSI around 0.7-0.9)
    for prompt_id, mean_ssi in prompt_mean_ssi:
        if 0.65 <= mean_ssi <= 0.85:
            selected_prompt_ids.add(prompt_id)
            if len(selected_prompt_ids) >= 40:
                break

    print(f"Selected {len(selected_prompt_ids)} prompts")
    return selected_prompt_ids


def build_prompt_data(prompt_id, prompt_text, labels_df, response_lookup):
    """Build the multi-model response data for a single prompt."""
    models_data = {}

    for model_id, model_config in MODELS.items():
        label_model_name = model_config["label_model_name"]

        # Get labels for this model and prompt
        model_labels = labels_df[
            (labels_df["prompt_id"] == prompt_id) &
            (labels_df["model"] == label_model_name)
        ]

        if len(model_labels) == 0:
            continue

        responses = []
        labels_list = []

        for temp in TEMPS:
            for seed in SEEDS:
                row = model_labels[
                    (model_labels["temperature"] == temp) &
                    (model_labels["seed"] == seed)
                ]
                if len(row) > 0:
                    label = row.iloc[0]["label"]
                    labels_list.append(label)

                    # Get response text
                    response_key = (label_model_name, prompt_id, temp, seed)
                    response_text = response_lookup.get(response_key, "")
                    if response_text:
                        response_text = response_text[:1000]  # Truncate long responses

                    responses.append({
                        "temp": temp,
                        "seed": seed,
                        "label": label,
                        "text": response_text,
                    })

        if len(responses) >= 15:  # At least 15/20 responses present
            ssi = compute_ssi(labels_list)
            models_data[model_id] = {
                "ssi": round(ssi, 3),
                "responses": responses,
            }

    if not models_data:
        return None

    return {
        "id": prompt_id,
        "text": prompt_text,
        "category": categorize_prompt(prompt_text),
        "models": models_data,
    }


def main():
    print("Loading data...")
    labels_df = load_labels()
    responses = load_responses()

    print(f"\nTotal labels: {len(labels_df)}")
    print(f"Total responses: {len(responses)}")

    # Build response lookup
    response_lookup = {}
    prompt_texts = {}
    for r in responses:
        key = (r["model"], r["prompt_id"], r["temperature"], r["seed"])
        response_lookup[key] = r["response"]
        prompt_texts[r["prompt_id"]] = r["prompt"]

    print("\nSelecting prompts...")
    selected_ids = select_prompts(labels_df, responses)

    print("\nBuilding demo data...")
    prompts = []
    for prompt_id in selected_ids:
        prompt_text = prompt_texts.get(prompt_id, "")
        if not prompt_text:
            continue

        prompt_data = build_prompt_data(prompt_id, prompt_text, labels_df, response_lookup)
        if prompt_data and len(prompt_data["models"]) >= 2:  # At least 2 models
            prompts.append(prompt_data)

    # Sort by average SSI (most unstable first)
    def avg_ssi(p):
        ssis = [m["ssi"] for m in p["models"].values()]
        return sum(ssis) / len(ssis) if ssis else 1.0

    prompts.sort(key=avg_ssi)

    print(f"\nFinal prompt count: {len(prompts)}")

    # Build output
    output = {
        "prompts": prompts,
        "metadata": {
            "models": list(MODELS.keys()),
            "temps": TEMPS,
            "seeds": SEEDS,
            "total_prompts": 876,
            "total_responses": 70080,
            "judge": "Claude 3.5 Haiku",
            "model_stats": MODEL_STATS,
            "subset_description": "Curated subset showing range of stability scores across 4 models"
        }
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {OUTPUT_FILE}")

    # Print summary
    print("\nSSI distribution in selected prompts:")
    ssi_values = [avg_ssi(p) for p in prompts]
    if ssi_values:
        print(f"  Min: {min(ssi_values):.3f}")
        print(f"  Max: {max(ssi_values):.3f}")
        print(f"  Mean: {sum(ssi_values)/len(ssi_values):.3f}")

    print("\nCategory distribution:")
    categories = defaultdict(int)
    for p in prompts:
        categories[p["category"]] += 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nModels per prompt:")
    model_counts = defaultdict(int)
    for p in prompts:
        model_counts[len(p["models"])] += 1
    for n_models, count in sorted(model_counts.items()):
        print(f"  {n_models} models: {count} prompts")


if __name__ == "__main__":
    main()
