"""
Preprocessing utilities for unifying and cleaning harmful prompt datasets.
"""

import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

from .load_datasets import load_all_datasets

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, collapse whitespace)."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Collapse multiple whitespace to single space
    return " ".join(text.split())


def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def deduplicate_prompts(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    similarity_threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Remove duplicate and near-duplicate prompts.

    Args:
        df: DataFrame with prompts
        prompt_col: Name of the column containing prompts
        similarity_threshold: Threshold for fuzzy deduplication (0.9 = 90% similar)

    Returns:
        DataFrame with duplicates removed
    """
    print(f"Deduplicating {len(df)} prompts (threshold={similarity_threshold})...")

    # First: exact deduplication (fast)
    df["_normalized"] = df[prompt_col].apply(normalize_text)
    df_deduped = df.drop_duplicates(subset="_normalized", keep="first").copy()
    exact_removed = len(df) - len(df_deduped)
    print(f"  Removed {exact_removed} exact duplicates")

    # Second: fuzzy deduplication (slower, but necessary)
    # For large datasets, this can be slow - O(n^2) comparison
    if len(df_deduped) > 2000:
        print(f"  Warning: {len(df_deduped)} prompts - fuzzy dedup may be slow")

    prompts = df_deduped["_normalized"].tolist()
    indices_to_keep = []
    seen_prompts = []

    for i, prompt in enumerate(prompts):
        is_duplicate = False
        for seen in seen_prompts:
            if compute_similarity(prompt, seen) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            indices_to_keep.append(i)
            seen_prompts.append(prompt)

        # Progress indicator for large datasets
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts...")

    df_final = df_deduped.iloc[indices_to_keep].copy()
    fuzzy_removed = len(df_deduped) - len(df_final)
    print(f"  Removed {fuzzy_removed} fuzzy duplicates")

    # Clean up temp column
    df_final = df_final.drop(columns=["_normalized"])

    print(f"  Final: {len(df_final)} unique prompts")
    return df_final


def extract_prompt_column(df: pd.DataFrame, source: str) -> pd.Series:
    """
    Extract the prompt/behavior column from a dataset.
    Different datasets use different column names.
    """
    # Common column names for harmful prompts
    prompt_columns = [
        "goal",           # AdvBench
        "Behavior",       # HarmBench
        "behavior",       # HarmBench (lowercase)
        "Goal",           # JailbreakBench
        "prompt",         # Generic
        "text",           # Generic
        "question",       # Generic
    ]

    for col in prompt_columns:
        if col in df.columns:
            return df[col]

    raise ValueError(
        f"Could not find prompt column in {source} dataset. "
        f"Available columns: {df.columns.tolist()}"
    )


def generate_prompt_id(prompt: str, source: str) -> str:
    """Generate a unique ID for a prompt based on content hash."""
    content = f"{source}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def preprocess_prompts(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    similarity_threshold: float = 0.9,
    sample_size: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, unify, and preprocess all harmful prompt datasets.

    Args:
        raw_data_dir: Directory containing raw datasets
        output_dir: Directory to save processed data
        similarity_threshold: Threshold for fuzzy deduplication
        sample_size: Number of prompts for sample file

    Returns:
        Tuple of (full DataFrame, sample DataFrame)
    """
    output_dir = output_dir or PROCESSED_DATA_DIR

    # Load all datasets
    print("Loading raw datasets...")
    datasets = load_all_datasets(raw_data_dir)

    if not datasets:
        raise ValueError("No datasets loaded. Run scripts/01_download_data.py first.")

    # Extract and unify prompts
    print("\nExtracting prompts from datasets...")
    all_prompts = []

    for source, df in datasets.items():
        try:
            prompts = extract_prompt_column(df, source)
            for prompt in prompts:
                if pd.notna(prompt) and str(prompt).strip():
                    all_prompts.append({
                        "prompt": str(prompt).strip(),
                        "source": source,
                    })
            print(f"  {source}: {len(prompts)} prompts")
        except ValueError as e:
            print(f"  {source}: Skipping - {e}")

    # Create unified DataFrame
    df_unified = pd.DataFrame(all_prompts)
    print(f"\nTotal prompts before deduplication: {len(df_unified)}")

    # Deduplicate
    df_deduped = deduplicate_prompts(
        df_unified,
        prompt_col="prompt",
        similarity_threshold=similarity_threshold,
    )

    # Add metadata columns
    df_deduped["id"] = [
        generate_prompt_id(row["prompt"], row["source"])
        for _, row in df_deduped.iterrows()
    ]
    df_deduped["harm_category"] = ""  # To be filled later
    df_deduped["char_length"] = df_deduped["prompt"].str.len()
    df_deduped["word_count"] = df_deduped["prompt"].str.split().str.len()

    # Reorder columns
    df_final = df_deduped[["id", "prompt", "source", "harm_category", "char_length", "word_count"]]

    # Create sample for testing
    df_sample = df_final.sample(n=min(sample_size, len(df_final)), random_state=42)

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "prompts.csv"
    sample_path = output_dir / "prompts_sample.csv"

    df_final.to_csv(full_path, index=False)
    df_sample.to_csv(sample_path, index=False)

    print(f"\nSaved {len(df_final)} prompts to: {full_path}")
    print(f"Saved {len(df_sample)} sample prompts to: {sample_path}")

    # Print statistics
    print("\n" + "=" * 40)
    print("Dataset Statistics")
    print("=" * 40)
    print(f"Total unique prompts: {len(df_final)}")
    print(f"Sources: {df_final['source'].value_counts().to_dict()}")
    print(f"Avg prompt length: {df_final['char_length'].mean():.1f} chars")
    print(f"Avg word count: {df_final['word_count'].mean():.1f} words")

    return df_final, df_sample
