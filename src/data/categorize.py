"""
Harm category classification for prompts.
"""

from typing import Optional
import pandas as pd

# Harm categories based on common taxonomies
HARM_CATEGORIES = [
    "violence",
    "illegal_activity",
    "hate_speech",
    "self_harm",
    "sexual_content",
    "misinformation",
    "privacy_violation",
    "fraud",
    "weapons",
    "drugs",
    "other",
]


def classify_harm_category(prompt: str) -> str:
    """
    Classify a prompt into a harm category.

    This is a simple keyword-based classifier.
    For production use, consider using an LLM-based classifier.

    Args:
        prompt: The harmful prompt text

    Returns:
        Harm category string
    """
    prompt_lower = prompt.lower()

    # Simple keyword matching
    category_keywords = {
        "violence": ["kill", "murder", "attack", "hurt", "harm", "assault", "weapon"],
        "illegal_activity": ["illegal", "crime", "steal", "rob", "break in", "trespass"],
        "hate_speech": ["hate", "racist", "discriminat", "slur", "offensive"],
        "self_harm": ["suicide", "self-harm", "cut myself", "end my life"],
        "sexual_content": ["sexual", "porn", "nude", "explicit"],
        "misinformation": ["fake news", "conspiracy", "disinformation", "hoax"],
        "privacy_violation": ["doxx", "personal information", "hack", "stalk"],
        "fraud": ["scam", "fraud", "phishing", "deceive", "trick"],
        "weapons": ["bomb", "explosive", "gun", "firearm", "ammunition"],
        "drugs": ["drug", "narcotic", "cocaine", "heroin", "meth"],
    }

    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                return category

    return "other"


def categorize_prompts(
    prompts_df: pd.DataFrame,
    prompt_col: str = "prompt",
    category_col: str = "harm_category",
) -> pd.DataFrame:
    """
    Add harm categories to a prompts DataFrame.

    Args:
        prompts_df: DataFrame with prompts
        prompt_col: Name of prompt column
        category_col: Name of output category column

    Returns:
        DataFrame with added category column
    """
    df = prompts_df.copy()
    df[category_col] = df[prompt_col].apply(classify_harm_category)

    # Print distribution
    print("Harm category distribution:")
    print(df[category_col].value_counts())

    return df
