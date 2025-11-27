"""
Functions for loading raw datasets from disk.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_advbench(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load AdvBench harmful behaviors dataset.

    Returns:
        DataFrame with columns: goal, target (the harmful prompt and target response)
    """
    data_dir = data_dir or RAW_DATA_DIR
    file_path = data_dir / "advbench.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"AdvBench dataset not found at {file_path}. "
            "Run scripts/01_download_data.py first."
        )

    df = pd.read_csv(file_path)

    # AdvBench has 'goal' and 'target' columns
    # 'goal' is the harmful prompt, 'target' is the expected harmful output
    if "goal" not in df.columns:
        raise ValueError(f"Expected 'goal' column in AdvBench data. Found: {df.columns.tolist()}")

    df["source"] = "advbench"
    return df


def load_harmbench(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load HarmBench dataset.

    Returns:
        DataFrame with harmful behavior prompts
    """
    data_dir = data_dir or RAW_DATA_DIR
    file_path = data_dir / "harmbench.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"HarmBench dataset not found at {file_path}. "
            "Run scripts/01_download_data.py first."
        )

    df = pd.read_csv(file_path)
    df["source"] = "harmbench"

    return df


def load_jailbreakbench(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load JailbreakBench dataset (optional).

    Returns:
        DataFrame with harmful behavior prompts
    """
    data_dir = data_dir or RAW_DATA_DIR
    file_path = data_dir / "jailbreakbench.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"JailbreakBench dataset not found at {file_path}. "
            "Run scripts/01_download_data.py first."
        )

    df = pd.read_csv(file_path)
    df["source"] = "jailbreakbench"

    return df


def load_all_datasets(data_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Load all available datasets.

    Returns:
        Dictionary mapping dataset name to DataFrame
    """
    data_dir = data_dir or RAW_DATA_DIR
    datasets = {}

    # Try loading each dataset
    loaders = {
        "advbench": load_advbench,
        "harmbench": load_harmbench,
        "jailbreakbench": load_jailbreakbench,
    }

    for name, loader in loaders.items():
        try:
            datasets[name] = loader(data_dir)
            print(f"Loaded {name}: {len(datasets[name])} rows")
        except FileNotFoundError:
            print(f"Skipping {name}: file not found")

    return datasets
