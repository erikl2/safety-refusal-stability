#!/usr/bin/env python3
"""
Download datasets for safety refusal stability experiments.

Downloads:
- AdvBench harmful behaviors dataset
- HarmBench dataset (via HuggingFace or direct download)

Usage:
    python scripts/01_download_data.py
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_file(url: str, dest_path: Path, description: str) -> bool:
    """Download a file from URL to destination path."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(response.text, encoding="utf-8")

        print(f"  Success! Downloaded {len(response.text):,} bytes")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  Error downloading: {e}")
        return False


def download_advbench() -> bool:
    """Download AdvBench harmful behaviors dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    dest_path = RAW_DATA_DIR / "advbench.csv"

    return download_file(url, dest_path, "AdvBench harmful behaviors")


def download_harmbench() -> bool:
    """
    Download HarmBench dataset.

    Tries HuggingFace datasets first, falls back to direct download.
    """
    dest_path = RAW_DATA_DIR / "harmbench.csv"

    # Try HuggingFace datasets library first
    print("Downloading HarmBench dataset...")
    print("  Trying HuggingFace datasets library...")

    try:
        from datasets import load_dataset

        # HarmBench is available on HuggingFace
        dataset = load_dataset("harmbench/harmbench", split="test")

        # Convert to pandas and save as CSV
        df = dataset.to_pandas()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest_path, index=False)

        print(f"  Success! Downloaded {len(df)} samples via HuggingFace")
        return True

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Trying alternative sources...")

    # Fallback: Try the official HarmBench GitHub repo
    urls_to_try = [
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    ]

    for url in urls_to_try:
        if download_file(url, dest_path, "HarmBench (GitHub fallback)"):
            return True

    print("  Warning: Could not download HarmBench. You may need to download manually.")
    return False


def download_jailbreakbench() -> bool:
    """
    Download JailbreakBench dataset (optional, additional harmful prompts).
    """
    dest_path = RAW_DATA_DIR / "jailbreakbench.csv"

    print("Downloading JailbreakBench dataset (optional)...")

    try:
        from datasets import load_dataset

        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        df = dataset.to_pandas()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest_path, index=False)

        print(f"  Success! Downloaded {len(df)} samples via HuggingFace")
        return True

    except Exception as e:
        print(f"  JailbreakBench download failed (optional): {e}")
        return False


def create_manifest(results: dict) -> None:
    """Create a manifest file documenting what was downloaded."""
    manifest_path = RAW_DATA_DIR / "manifest.json"

    manifest = {
        "download_date": datetime.now().isoformat(),
        "datasets": {},
    }

    for name, success in results.items():
        file_path = RAW_DATA_DIR / f"{name}.csv"
        if success and file_path.exists():
            manifest["datasets"][name] = {
                "status": "downloaded",
                "file": str(file_path.name),
                "size_bytes": file_path.stat().st_size,
            }

            # Count rows
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    num_lines = sum(1 for _ in f) - 1  # subtract header
                manifest["datasets"][name]["num_rows"] = num_lines
            except Exception:
                pass
        else:
            manifest["datasets"][name] = {
                "status": "failed",
            }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Safety Refusal Stability - Dataset Download")
    print("=" * 60)
    print()

    # Ensure raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download all datasets
    results = {}

    print("-" * 40)
    results["advbench"] = download_advbench()
    print()

    print("-" * 40)
    results["harmbench"] = download_harmbench()
    print()

    print("-" * 40)
    results["jailbreakbench"] = download_jailbreakbench()
    print()

    # Create manifest
    print("-" * 40)
    create_manifest(results)

    # Summary
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    successful = sum(results.values())
    total = len(results)
    print()
    print(f"Downloaded {successful}/{total} datasets successfully.")

    if successful == 0:
        print("\nError: No datasets downloaded. Check your internet connection.")
        return 1

    print(f"\nRaw data saved to: {RAW_DATA_DIR}")
    return 0


if __name__ == "__main__":
    exit(main())
