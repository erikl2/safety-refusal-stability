"""
Configuration loading utilities for generation experiments.
"""

import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    hf_name: str
    short_name: str
    family: str
    size: str
    requires_auth: bool = False


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperatures: list[float]
    seeds: list[int]
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    batch_size: int


@dataclass
class VLLMConfig:
    """Configuration for vLLM settings."""
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    max_model_len: int


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sampling_config(config_path: Optional[Path] = None) -> SamplingConfig:
    """
    Load sampling configuration from YAML file.

    Args:
        config_path: Path to sampling.yaml (default: configs/sampling.yaml)

    Returns:
        SamplingConfig dataclass
    """
    config_path = config_path or CONFIGS_DIR / "sampling.yaml"
    data = load_yaml(config_path)

    return SamplingConfig(
        temperatures=data["temperatures"],
        seeds=data["seeds"],
        max_new_tokens=data["generation"]["max_new_tokens"],
        top_p=data["generation"]["top_p"],
        top_k=data["generation"].get("top_k", -1),
        repetition_penalty=data["generation"].get("repetition_penalty", 1.0),
        batch_size=data["batch"]["size"],
    )


def load_model_config(
    model_name: str,
    config_path: Optional[Path] = None,
) -> ModelConfig:
    """
    Load configuration for a specific model.

    Args:
        model_name: Key name of the model (e.g., "llama-3.1-8b-instruct")
        config_path: Path to models.yaml (default: configs/models.yaml)

    Returns:
        ModelConfig dataclass
    """
    config_path = config_path or CONFIGS_DIR / "models.yaml"
    data = load_yaml(config_path)

    if model_name not in data["models"]:
        available = list(data["models"].keys())
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available models: {available}"
        )

    model_data = data["models"][model_name]
    return ModelConfig(
        name=model_name,
        hf_name=model_data["hf_name"],
        short_name=model_data["short_name"],
        family=model_data["family"],
        size=model_data["size"],
        requires_auth=model_data.get("requires_auth", False),
    )


def load_vllm_config(config_path: Optional[Path] = None) -> VLLMConfig:
    """
    Load vLLM configuration from YAML file.

    Args:
        config_path: Path to models.yaml (default: configs/models.yaml)

    Returns:
        VLLMConfig dataclass
    """
    config_path = config_path or CONFIGS_DIR / "models.yaml"
    data = load_yaml(config_path)

    vllm_data = data.get("vllm", {})
    return VLLMConfig(
        tensor_parallel_size=vllm_data.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_data.get("gpu_memory_utilization", 0.9),
        dtype=vllm_data.get("dtype", "auto"),
        max_model_len=vllm_data.get("max_model_len", 4096),
    )


def get_all_model_names(config_path: Optional[Path] = None) -> list[str]:
    """
    Get list of all available model names from config.

    Args:
        config_path: Path to models.yaml (default: configs/models.yaml)

    Returns:
        List of model key names
    """
    config_path = config_path or CONFIGS_DIR / "models.yaml"
    data = load_yaml(config_path)
    return list(data["models"].keys())


def load_pilot_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load pilot experiment configuration.

    Args:
        config_path: Path to sampling.yaml (default: configs/sampling.yaml)

    Returns:
        Dict with pilot configuration
    """
    config_path = config_path or CONFIGS_DIR / "sampling.yaml"
    data = load_yaml(config_path)
    return data.get("pilot", {
        "num_prompts": 50,
        "temperatures": [0.7],
        "seeds": [42, 43],
    })
