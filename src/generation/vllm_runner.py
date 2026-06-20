"""
vLLM batch generation runner for safety refusal stability experiments.
"""

import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from tqdm import tqdm

from .config import load_model_config, load_vllm_config, ModelConfig, VLLMConfig


@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt_id: str
    prompt: str
    response: str
    model: str
    temperature: float
    seed: int
    generation_time_ms: float


class VLLMRunner:
    """
    Batch generation runner using vLLM.

    Example usage:
        runner = VLLMRunner(model_name="meta-llama/Llama-3.1-8B-Instruct")
        results = runner.generate(
            prompts=[{"id": "1", "prompt": "Hello"}],
            temperature=0.7,
            seed=42,
        )
    """

    def __init__(
        self,
        model_name: str,
        model_config: Optional[ModelConfig] = None,
        vllm_config: Optional[VLLMConfig] = None,
        tensor_parallel_size: Optional[int] = None,
    ):
        """
        Initialize the vLLM runner.

        Args:
            model_name: HuggingFace model name or path
            model_config: Optional ModelConfig (loads from config if not provided)
            vllm_config: Optional VLLMConfig (loads from config if not provided)
            tensor_parallel_size: Override tensor parallel size (for multi-GPU)
        """
        self.model_name = model_name
        self.model_config = model_config
        self.vllm_config = vllm_config or load_vllm_config()

        if tensor_parallel_size is not None:
            self.vllm_config.tensor_parallel_size = tensor_parallel_size

        self._llm = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the vLLM model."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        print(f"Loading model: {self.model_name}")
        print(f"  Tensor parallel size: {self.vllm_config.tensor_parallel_size}")
        print(f"  GPU memory utilization: {self.vllm_config.gpu_memory_utilization}")
        print(f"  Max model length: {self.vllm_config.max_model_len}")

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.vllm_config.tensor_parallel_size,
            gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
            dtype=self.vllm_config.dtype,
            max_model_len=self.vllm_config.max_model_len,
            trust_remote_code=True,
        )

        self._tokenizer = self._llm.get_tokenizer()
        print(f"Model loaded successfully!")

    def _format_chat_prompt(self, prompt: str) -> str:
        """
        Format a prompt using the model's chat template.

        Args:
            prompt: The user message content

        Returns:
            Formatted prompt string ready for generation
        """
        messages = [{"role": "user", "content": prompt}]

        # Use the tokenizer's chat template
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return formatted

    def generate(
        self,
        prompts: list[dict],
        temperature: float,
        seed: int,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[GenerationResult]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of dicts with 'id' and 'prompt' keys
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            batch_size: Number of prompts per batch
            show_progress: Whether to show progress bar

        Returns:
            List of GenerationResult objects
        """
        self._load_model()

        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed")

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        # Format all prompts using chat template
        formatted_prompts = []
        prompt_mapping = {}  # Map formatted prompt to original

        for item in prompts:
            formatted = self._format_chat_prompt(item["prompt"])
            formatted_prompts.append(formatted)
            prompt_mapping[formatted] = item

        # Generate in batches
        results = []
        total_batches = (len(formatted_prompts) + batch_size - 1) // batch_size

        iterator = range(0, len(formatted_prompts), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"Generating (T={temperature}, seed={seed})",
            )

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(formatted_prompts))
            batch_prompts = formatted_prompts[batch_start:batch_end]
            batch_items = [prompt_mapping[p] for p in batch_prompts]

            # Time the generation
            start_time = time.time()

            try:
                outputs = self._llm.generate(batch_prompts, sampling_params)
            except Exception as e:
                print(f"  Error generating batch: {e}")
                # Create error results for this batch
                for item in batch_items:
                    results.append(GenerationResult(
                        prompt_id=item["id"],
                        prompt=item["prompt"],
                        response=f"[ERROR: {str(e)}]",
                        model=self.model_name,
                        temperature=temperature,
                        seed=seed,
                        generation_time_ms=0,
                    ))
                continue

            elapsed_ms = (time.time() - start_time) * 1000
            time_per_prompt = elapsed_ms / len(batch_prompts)

            # Extract results
            for output, item in zip(outputs, batch_items):
                response_text = output.outputs[0].text.strip()

                results.append(GenerationResult(
                    prompt_id=item["id"],
                    prompt=item["prompt"],
                    response=response_text,
                    model=self.model_name,
                    temperature=temperature,
                    seed=seed,
                    generation_time_ms=time_per_prompt,
                ))

        return results

    def generate_single(
        self,
        prompt: str,
        prompt_id: str = "single",
        temperature: float = 0.7,
        seed: int = 42,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        """
        Generate a single response (convenience method).

        Args:
            prompt: The prompt text
            prompt_id: ID for the prompt
            temperature: Sampling temperature
            seed: Random seed
            max_new_tokens: Maximum tokens to generate

        Returns:
            GenerationResult object
        """
        results = self.generate(
            prompts=[{"id": prompt_id, "prompt": prompt}],
            temperature=temperature,
            seed=seed,
            max_new_tokens=max_new_tokens,
            show_progress=False,
        )
        return results[0]


def results_to_dataframe(results: list[GenerationResult]):
    """Convert list of GenerationResult to pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame([
        {
            "prompt_id": r.prompt_id,
            "prompt": r.prompt,
            "response": r.response,
            "model": r.model,
            "temperature": r.temperature,
            "seed": r.seed,
            "generation_time_ms": r.generation_time_ms,
        }
        for r in results
    ])
