from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class LocalModelClient:
    """Drop-in replacement for InferenceClient using local transformers models."""

    def __init__(self):
        self._models: dict[str, tuple] = {}  # model_name -> (model, tokenizer)

    def _load_model(self, model_name: str):
        if model_name in self._models:
            return self._models[model_name]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local model: {model_name} (float16, device_map=auto)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        self._models[model_name] = (model, tokenizer)
        logger.info(f"Model {model_name} loaded successfully")
        return model, tokenizer

    def chat_completion(self, model, messages, temperature=1.0, max_tokens=128, **kwargs):
        import torch

        lm, tokenizer = self._load_model(model)

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(lm.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.05
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.get("top_p", 1.0)

        with torch.no_grad():
            output_ids = lm.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_len:]
        content = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Mimic HF ChatCompletionOutput structure
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and inject HF_TOKEN from environment or .env file."""
    load_dotenv()
    with open(path) as f:
        config = yaml.safe_load(f)
    backend = config.get("backend", "api")
    if backend == "local":
        config["hf_token"] = os.environ.get("HF_TOKEN", "")
    else:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found. Set it as an environment variable or in a .env file."
            )
        config["hf_token"] = hf_token
    return config


def set_seed(seed: int):
    """Set random and numpy seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class JsonlCache:
    """Append-mode JSONL cache for resumability."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def load(self) -> list[dict]:
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def append(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def __len__(self) -> int:
        return len(self.load())

    def get_processed_indices(self) -> set[int]:
        return {r["query_idx"] for r in self.load()}


def setup_logging() -> logging.Logger:
    """Configure logging with timestamps and INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_hf_client(config: dict):
    """Return an InferenceClient or LocalModelClient based on backend config."""
    backend = config.get("backend", "api")
    if backend == "local":
        return LocalModelClient()
    return InferenceClient(token=config["hf_token"])
