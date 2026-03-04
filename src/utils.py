from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and inject HF_TOKEN from environment or .env file."""
    load_dotenv()
    with open(path) as f:
        config = yaml.safe_load(f)
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


def get_hf_client(config: dict) -> InferenceClient:
    """Return an InferenceClient authenticated with the HF token."""
    return InferenceClient(token=config["hf_token"])
