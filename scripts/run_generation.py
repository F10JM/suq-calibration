"""Step 1: Generate K responses per query via HF Inference API."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config
from src.generate import run_generation


def main():
    parser = argparse.ArgumentParser(description="Run generation step")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_generation(config)


if __name__ == "__main__":
    main()
