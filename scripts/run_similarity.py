"""Step 2: Compute pairwise and reference similarities."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config
from src.similarity import run_similarity


def main():
    parser = argparse.ArgumentParser(description="Run similarity step")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_similarity(config)


if __name__ == "__main__":
    main()
