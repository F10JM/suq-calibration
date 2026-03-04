"""Step 3: Compute ECE, plot reliability diagrams, save metrics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config
from src.calibration import run_calibration


def main():
    parser = argparse.ArgumentParser(description="Run calibration step")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_calibration(config)


if __name__ == "__main__":
    main()
