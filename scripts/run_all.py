"""Run the full pipeline: generation -> similarity -> calibration.

Each step checks its cache and skips already-completed work,
so this is safe to re-run after interruptions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.generate import run_generation
from src.similarity import run_similarity
from src.calibration import run_calibration


def main():
    parser = argparse.ArgumentParser(description="Run full SUQ calibration pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("STEP 1/3: Generation")
    logger.info("=" * 60)
    run_generation(config)

    logger.info("=" * 60)
    logger.info("STEP 2/3: Similarity")
    logger.info("=" * 60)
    run_similarity(config)

    logger.info("=" * 60)
    logger.info("STEP 3/3: Calibration")
    logger.info("=" * 60)
    run_calibration(config)

    logger.info("=" * 60)
    logger.info("Pipeline complete! Check results/ for outputs.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
