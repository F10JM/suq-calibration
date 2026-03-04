from __future__ import annotations

import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_dataset_queries(config: dict) -> list[dict]:
    """Load and preprocess dataset queries."""
    dataset_name = config["dataset"]
    split = config["dataset_split"]
    num_queries = config["num_queries"]
    seed = config["seed"]

    if dataset_name == "trivia_qa":
        # Use "rc.nocontext" to avoid downloading large context files.
        # Use streaming to avoid downloading full dataset.
        ds = load_dataset(
            "trivia_qa", "rc.nocontext", split=split, streaming=True
        )
        # Shuffle the stream with a buffer, then take what we need.
        # Buffer size should be larger than num_queries for good randomization.
        ds = ds.shuffle(seed=seed, buffer_size=max(num_queries * 5, 5000))
        queries = []
        for i, row in enumerate(ds):
            if i >= num_queries:
                break
            queries.append({
                "query_idx": i,
                "question": row["question"],
                "context": None,
                "reference_answer": row["answer"]["value"],
                "reference_answers": row["answer"]["aliases"],
            })
        logger.info(f"Loaded {len(queries)} TriviaQA queries")

    elif dataset_name == "coqa":
        ds = load_dataset("stanfordnlp/coqa", split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=max(num_queries * 5, 1000))
        queries = []
        for i, row in enumerate(ds):
            if i >= num_queries:
                break
            queries.append({
                "query_idx": i,
                "question": row["questions"][0],
                "context": row["story"][:500],
                "reference_answer": row["answers"]["input_text"][0],
                "reference_answers": [row["answers"]["input_text"][0]],
            })
        logger.info(f"Loaded {len(queries)} CoQA queries")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return queries
