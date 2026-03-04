from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer

from src.utils import JsonlCache, get_hf_client, setup_logging

logger = logging.getLogger(__name__)


class LLMJudgeSimilarity:
    """Similarity via LLM-as-a-judge evaluation."""

    def __init__(self, client: InferenceClient, model: str, config: dict):
        self.client = client
        self.model = model
        self.max_retries = config.get("api_max_retries", 5)
        self.retry_delay = config.get("api_retry_delay", 2.0)

    def compute(self, question: str, answer_a: str, answer_b: str) -> float:
        prompt = (
            "You are evaluating whether two answers to a question are consistent.\n\n"
            f"Question: {question}\n"
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}\n\n"
            "Are these answers consistent with each other? Rate on a scale of 0 to 1:\n"
            "- 1.0: Fully consistent (same core information)\n"
            "- 0.5: Partially consistent\n"
            "- 0.0: Contradictory or completely different\n\n"
            "Respond with ONLY a single number between 0 and 1."
        )
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=0.01,
                    max_tokens=10,
                )
                text = response.choices[0].message.content.strip()
                try:
                    value = float(text.split()[0].strip(".,;:"))
                    return max(0.0, min(1.0, value))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse similarity score: '{text}', using 0.5")
                    return 0.5
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise


class RougeLSimilarity:
    """Similarity via ROUGE-L F1 score."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute(self, question: str, answer_a: str, answer_b: str) -> float:
        scores = self.scorer.score(answer_a, answer_b)
        return scores["rougeL"].fmeasure


def get_similarity_fn(method: str, client, config: dict):
    """Factory for similarity functions."""
    if method == "llm_judge":
        return LLMJudgeSimilarity(client, config["evaluator_model"], config)
    elif method == "rouge_l":
        return RougeLSimilarity()
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_pairwise_matrix(
    similarity_fn, question: str, samples: list[str],
    max_workers: int = 1,
) -> np.ndarray:
    """Compute K×K symmetric similarity matrix."""
    K = len(samples)
    matrix = np.eye(K)

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]

    if max_workers > 1 and hasattr(similarity_fn, 'client'):
        # Parallel for API-based similarity
        def _compute_pair(pair):
            i, j = pair
            return i, j, similarity_fn.compute(question, samples[i], samples[j])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compute_pair, p) for p in pairs]
            for future in as_completed(futures):
                i, j, score = future.result()
                matrix[i, j] = score
                matrix[j, i] = score
    else:
        for i, j in pairs:
            score = similarity_fn.compute(question, samples[i], samples[j])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix


def compute_reference_similarities(
    similarity_fn, question: str, samples: list[str], reference: str,
    max_workers: int = 1,
) -> np.ndarray:
    """Compute S(y_k, y_ref) for each sample k."""
    K = len(samples)

    if max_workers > 1 and hasattr(similarity_fn, 'client'):
        results = [None] * K

        def _compute(idx):
            return idx, similarity_fn.compute(question, samples[idx], reference)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compute, k) for k in range(K)]
            for future in as_completed(futures):
                idx, score = future.result()
                results[idx] = score
        return np.array(results)
    else:
        return np.array([
            similarity_fn.compute(question, s, reference) for s in samples
        ])


def run_similarity(config: dict):
    """Main entry point: compute pairwise and reference similarities."""
    setup_logging()
    client = get_hf_client(config)

    gen_cache = JsonlCache(config["generation_cache"])
    generations = gen_cache.load()
    if not generations:
        raise RuntimeError(
            "No generations found. Run generation step first."
        )
    logger.info(f"Loaded {len(generations)} generated queries")

    max_workers = config.get("api_concurrent_requests", 4)

    for method in config["similarity_methods"]:
        cache_path = f"{config['similarity_cache_dir']}/similarities_{method}.jsonl"
        cache = JsonlCache(cache_path)
        processed = cache.get_processed_indices()

        sim_fn = get_similarity_fn(method, client, config)
        workers = max_workers if method == "llm_judge" else 1

        total = len(generations)
        skipped = sum(1 for g in generations if g["query_idx"] in processed)
        logger.info(
            f"Similarity [{method}]: {total} queries, {skipped} cached, "
            f"{total - skipped} remaining"
        )

        for gen in generations:
            if gen["query_idx"] in processed:
                continue

            pairwise = compute_pairwise_matrix(
                sim_fn, gen["question"], gen["samples"], max_workers=workers,
            )
            vs_ref = compute_reference_similarities(
                sim_fn, gen["question"], gen["samples"],
                gen["reference_answer"], max_workers=workers,
            )

            record = {
                "query_idx": gen["query_idx"],
                "method": method,
                "pairwise": pairwise.tolist(),
                "vs_reference": vs_ref.tolist(),
            }
            cache.append(record)
            processed.add(gen["query_idx"])

            done = len(processed)
            if done % 10 == 0 or done == total:
                logger.info(f"Similarity [{method}] progress: {done}/{total}")

        logger.info(f"Similarity [{method}] complete. {len(processed)} queries.")
