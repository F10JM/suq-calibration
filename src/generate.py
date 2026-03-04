from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import InferenceClient

from src.data import load_dataset_queries
from src.utils import JsonlCache, get_hf_client, set_seed, setup_logging

logger = logging.getLogger(__name__)


def build_messages(question: str, context: str | None = None) -> list[dict]:
    """Build chat messages for QA generation."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer the question concisely in a "
            "single line. Do not include any preamble or explanation beyond "
            "the answer itself."
        ),
    }
    user_content = ""
    if context:
        user_content += f"Context: {context}\n\n"
    user_content += f"Question: {question}"
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]


def sample_one_response(
    client: InferenceClient,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> str:
    """Sample a single response with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                raise


def sample_k_responses(
    client: InferenceClient,
    model: str,
    messages: list[dict],
    num_samples: int,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    max_workers: int = 4,
) -> list[str]:
    """Sample K independent responses, optionally in parallel."""
    results = [None] * num_samples

    def _sample(idx):
        return idx, sample_one_response(
            client, model, messages, temperature, max_tokens,
            max_retries, retry_delay,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    return results


def run_generation(config: dict):
    """Main entry point: generate K samples per query and cache results."""
    setup_logging()
    set_seed(config["seed"])

    queries = load_dataset_queries(config)
    client = get_hf_client(config)
    cache = JsonlCache(config["generation_cache"])
    processed = cache.get_processed_indices()

    model = config["generator_model"]
    num_samples = config["num_samples"]
    temperature = config["temperature"]
    max_tokens = config["max_new_tokens"]
    max_retries = config.get("api_max_retries", 5)
    retry_delay = config.get("api_retry_delay", 2.0)
    max_workers = config.get("api_concurrent_requests", 4)

    total = len(queries)
    skipped = sum(1 for q in queries if q["query_idx"] in processed)
    logger.info(
        f"Generation: {total} queries, {skipped} already cached, "
        f"{total - skipped} remaining"
    )

    for q in queries:
        if q["query_idx"] in processed:
            continue

        messages = build_messages(q["question"], q.get("context"))
        samples = sample_k_responses(
            client, model, messages, num_samples,
            temperature, max_tokens, max_retries, retry_delay, max_workers,
        )

        record = {
            "query_idx": q["query_idx"],
            "question": q["question"],
            "context": q.get("context"),
            "reference_answer": q["reference_answer"],
            "reference_answers": q["reference_answers"],
            "samples": samples,
        }
        cache.append(record)
        processed.add(q["query_idx"])

        done = len(processed)
        if done % 10 == 0 or done == total:
            logger.info(f"Generation progress: {done}/{total}")

    logger.info(f"Generation complete. {len(processed)} queries cached.")
