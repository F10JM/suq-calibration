# SUQ Calibration: Reproducing the QA Calibration Experiment

A clean, modular reproduction of the QA calibration experiment (Section 4.1) from:

> **"On Subjective Uncertainty Quantification and Calibration in Natural Language Generation"**
> Ziyu Wang & Thomas P. Holmes (2024)
> [arXiv:2406.05213v2](https://arxiv.org/abs/2406.05213v2)

The paper proposes a decision-theoretic framework for uncertainty quantification in free-form natural language generation. Instead of mapping outputs to discrete classes (as in standard classification calibration), it defines calibration through a **subjective utility function** parameterized by a single choice: the similarity measure *S(y', y; I)*. This project reproduces their experiment: generate multiple responses per question, compute pairwise similarity, derive subjective uncertainty (Bayes risk), and evaluate calibration via reliability diagrams and a generalized Expected Calibration Error (ECE).

## Project Structure

```
suq-calibration/
├── config.yaml                  # All hyperparameters and paths
├── config_test.yaml             # Small config (3 queries) for validation
├── .env.example                 # Template for HF API token
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Config loading, caching, logging, seeding
│   ├── data.py                  # Dataset loading (TriviaQA, CoQA) via streaming
│   ├── generate.py              # LLM sampling via HF Inference API
│   ├── similarity.py            # Similarity measures: LLM-judge and ROUGE-L
│   └── calibration.py           # Uncertainty computation, ECE, reliability plots
├── scripts/
│   ├── run_generation.py        # Step 1: generate K responses per query
│   ├── run_similarity.py        # Step 2: compute pairwise similarity matrices
│   ├── run_calibration.py       # Step 3: compute ECE and plot diagrams
│   └── run_all.py               # Run the full pipeline sequentially
└── results/                     # Created at runtime (gitignored)
    ├── generations.jsonl         # Cached generated responses
    ├── similarities_rouge_l.jsonl
    ├── similarities_llm_judge.jsonl
    ├── reliability_rouge_l.png
    ├── reliability_llm_judge.png
    ├── reliability_comparison.png
    └── metrics.json
```

### Source Modules

| File | Purpose |
|------|---------|
| `src/utils.py` | `load_config()` reads YAML + injects `HF_TOKEN` from env. `set_seed()` for reproducibility. `JsonlCache` provides append-mode JSONL caching with resumability. `setup_logging()` and `get_hf_client()` for shared infrastructure. |
| `src/data.py` | `load_dataset_queries()` loads TriviaQA (rc.nocontext) or CoQA via HuggingFace streaming, shuffles with a buffer, and returns `num_queries` question/answer dicts. |
| `src/generate.py` | `build_messages()` constructs chat prompts. `sample_one_response()` calls the HF Inference API with exponential backoff retry. `sample_k_responses()` generates K independent samples in parallel. `run_generation()` orchestrates the full generation loop with caching. |
| `src/similarity.py` | `LLMJudgeSimilarity` uses a large LLM to rate answer consistency 0-1. `RougeLSimilarity` computes ROUGE-L F1. `compute_pairwise_matrix()` builds the K x K symmetric similarity matrix. `compute_reference_similarities()` compares each sample to the ground truth. `run_similarity()` orchestrates everything with per-method caching. |
| `src/calibration.py` | `compute_subjective_uncertainty()` implements the Gibbs predictor Bayes risk. `compute_ece()` and `bootstrap_ece()` compute calibration metrics. `plot_reliability_diagram()` and `plot_comparison()` produce publication-quality figures. `run_calibration()` ties it all together. |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_generation.py` | Entry point for Step 1. Parses `--config`, calls `run_generation()`. |
| `scripts/run_similarity.py` | Entry point for Step 2. Parses `--config`, calls `run_similarity()`. |
| `scripts/run_calibration.py` | Entry point for Step 3. Parses `--config`, calls `run_calibration()`. No API calls. |
| `scripts/run_all.py` | Runs all three steps sequentially. Safe to re-run (caching skips completed work). |

## Setup

### Prerequisites

- Python 3.9+
- A [HuggingFace](https://huggingface.co/) account with an API token
- HuggingFace Pro subscription recommended (higher rate limits, access to larger models)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd suq-calibration

# Install dependencies
pip install -r requirements.txt
```

For the default **local backend**, PyTorch and Transformers are included in requirements. For the **API backend** only, these are optional.

### Environment Setup

```bash
# Copy the template and add your token
cp .env.example .env
# Edit .env and replace hf_your_token_here with your actual HuggingFace token
```

You can also set `HF_TOKEN` as an environment variable directly:

```bash
export HF_TOKEN=hf_your_actual_token
```

## Backend

The project supports two inference backends, configured via `backend` in `config.yaml`:

### Local Backend (`backend: "local"`)

Runs models locally using `transformers` on GPU (float16, `device_map="auto"`). No API token or credits needed. Models are downloaded from HuggingFace Hub on first use and cached.

**Requirements**: NVIDIA GPU with sufficient VRAM. `Qwen/Qwen2.5-7B-Instruct` fits in ~16GB. Set `api_concurrent_requests: 1` to avoid GPU contention.

```yaml
backend: "local"
generator_model: "Qwen/Qwen2.5-7B-Instruct"
evaluator_model: "Qwen/Qwen2.5-7B-Instruct"
api_concurrent_requests: 1
```

### API Backend (`backend: "api"`)

Uses the HuggingFace Inference API. Requires `HF_TOKEN` and API credits. Supports larger evaluator models (e.g., 72B) that may not fit locally.

```yaml
backend: "api"
generator_model: "Qwen/Qwen2.5-7B-Instruct"
evaluator_model: "Qwen/Qwen2.5-72B-Instruct"
api_concurrent_requests: 4
```

## Usage

### Full Pipeline

```bash
python scripts/run_all.py
```

This runs generation, similarity, and calibration sequentially. Each step checks its cache and skips completed work, so it is safe to re-run after interruptions.

### Individual Steps

Run each step independently for more control:

```bash
# Step 1: Generate K=10 responses per query
python scripts/run_generation.py

# Step 2: Compute pairwise similarity matrices
python scripts/run_similarity.py

# Step 3: Compute ECE and produce reliability diagrams (no API calls)
python scripts/run_calibration.py
```

### Custom Config

All scripts accept a `--config` flag:

```bash
python scripts/run_all.py --config config_test.yaml
```

### Recommended Workflow

For a cost-effective workflow, run ROUGE-L similarity first (instant, no API calls), then add LLM-judge later:

1. Edit `config.yaml`: comment out `"llm_judge"` from `similarity_methods`
2. Run the full pipeline: `python scripts/run_all.py`
3. Review ROUGE-L results in `results/`
4. Re-enable `"llm_judge"` in `config.yaml`
5. Run similarity and calibration again:
   ```bash
   python scripts/run_similarity.py   # Skips ROUGE-L (cached), runs LLM-judge
   python scripts/run_calibration.py  # Produces comparison plot with both methods
   ```

## Methodology

This section describes the theoretical framework from the paper and how it maps to the code.

### The Decision-Theoretic Framework

Standard calibration (Guo et al., 2017) applies to classification: a model outputs class probabilities, and calibration asks whether predicted confidence matches observed accuracy. For free-form text generation, there are no discrete classes -- the output space is the set of all possible strings.

Wang & Holmes (2024) resolve this by framing NLG as a **decision problem**. Given an input *I* (a question), a language model *p_M* defines a distribution over responses. The key insight is that everything -- uncertainty, the "best" response, and calibration -- is parameterized by a single **similarity function** *S(y', y; I)* that scores how well a candidate response *y'* serves when the "true" response is *y*.

### Subjective Utility and Bayes Risk

Given *K* samples *y_1, ..., y_K* from the language model, the **per-sample utility** of sample *k* as the chosen action is (Eq. 2 in the paper):

```
u_k = (1 / (K-1)) * sum_{j != k} S(y_k, y_j)
```

This measures how well sample *k* agrees with the other samples -- high agreement means the model is "confident" in that answer.

The **subjective utility** of the Gibbs predictor (randomly picking one sample) is:

```
subjective_utility = (1/K) * sum_k u_k
```

The **Bayes risk** is `R_B = 1 - subjective_utility`. High Bayes risk means high uncertainty (samples disagree).

The **Minimum Bayes Risk (MBR) generation** (Eq. 1) is the sample that maximizes expected utility:

```
y_MBR = y_{argmax_k u_k}
```

This is the "consensus" answer -- the sample most consistent with all others.

### Observed Utility

The **observed utility** measures actual performance against the ground-truth reference answer:

```
observed_utility = S(y_MBR, y_reference)
```

### Expected Calibration Error (ECE)

A model is **calibrated** if its subjective utility matches its observed utility in expectation (Eq. 4):

```
ECE(p_M) = E_s |f_M(s) - s|
```

where *f_M(s) = E[observed utility | subjective utility = s]*.

We estimate this via histogram binning (Naeini et al., 2015): partition [0, 1] into equal-width bins, compute the mean subjective and observed utilities in each bin, and take the weighted average of their absolute differences:

```
ECE = sum_b (n_b / N) * |mean_subjective_b - mean_observed_b|
```

Bootstrap resampling (1,000 iterations) provides 95% confidence intervals.

### Reliability Diagrams

Reliability diagrams (Figure 9 in the paper) visualize calibration by plotting observed utility against subjective utility per bin. A perfectly calibrated model follows the diagonal. Bars below the diagonal indicate **overconfidence** (the model thinks it's more certain than it actually is).

## Output Files

| File | Contents |
|------|----------|
| `results/generations.jsonl` | One JSON object per query: question, reference answers, and K generated samples. |
| `results/similarities_{method}.jsonl` | One JSON object per query: K x K pairwise similarity matrix and K-vector of reference similarities. |
| `results/reliability_{method}.png` | Reliability diagram (left) and query distribution histogram (right) for each similarity method. |
| `results/reliability_comparison.png` | Side-by-side reliability diagrams when multiple methods are run. |
| `results/metrics.json` | All numerical results: avg subjective/observed utility, ECE, bootstrap 95% CI, per method. Config is included for reproducibility. |

### JSONL Record Formats

**generations.jsonl:**
```json
{
  "query_idx": 0,
  "question": "What is the capital of France?",
  "context": null,
  "reference_answer": "Paris",
  "reference_answers": ["Paris", "paris"],
  "samples": ["Paris", "Paris", "The capital is Paris", ...]
}
```

**similarities_{method}.jsonl:**
```json
{
  "query_idx": 0,
  "method": "rouge_l",
  "pairwise": [[1.0, 0.8, ...], [0.8, 1.0, ...], ...],
  "vs_reference": [0.9, 0.85, ...]
}
```

## Configuration Reference

All parameters are set in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"local"` | Inference backend: `"local"` (GPU via transformers) or `"api"` (HF Inference API). |
| `generator_model` | `"Qwen/Qwen2.5-7B-Instruct"` | HF model for generating responses. Smaller models are cheaper and faster. |
| `evaluator_model` | `"Qwen/Qwen2.5-72B-Instruct"` | HF model for LLM-judge similarity. Larger models give more reliable judgments. |
| `dataset` | `"trivia_qa"` | Dataset to use. Supports `"trivia_qa"` and `"coqa"`. |
| `dataset_split` | `"validation"` | Which split to sample queries from. |
| `num_queries` | `300` | Number of questions to evaluate. Paper uses 1,000. |
| `seed` | `42` | Random seed for reproducibility (dataset shuffling, bootstrap). |
| `num_samples` | `10` | K: number of responses generated per query. Paper uses K=10. |
| `temperature` | `1.0` | Sampling temperature. 1.0 = pure ancestral sampling (matches paper). |
| `max_new_tokens` | `128` | Maximum response length in tokens. |
| `similarity_methods` | `["llm_judge", "rouge_l"]` | Which similarity functions to compute. Can include one or both. |
| `num_bins` | `10` | Number of equal-width bins for ECE histogram. |
| `num_bootstrap` | `1000` | Number of bootstrap iterations for confidence intervals. |
| `output_dir` | `"./results"` | Directory for all output files. |
| `generation_cache` | `"./results/generations.jsonl"` | Path to generation cache file. |
| `similarity_cache_dir` | `"./results"` | Directory for similarity cache files. |
| `api_max_retries` | `5` | Maximum retry attempts on API errors (429, 503). |
| `api_retry_delay` | `2.0` | Base delay in seconds for exponential backoff. |
| `api_concurrent_requests` | `4` | Maximum parallel API requests. |

### Model Alternatives

If a model is unavailable on the HF Inference API, swap it in `config.yaml`:

| Role | Good Options |
|------|-------------|
| Generator | `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` |
| Evaluator | `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct` |

## Caching and Resumability

Every expensive step (generation and similarity) writes results **incrementally** to JSONL files -- one JSON object per line, flushed immediately after each query. This design provides:

- **Crash recovery**: If the process is interrupted at query 200 out of 300, re-running the same script resumes from query 201. No work is lost.
- **Append-mode safety**: JSONL supports append-mode writing, so partial writes don't corrupt earlier results (unlike a single JSON file).
- **Independent caching per method**: Each similarity method has its own cache file (`similarities_rouge_l.jsonl`, `similarities_llm_judge.jsonl`), so you can run ROUGE-L first and add LLM-judge later without recomputing.

On startup, each step loads its cache, extracts the set of already-processed `query_idx` values, and skips those queries.

## Results

### ROUGE-L Calibration (300 queries, TriviaQA)

| Metric | Value |
|--------|-------|
| Generator | Qwen/Qwen2.5-7B-Instruct |
| Num queries | 300 |
| K (samples per query) | 10 |
| Avg subjective utility | 0.653 |
| Avg observed utility | 0.501 |
| **ECE** | **0.168** |
| ECE 95% CI | [0.141, 0.223] |

### Reliability Diagram

The reliability diagram shows clear **overconfidence**: bars consistently fall below the perfect-calibration diagonal, especially in the middle-to-high subjective utility range (0.4-0.8). The model's self-assessed certainty (subjective utility) systematically exceeds its actual performance (observed utility).

The query distribution histogram shows concentration at the extremes -- many queries cluster near subjective utility 1.0 (high agreement among samples, often correct) and a spread of queries in the 0.0-0.5 range (low agreement, more uncertain).

The gap between avg subjective utility (0.653) and avg observed utility (0.501) quantifies the overall overconfidence: the model believes it performs ~15 percentage points better than it actually does.

## The Role of the Similarity Measure *S*

The paper's central insight is that **everything is parameterized by the choice of *S***. Different similarity functions define different notions of "correctness" and therefore different uncertainty landscapes.

### ROUGE-L (Lexical Similarity)

ROUGE-L measures the longest common subsequence between two texts, normalized by length. It captures **surface-level lexical overlap**: two answers are similar if they share many words in sequence.

**Strengths**: Fast (local computation, no API calls), deterministic, well-understood.

**Limitations**: Purely lexical -- "Paris" and "The capital of France is Paris" get a low score despite being semantically equivalent. "44th" and "forty-fourth" score zero. This means ROUGE-L underestimates true similarity for semantically equivalent but lexically different responses, potentially inflating Bayes risk.

### LLM-Judge (Semantic Similarity)

The LLM-judge approach uses a large language model (72B parameters) to evaluate whether two answers are consistent, producing a 0-1 score. It captures **semantic equivalence**: it can recognize that "George Washington" and "The first US president, Washington" convey the same information.

**Strengths**: Semantically aware, handles paraphrases and variations in phrasing.

**Limitations**: Requires API calls (expensive, slow -- ~16,500 calls for 300 queries), slightly non-deterministic (temperature=0.01), and judge quality depends on the evaluator model's capabilities.

### How *S* Affects Calibration

Under ROUGE-L, the model appears more uncertain (lower subjective utility) because lexically different but correct answers are scored as dissimilar. Under LLM-judge, the model appears more confident because semantically equivalent answers are properly recognized. This means:

- ECE values are not directly comparable across similarity methods
- Each *S* defines its own notion of calibration
- The "right" *S* depends on the downstream application

This is exactly the paper's point: there is no single "correct" calibration for NLG. Calibration is always relative to the utility function.

## Known Limitations

1. **HF API credits**: The LLM-judge similarity step requires ~16,500 API calls to the evaluator model for 300 queries. With a 72B model, this can exhaust monthly HF Pro credits. Monitor your usage or start with fewer queries.

2. **ROUGE-L is lexical only**: It cannot capture semantic equivalence, which limits its usefulness as a similarity measure for free-form QA. It serves as a fast baseline but should not be the sole evaluation.

3. **Single reference answer**: TriviaQA provides multiple valid answer aliases, but we compare against only the primary reference answer. Some correct MBR generations may score low observed utility if they match an alias rather than the primary form.

4. **Model availability**: HF Inference API model availability can change. If a model returns 503 errors persistently, switch to an alternative in `config.yaml`.

5. **Sample size**: We use 300 queries (paper uses 1,000). Confidence intervals are wider with fewer queries. Scale up `num_queries` for tighter estimates.

6. **Evaluator reliability**: The LLM-judge's quality depends on the evaluator model. A weaker evaluator may produce noisier or less accurate similarity scores.

## References

- Wang, Z. & Holmes, T.P. (2024). *On Subjective Uncertainty Quantification and Calibration in Natural Language Generation*. arXiv:2406.05213v2. [[paper]](https://arxiv.org/abs/2406.05213v2)
- Naeini, M.P., Cooper, G.F., & Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning*. AAAI.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Eikema, B. & Aziz, W. (2022). *Sampling-Based Approximations to Minimum Bayes Risk Decoding for Neural Machine Translation*. EACL.

## License

This project is a research reproduction for educational purposes.
