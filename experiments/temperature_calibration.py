"""
Temperature calibration experiment.

Sweeps sampling temperature and measures how ECE, Bayes risk,
and observed utility change — using the existing pipeline modules.

Usage:
    python experiments/temperature_calibration.py
"""

import copy
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.generate
import src.similarity
from src.calibration import (
    bootstrap_ece,
    compute_ece,
    compute_observed_utility,
    compute_subjective_uncertainty,
)
from src.generate import run_generation
from src.similarity import run_similarity
from src.utils import (
    JsonlCache,
    LocalModelClient,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)

# ── Experiment settings ──────────────────────────────────────────────
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
NUM_QUERIES = 100
SIMILARITY_METHODS = ["rouge_l", "llm_judge"]
OUTPUT_ROOT = "./results_temp"
RELIABILITY_TEMPS = [0.3, 0.7, 1.0, 1.5]

# ── Colors ───────────────────────────────────────────────────────────
METHOD_COLORS = {"rouge_l": "#2c7bb6", "llm_judge": "#d7191c"}
METHOD_LABELS = {"rouge_l": "ROUGE-L", "llm_judge": "LLM Judge"}


def make_config_for_temp(base_config: dict, temp: float) -> dict:
    """Deep-copy config and set temperature + per-temperature output paths."""
    cfg = copy.deepcopy(base_config)
    cfg["temperature"] = temp
    cfg["num_queries"] = NUM_QUERIES

    temp_dir = f"{OUTPUT_ROOT}/T{temp}"
    cfg["output_dir"] = temp_dir
    cfg["generation_cache"] = f"{temp_dir}/generations.jsonl"
    cfg["similarity_cache_dir"] = temp_dir

    return cfg


def collect_metrics_for_method(
    sim_records: list[dict], num_bins: int, num_bootstrap: int, seed: int,
) -> dict:
    """Compute calibration metrics from similarity records."""
    subjective_utilities = []
    observed_utilities = []

    for rec in sim_records:
        pairwise = np.array(rec["pairwise"])
        vs_ref = np.array(rec["vs_reference"])
        su = compute_subjective_uncertainty(pairwise)
        obs = compute_observed_utility(vs_ref, su["mbr_idx"])
        subjective_utilities.append(su["subjective_utility"])
        observed_utilities.append(obs)

    subj_arr = np.array(subjective_utilities)
    obs_arr = np.array(observed_utilities)

    cal_data = compute_ece(subj_arr, obs_arr, num_bins)
    boot = bootstrap_ece(subj_arr, obs_arr, num_bins, num_bootstrap, seed)

    return {
        "avg_subjective_utility": float(subj_arr.mean()),
        "avg_observed_utility": float(obs_arr.mean()),
        "ece": cal_data["ece"],
        "ece_ci": [boot["ci_low"], boot["ci_high"]],
        "num_queries": len(subj_arr),
        # Keep raw arrays for reliability diagrams
        "_cal_data": cal_data,
        "_subj_arr": subj_arr,
        "_obs_arr": obs_arr,
    }


# ── Plotting ─────────────────────────────────────────────────────────

def plot_ece_vs_temperature(all_metrics: dict, fig_dir: Path):
    """ECE vs Temperature with bootstrap CI error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in SIMILARITY_METHODS:
        temps, eces, ci_lo, ci_hi = [], [], [], []
        for t in TEMPERATURES:
            key = str(t)
            if key not in all_metrics or method not in all_metrics[key]:
                continue
            m = all_metrics[key][method]
            temps.append(t)
            eces.append(m["ece"])
            ci_lo.append(m["ece"] - m["ece_ci"][0])
            ci_hi.append(m["ece_ci"][1] - m["ece"])
        if not temps:
            continue
        ax.errorbar(
            temps, eces, yerr=[ci_lo, ci_hi],
            marker="o", capsize=4, linewidth=2,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("ECE", fontsize=12)
    ax.set_title("Expected Calibration Error vs Temperature", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ece_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_utility_vs_temperature(all_metrics: dict, fig_dir: Path, key: str, ylabel: str, title: str, fname: str):
    """Generic line plot of a metric vs temperature, per similarity method."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in SIMILARITY_METHODS:
        temps, vals = [], []
        for t in TEMPERATURES:
            tk = str(t)
            if tk not in all_metrics or method not in all_metrics[tk]:
                continue
            temps.append(t)
            vals.append(all_metrics[tk][method][key])
        if not temps:
            continue
        ax.plot(
            temps, vals, marker="o", linewidth=2,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)


def plot_utility_gap_vs_temperature(all_metrics: dict, fig_dir: Path):
    """Utility gap (subjective - observed) vs temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in SIMILARITY_METHODS:
        temps, gaps = [], []
        for t in TEMPERATURES:
            tk = str(t)
            if tk not in all_metrics or method not in all_metrics[tk]:
                continue
            m = all_metrics[tk][method]
            temps.append(t)
            gaps.append(m["avg_subjective_utility"] - m["avg_observed_utility"])
        if not temps:
            continue
        ax.plot(
            temps, gaps, marker="o", linewidth=2,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Utility Gap (Subjective - Observed)", fontsize=12)
    ax.set_title("Overconfidence Gap vs Temperature", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "utility_gap_vs_temperature.png", dpi=150)
    plt.close(fig)


def plot_reliability_grid(all_metrics: dict, fig_dir: Path):
    """2x2 reliability diagrams for selected temperatures, per similarity method."""
    for method in SIMILARITY_METHODS:
        available = [t for t in RELIABILITY_TEMPS if str(t) in all_metrics and method in all_metrics[str(t)]]
        if not available:
            continue

        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 9))
        axes_flat = axes.flatten()

        for idx, t in enumerate(available[:4]):
            ax = axes_flat[idx]
            cal = all_metrics[str(t)][method]["_cal_data"]
            bin_edges = cal["bin_edges"]
            bin_obs = cal["bin_observed_means"]
            bin_counts = cal["bin_counts"]
            ece = cal["ece"]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = bin_edges[1] - bin_edges[0]
            mask = bin_counts > 0

            ax.bar(
                bin_centers[mask], bin_obs[mask], width=width * 0.8,
                alpha=0.7, color=METHOD_COLORS[method], edgecolor="black",
            )
            ax.plot([0, 1], [0, 1], "r--", linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"T={t}  (ECE={ece:.4f})", fontsize=11)
            ax.set_xlabel("Subjective Utility", fontsize=10)
            ax.set_ylabel("Observed Utility", fontsize=10)

        # Hide unused subplots
        for idx in range(len(available), 4):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"Reliability Diagrams — {METHOD_LABELS[method]}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_dir / f"reliability_grid_{method}.png", dpi=150)
        plt.close(fig)


def generate_all_plots(all_metrics: dict, fig_dir: Path):
    """Generate all plots."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_ece_vs_temperature(all_metrics, fig_dir)
    plot_utility_vs_temperature(
        all_metrics, fig_dir,
        key="avg_subjective_utility",
        ylabel="Avg Subjective Utility",
        title="Avg Subjective Utility vs Temperature",
        fname="subjective_utility_vs_temperature.png",
    )
    plot_utility_vs_temperature(
        all_metrics, fig_dir,
        key="avg_observed_utility",
        ylabel="Avg Observed Utility",
        title="Avg Observed Utility vs Temperature",
        fname="observed_utility_vs_temperature.png",
    )
    plot_utility_gap_vs_temperature(all_metrics, fig_dir)
    plot_reliability_grid(all_metrics, fig_dir)
    logger.info(f"All plots saved to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    setup_logging()
    base_config = load_config()
    base_config["backend"] = "local"
    base_config["num_queries"] = NUM_QUERIES

    # Share a single LocalModelClient so the model stays loaded in GPU memory.
    shared_client = LocalModelClient()
    src.generate.get_hf_client = lambda _cfg: shared_client
    src.similarity.get_hf_client = lambda _cfg: shared_client

    num_bins = base_config.get("num_bins", 10)
    num_bootstrap = base_config.get("num_bootstrap", 1000)
    seed = base_config.get("seed", 42)

    all_metrics = {}  # temp_str -> method -> metrics dict

    for sim_method in SIMILARITY_METHODS:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Similarity method: {sim_method}")
        logger.info(f"{'#' * 60}")

        for temp in TEMPERATURES:
            temp_str = str(temp)
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Temperature = {temp}  |  Similarity = {sim_method}")
            logger.info(f"{'=' * 60}")

            cfg = make_config_for_temp(base_config, temp)
            cfg["similarity_methods"] = [sim_method]

            set_seed(seed)

            # Generation (cached across similarity methods — same temp dir)
            run_generation(cfg)

            # Similarity
            run_similarity(cfg)

            # Collect metrics
            cache_path = f"{cfg['similarity_cache_dir']}/similarities_{sim_method}.jsonl"
            sim_records = JsonlCache(cache_path).load()
            if not sim_records:
                logger.warning(f"No similarity data for T={temp}, {sim_method}")
                continue

            metrics = collect_metrics_for_method(sim_records, num_bins, num_bootstrap, seed)

            if temp_str not in all_metrics:
                all_metrics[temp_str] = {}
            all_metrics[temp_str][sim_method] = metrics

            logger.info(
                f"  ECE={metrics['ece']:.4f}  "
                f"CI=[{metrics['ece_ci'][0]:.4f}, {metrics['ece_ci'][1]:.4f}]  "
                f"Subj={metrics['avg_subjective_utility']:.4f}  "
                f"Obs={metrics['avg_observed_utility']:.4f}  "
                f"n={metrics['num_queries']}"
            )

    # Save JSON-serializable metrics
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    json_metrics = {}
    for temp_str, methods in all_metrics.items():
        json_metrics[temp_str] = {}
        for method, m in methods.items():
            json_metrics[temp_str][method] = {
                k: v for k, v in m.items() if not k.startswith("_")
            }

    metrics_path = output_root / "temperature_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Generate plots
    generate_all_plots(all_metrics, output_root / "figures")

    logger.info("Temperature calibration experiment complete.")


if __name__ == "__main__":
    main()
