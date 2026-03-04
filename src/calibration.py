from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import JsonlCache, setup_logging

logger = logging.getLogger(__name__)


def compute_subjective_uncertainty(pairwise: np.ndarray) -> dict:
    """Compute Bayes risk for the Gibbs predictor from pairwise similarity matrix."""
    K = pairwise.shape[0]
    # u_k = mean of S(y_k, y_j) for j != k
    per_sample_utility = np.zeros(K)
    for k in range(K):
        per_sample_utility[k] = (pairwise[k].sum() - pairwise[k, k]) / (K - 1)

    subjective_utility = per_sample_utility.mean()
    bayes_risk = 1.0 - subjective_utility
    mbr_idx = int(np.argmax(per_sample_utility))

    return {
        "subjective_utility": float(subjective_utility),
        "bayes_risk": float(bayes_risk),
        "mbr_idx": mbr_idx,
        "per_sample_utility": per_sample_utility,
    }


def compute_observed_utility(vs_reference: np.ndarray, mbr_idx: int) -> float:
    """Return the observed utility: how well the MBR generation matches the reference."""
    return float(vs_reference[mbr_idx])


def compute_ece(
    subjective_utilities: np.ndarray,
    observed_utilities: np.ndarray,
    num_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error via histogram binning."""
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_subjective_means = np.zeros(num_bins)
    bin_observed_means = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == num_bins - 1:
            mask = (subjective_utilities >= low) & (subjective_utilities <= high)
        else:
            mask = (subjective_utilities >= low) & (subjective_utilities < high)

        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            bin_subjective_means[b] = subjective_utilities[mask].mean()
            bin_observed_means[b] = observed_utilities[mask].mean()

    total = len(subjective_utilities)
    ece = 0.0
    for b in range(num_bins):
        if bin_counts[b] > 0:
            ece += (bin_counts[b] / total) * abs(
                bin_subjective_means[b] - bin_observed_means[b]
            )

    return {
        "ece": float(ece),
        "bin_edges": bin_edges,
        "bin_subjective_means": bin_subjective_means,
        "bin_observed_means": bin_observed_means,
        "bin_counts": bin_counts,
    }


def bootstrap_ece(
    subjective_utilities: np.ndarray,
    observed_utilities: np.ndarray,
    num_bins: int = 10,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for ECE."""
    rng = np.random.RandomState(seed)
    n = len(subjective_utilities)
    ece_samples = []

    for _ in range(num_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        result = compute_ece(
            subjective_utilities[indices],
            observed_utilities[indices],
            num_bins,
        )
        ece_samples.append(result["ece"])

    ece_samples = np.array(ece_samples)
    return {
        "ece_mean": float(ece_samples.mean()),
        "ece_std": float(ece_samples.std()),
        "ci_low": float(np.percentile(ece_samples, 2.5)),
        "ci_high": float(np.percentile(ece_samples, 97.5)),
    }


def plot_reliability_diagram(cal_data: dict, title: str, save_path: str):
    """Plot reliability diagram with histogram of bin counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bin_edges = cal_data["bin_edges"]
    bin_subj = cal_data["bin_subjective_means"]
    bin_obs = cal_data["bin_observed_means"]
    bin_counts = cal_data["bin_counts"]
    ece = cal_data["ece"]

    # Reliability diagram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]
    mask = bin_counts > 0

    ax1.bar(
        bin_centers[mask], bin_obs[mask], width=width * 0.8,
        alpha=0.7, color="steelblue", edgecolor="black", label="Observed",
    )
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax1.set_xlabel("Subjective Utility", fontsize=12)
    ax1.set_ylabel("Observed Utility", fontsize=12)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=11)

    # Histogram of counts
    ax2.bar(
        bin_centers, bin_counts, width=width * 0.8,
        alpha=0.7, color="steelblue", edgecolor="black",
    )
    ax2.set_xlabel("Subjective Utility", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Queries", fontsize=13)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram to {save_path}")


def plot_comparison(cal_data_dict: dict, title: str, save_path: str):
    """Side-by-side reliability diagrams for multiple similarity methods."""
    methods = list(cal_data_dict.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        cal_data = cal_data_dict[method]
        bin_edges = cal_data["bin_edges"]
        bin_obs = cal_data["bin_observed_means"]
        bin_counts = cal_data["bin_counts"]
        ece = cal_data["ece"]

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]
        mask = bin_counts > 0

        ax.bar(
            bin_centers[mask], bin_obs[mask], width=width * 0.8,
            alpha=0.7, color="steelblue", edgecolor="black",
        )
        ax.plot([0, 1], [0, 1], "r--", linewidth=2)
        ax.set_xlabel("Subjective Utility", fontsize=12)
        ax.set_title(f"{method}\nECE = {ece:.4f}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Observed Utility", fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison plot to {save_path}")


def run_calibration(config: dict):
    """Main entry point: compute ECE, plot reliability diagrams, save metrics."""
    setup_logging()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    num_bins = config.get("num_bins", 10)
    num_bootstrap = config.get("num_bootstrap", 1000)
    seed = config.get("seed", 42)

    all_cal_data = {}
    all_metrics = {}

    for method in config["similarity_methods"]:
        cache_path = f"{config['similarity_cache_dir']}/similarities_{method}.jsonl"
        cache = JsonlCache(cache_path)
        sim_records = cache.load()

        if not sim_records:
            logger.warning(f"No similarity data for method '{method}', skipping.")
            continue

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

        all_cal_data[method] = cal_data

        # Plot individual reliability diagram
        plot_path = str(output_dir / f"reliability_{method}.png")
        plot_reliability_diagram(
            cal_data,
            f"Reliability Diagram ({method})",
            plot_path,
        )

        avg_subj = float(subj_arr.mean())
        avg_obs = float(obs_arr.mean())

        logger.info(
            f"\n{'='*50}\n"
            f"Method: {method}\n"
            f"  Num queries:            {len(subj_arr)}\n"
            f"  Avg subjective utility: {avg_subj:.4f}\n"
            f"  Avg observed utility:   {avg_obs:.4f}\n"
            f"  ECE:                    {cal_data['ece']:.4f}\n"
            f"  ECE 95% CI:            [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]\n"
            f"{'='*50}"
        )

        all_metrics[method] = {
            "avg_subjective_utility": avg_subj,
            "avg_observed_utility": avg_obs,
            "ece": cal_data["ece"],
            "ece_ci": [boot["ci_low"], boot["ci_high"]],
            "num_queries": len(subj_arr),
        }

    # Comparison plot if multiple methods
    if len(all_cal_data) > 1:
        comp_path = str(output_dir / "reliability_comparison.png")
        plot_comparison(all_cal_data, "Calibration Comparison", comp_path)

    # Save metrics
    metrics_out = {
        "config": {
            k: v for k, v in config.items()
            if k not in ("hf_token",)
        },
        "results": all_metrics,
    }
    metrics_path = str(output_dir / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
