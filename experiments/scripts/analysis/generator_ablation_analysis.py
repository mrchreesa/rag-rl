"""
Generator Ablation Analysis: Llama vs GPT-4o-mini

Compares RL policies trained with different generators:
1. Side-by-side F1 bar chart across all configurations
2. TopK distribution comparison between RL policies
3. "Does the weaker generator need more retrieval?" analysis

Requires: Run controlled_comparison.py with both --use-ollama and without.

Usage:
    python experiments/scripts/analysis/generator_ablation_analysis.py \
        --gpt-results experiments/results/controlled_comparison_custom_YYYYMMDD/ \
        --ollama-results experiments/results/controlled_comparison_custom_ollama_YYYYMMDD/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_comparison_results(results_dir: Path) -> dict:
    """Load comparison_results.json from a controlled comparison directory."""
    path = results_dir / "comparison_results.json"
    with open(path) as f:
        return json.load(f)


def load_predictions(path: str) -> list:
    """Load per-sample predictions from JSONL."""
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds


def create_f1_comparison_chart(gpt_results: dict, ollama_results: dict, output_dir: Path):
    """Create side-by-side F1 bar chart comparing generators across configs."""
    gpt_configs = gpt_results.get("configs", gpt_results)
    ollama_configs = ollama_results.get("configs", ollama_results)

    if isinstance(gpt_configs, list):
        gpt_by_name = {r["config"]: r for r in gpt_configs}
    else:
        gpt_by_name = gpt_configs

    if isinstance(ollama_configs, list):
        ollama_by_name = {r["config"]: r for r in ollama_configs}
    else:
        ollama_by_name = ollama_configs

    # Get common config names
    config_names = sorted(set(gpt_by_name.keys()) & set(ollama_by_name.keys()))
    if not config_names:
        # Try matching by position
        config_names = [r["config"] for r in (gpt_configs if isinstance(gpt_configs, list) else [])]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(config_names))
    width = 0.35

    gpt_f1s = [gpt_by_name[c]["avg_f1"] * 100 for c in config_names]
    ollama_f1s = [ollama_by_name[c]["avg_f1"] * 100 for c in config_names]

    bars1 = ax.bar(x - width / 2, gpt_f1s, width, label="GPT-4o-mini",
                   color="#4CAF50", edgecolor="gray", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ollama_f1s, width, label="Llama 3.1 8B (Ollama)",
                   color="#FF9800", edgecolor="gray", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Generator Ablation: GPT-4o-mini vs Llama 3.1 8B")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in config_names], fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(max(gpt_f1s), max(ollama_f1s)) * 1.15)

    plt.tight_layout()
    fig.savefig(output_dir / "generator_ablation_f1.pdf")
    fig.savefig(output_dir / "generator_ablation_f1.png")
    plt.close(fig)
    print(f"Saved: generator_ablation_f1.pdf")


def create_topk_distribution_comparison(gpt_dir: Path, ollama_dir: Path, output_dir: Path):
    """Compare TopK distributions between RL policies trained with different generators."""
    gpt_rl_path = gpt_dir / "predictions_RL_Dynamic_TopK.jsonl"
    ollama_rl_path = ollama_dir / "predictions_RL_Dynamic_TopK.jsonl"

    if not gpt_rl_path.exists() or not ollama_rl_path.exists():
        print("Warning: RL prediction files not found for both generators")
        return

    gpt_preds = load_predictions(str(gpt_rl_path))
    ollama_preds = load_predictions(str(ollama_rl_path))

    gpt_topks = [p["topk_used"] for p in gpt_preds]
    ollama_topks = [p["topk_used"] for p in ollama_preds]

    all_k_values = sorted(set(gpt_topks + ollama_topks))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: TopK distribution comparison
    ax1 = axes[0]
    from collections import Counter
    gpt_counts = Counter(gpt_topks)
    ollama_counts = Counter(ollama_topks)

    x = np.arange(len(all_k_values))
    width = 0.35

    gpt_pcts = [gpt_counts.get(k, 0) / len(gpt_topks) * 100 for k in all_k_values]
    ollama_pcts = [ollama_counts.get(k, 0) / len(ollama_topks) * 100 for k in all_k_values]

    ax1.bar(x - width / 2, gpt_pcts, width, label="GPT-4o-mini RL",
            color="#4CAF50", edgecolor="gray", linewidth=0.5)
    ax1.bar(x + width / 2, ollama_pcts, width, label="Llama RL",
            color="#FF9800", edgecolor="gray", linewidth=0.5)

    ax1.set_xlabel("Chosen TopK")
    ax1.set_ylabel("% of Queries")
    ax1.set_title("RL TopK Distribution by Generator")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in all_k_values])
    ax1.legend()

    # Right: F1 by TopK bucket comparison
    ax2 = axes[1]
    gpt_f1_by_k = defaultdict(list)
    ollama_f1_by_k = defaultdict(list)
    for p in gpt_preds:
        gpt_f1_by_k[p["topk_used"]].append(p["f1"])
    for p in ollama_preds:
        ollama_f1_by_k[p["topk_used"]].append(p["f1"])

    gpt_mean_f1 = [np.mean(gpt_f1_by_k[k]) * 100 if k in gpt_f1_by_k else 0 for k in all_k_values]
    ollama_mean_f1 = [np.mean(ollama_f1_by_k[k]) * 100 if k in ollama_f1_by_k else 0 for k in all_k_values]

    ax2.bar(x - width / 2, gpt_mean_f1, width, label="GPT-4o-mini RL",
            color="#4CAF50", edgecolor="gray", linewidth=0.5)
    ax2.bar(x + width / 2, ollama_mean_f1, width, label="Llama RL",
            color="#FF9800", edgecolor="gray", linewidth=0.5)

    ax2.set_xlabel("Chosen TopK")
    ax2.set_ylabel("Mean F1 (%)")
    ax2.set_title("F1 by TopK Bucket — Generator Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in all_k_values])
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "generator_ablation_topk_dist.pdf")
    fig.savefig(output_dir / "generator_ablation_topk_dist.png")
    plt.close(fig)
    print(f"Saved: generator_ablation_topk_dist.pdf")


def retrieval_benefit_table(gpt_results: dict, ollama_results: dict):
    """Print a table showing whether weaker generator benefits more from retrieval."""
    print("\n" + "=" * 60)
    print("RETRIEVAL BENEFIT ANALYSIS")
    print("Does the weaker generator benefit more from retrieval?")
    print("=" * 60)

    gpt_configs = gpt_results.get("configs", gpt_results)
    ollama_configs = ollama_results.get("configs", ollama_results)

    if isinstance(gpt_configs, list):
        gpt_by_name = {r["config"]: r for r in gpt_configs}
    else:
        gpt_by_name = gpt_configs

    if isinstance(ollama_configs, list):
        ollama_by_name = {r["config"]: r for r in ollama_configs}
    else:
        ollama_by_name = ollama_configs

    # Parametric vs best fixed-k
    gpt_k0 = gpt_by_name.get("k=0 (parametric)", {}).get("avg_f1", 0)
    ollama_k0 = ollama_by_name.get("k=0 (parametric)", {}).get("avg_f1", 0)

    # Find best fixed k
    gpt_best_fixed = max(
        (r for r in (gpt_configs if isinstance(gpt_configs, list) else []) if "fixed" in r.get("config", "")),
        key=lambda r: r["avg_f1"],
        default={"avg_f1": 0, "config": "N/A"}
    )
    ollama_best_fixed = max(
        (r for r in (ollama_configs if isinstance(ollama_configs, list) else []) if "fixed" in r.get("config", "")),
        key=lambda r: r["avg_f1"],
        default={"avg_f1": 0, "config": "N/A"}
    )

    gpt_rl = gpt_by_name.get("RL Dynamic TopK", {"avg_f1": 0, "avg_topk": 0})
    ollama_rl = ollama_by_name.get("RL Dynamic TopK", {"avg_f1": 0, "avg_topk": 0})

    print(f"\n{'Metric':<35}{'GPT-4o-mini':<18}{'Llama 3.1 8B'}")
    print("-" * 70)
    print(f"{'Parametric (k=0) F1':<35}{gpt_k0*100:<18.2f}{ollama_k0*100:.2f}")
    print(f"{'Best Fixed-K F1':<35}{gpt_best_fixed['avg_f1']*100:<18.2f}{ollama_best_fixed['avg_f1']*100:.2f}")
    print(f"{'Best Fixed-K Config':<35}{gpt_best_fixed['config']:<18}{ollama_best_fixed['config']}")
    print(f"{'Retrieval Benefit (best-k0)':<35}{(gpt_best_fixed['avg_f1']-gpt_k0)*100:<+18.2f}{(ollama_best_fixed['avg_f1']-ollama_k0)*100:+.2f}")
    print(f"{'RL Dynamic TopK F1':<35}{gpt_rl.get('avg_f1',0)*100:<18.2f}{ollama_rl.get('avg_f1',0)*100:.2f}")
    print(f"{'RL Avg TopK':<35}{gpt_rl.get('avg_topk',0):<18.1f}{ollama_rl.get('avg_topk',0):.1f}")
    print(f"{'RL vs Best Fixed':<35}{(gpt_rl.get('avg_f1',0)-gpt_best_fixed['avg_f1'])*100:<+18.2f}{(ollama_rl.get('avg_f1',0)-ollama_best_fixed['avg_f1'])*100:+.2f}")

    return {
        "gpt_k0": float(gpt_k0),
        "ollama_k0": float(ollama_k0),
        "gpt_best_fixed": float(gpt_best_fixed["avg_f1"]),
        "ollama_best_fixed": float(ollama_best_fixed["avg_f1"]),
        "gpt_retrieval_benefit": float(gpt_best_fixed["avg_f1"] - gpt_k0),
        "ollama_retrieval_benefit": float(ollama_best_fixed["avg_f1"] - ollama_k0),
    }


def main():
    parser = argparse.ArgumentParser(description="Generator ablation analysis")
    parser.add_argument("--gpt-results", type=str, required=True,
                        help="Path to GPT-4o-mini controlled comparison results dir")
    parser.add_argument("--ollama-results", type=str, required=True,
                        help="Path to Ollama controlled comparison results dir")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    gpt_dir = Path(args.gpt_results)
    if not gpt_dir.is_absolute():
        gpt_dir = project_root / gpt_dir

    ollama_dir = Path(args.ollama_results)
    if not ollama_dir.is_absolute():
        ollama_dir = project_root / ollama_dir

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATOR ABLATION ANALYSIS")
    print("GPT-4o-mini vs Llama 3.1 8B (Ollama)")
    print("=" * 60)
    print(f"GPT results: {gpt_dir}")
    print(f"Ollama results: {ollama_dir}")

    # Load results
    gpt_results = load_comparison_results(gpt_dir)
    ollama_results = load_comparison_results(ollama_dir)

    # Create figures
    create_f1_comparison_chart(gpt_results, ollama_results, output_dir)
    create_topk_distribution_comparison(gpt_dir, ollama_dir, output_dir)

    # Analysis table
    benefit_analysis = retrieval_benefit_table(gpt_results, ollama_results)

    # Save analysis
    analysis = {
        "gpt_dir": str(gpt_dir),
        "ollama_dir": str(ollama_dir),
        "retrieval_benefit": benefit_analysis
    }
    with open(output_dir / "generator_ablation_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {output_dir / 'generator_ablation_analysis.json'}")


if __name__ == "__main__":
    main()
