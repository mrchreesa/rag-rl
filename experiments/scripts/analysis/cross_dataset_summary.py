"""
Cross-Dataset Summary

Creates a master results table combining:
- Custom dataset results
- HotpotQA results
- Generator ablation results
- Statistical significance indicators
- Confidence intervals (if available)

Usage:
    python experiments/scripts/analysis/cross_dataset_summary.py \
        --custom-results experiments/results/controlled_comparison_custom_YYYYMMDD/ \
        --hotpotqa-results experiments/results/controlled_comparison_hotpotqa_YYYYMMDD/ \
        [--ollama-results experiments/results/controlled_comparison_custom_ollama_YYYYMMDD/] \
        [--stats-file experiments/results/figures/statistical_analysis.json]
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
    """Load comparison_results.json."""
    path = results_dir / "comparison_results.json"
    with open(path) as f:
        return json.load(f)


def extract_config_table(results: dict) -> list:
    """Extract list of config dicts from results format."""
    configs = results.get("configs", results)
    if isinstance(configs, list):
        return configs
    return []


def print_master_table(datasets: dict, stats: dict = None):
    """Print master results table across datasets."""
    print("\n" + "=" * 80)
    print("MASTER RESULTS TABLE")
    print("=" * 80)

    # Collect all config names
    all_configs = set()
    for ds_name, ds_results in datasets.items():
        configs = extract_config_table(ds_results)
        for c in configs:
            all_configs.add(c["config"])

    # Standard order
    config_order = [
        "k=0 (parametric)",
        "k=1 (fixed)",
        "k=3 (fixed)",
        "k=5 (fixed)",
        "k=10 (fixed)",
        "RL Dynamic TopK",
        "RL Dynamic TopK + Learned Rewrite",
    ]
    config_order = [c for c in config_order if c in all_configs]
    # Add any remaining
    for c in sorted(all_configs):
        if c not in config_order:
            config_order.append(c)

    ds_names = list(datasets.keys())
    header = f"{'Configuration':<25}" + "".join(f"{ds:<20}" for ds in ds_names)
    print(f"\n{header}")
    print("-" * (25 + 20 * len(ds_names)))

    master_data = {}
    for config in config_order:
        row = f"{config:<25}"
        config_data = {}
        for ds_name in ds_names:
            ds_results = datasets[ds_name]
            configs = extract_config_table(ds_results)
            config_dict = {c["config"]: c for c in configs}
            if config in config_dict:
                f1 = config_dict[config]["avg_f1"] * 100
                em = config_dict[config]["avg_em"] * 100

                # Check for significance marker
                sig = ""
                if stats and f"{config}" in str(stats):
                    sig = "*"

                row += f"{f1:>6.2f}% (EM: {em:>5.2f}%){sig:<2}"
                config_data[ds_name] = {"f1": f1, "em": em}
            else:
                row += f"{'N/A':<20}"
        print(row)
        master_data[config] = config_data

    return master_data


def create_cross_dataset_figure(datasets: dict, output_dir: Path):
    """Create a grouped bar chart comparing F1 across datasets."""
    ds_names = list(datasets.keys())
    n_datasets = len(ds_names)

    # Collect configs
    all_configs_set = set()
    for ds_results in datasets.values():
        configs = extract_config_table(ds_results)
        for c in configs:
            all_configs_set.add(c["config"])

    config_order = [
        "k=0 (parametric)", "k=1 (fixed)", "k=3 (fixed)",
        "k=5 (fixed)", "k=10 (fixed)", "RL Dynamic TopK",
        "RL Dynamic TopK + Learned Rewrite"
    ]
    config_order = [c for c in config_order if c in all_configs_set]
    for c in sorted(all_configs_set):
        if c not in config_order:
            config_order.append(c)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(config_order))
    width = 0.8 / n_datasets
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for i, ds_name in enumerate(ds_names):
        ds_results = datasets[ds_name]
        configs = extract_config_table(ds_results)
        config_dict = {c["config"]: c for c in configs}

        f1s = []
        for config in config_order:
            if config in config_dict:
                f1s.append(config_dict[config]["avg_f1"] * 100)
            else:
                f1s.append(0)

        offset = (i - n_datasets / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width,
                      label=ds_name, color=colors[i % len(colors)],
                      edgecolor="gray", linewidth=0.5)

        # Value labels
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Cross-Dataset RAG Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in config_order], fontsize=8)
    ax.legend(loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / "cross_dataset_comparison.pdf")
    fig.savefig(output_dir / "cross_dataset_comparison.png")
    plt.close(fig)
    print(f"\nSaved: cross_dataset_comparison.pdf")


def create_retrieval_benefit_figure(datasets: dict, output_dir: Path):
    """Show retrieval benefit (best_fixed - k0) across datasets."""
    ds_names = []
    k0_f1s = []
    best_fixed_f1s = []
    rl_f1s = []

    for ds_name, ds_results in datasets.items():
        configs = extract_config_table(ds_results)
        config_dict = {c["config"]: c for c in configs}

        k0 = config_dict.get("k=0 (parametric)", {}).get("avg_f1", 0) * 100
        best_fixed = max(
            (c["avg_f1"] * 100 for c in configs if "fixed" in c.get("config", "")),
            default=0
        )
        # Try Phase 2 name first, then Phase 1
        rl_config = config_dict.get("RL Dynamic TopK + Learned Rewrite",
                                    config_dict.get("RL Dynamic TopK", {}))
        rl = rl_config.get("avg_f1", 0) * 100

        ds_names.append(ds_name)
        k0_f1s.append(k0)
        best_fixed_f1s.append(best_fixed)
        rl_f1s.append(rl)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ds_names))
    width = 0.25

    ax.bar(x - width, k0_f1s, width, label="Parametric (k=0)", color="#E57373")
    ax.bar(x, best_fixed_f1s, width, label="Best Fixed-K", color="#64B5F6")
    ax.bar(x + width, rl_f1s, width, label="RL Agent (Best)", color="#81C784")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Retrieval Benefit Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "retrieval_benefit_comparison.pdf")
    fig.savefig(output_dir / "retrieval_benefit_comparison.png")
    plt.close(fig)
    print(f"Saved: retrieval_benefit_comparison.pdf")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset summary")
    parser.add_argument("--custom-results", type=str, default=None,
                        help="Custom dataset results directory")
    parser.add_argument("--hotpotqa-results", type=str, default=None,
                        help="HotpotQA results directory")
    parser.add_argument("--ollama-results", type=str, default=None,
                        help="Ollama (generator ablation) results directory")
    parser.add_argument("--stats-file", type=str, default=None,
                        help="Statistical analysis JSON file")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/figures",
                        help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CROSS-DATASET SUMMARY")
    print("=" * 60)

    datasets = {}

    # Load available results
    if args.custom_results:
        path = Path(args.custom_results)
        if not path.is_absolute():
            path = project_root / path
        datasets["Custom (arXiv QA)"] = load_comparison_results(path)

    if args.hotpotqa_results:
        path = Path(args.hotpotqa_results)
        if not path.is_absolute():
            path = project_root / path
        datasets["HotpotQA"] = load_comparison_results(path)

    if args.ollama_results:
        path = Path(args.ollama_results)
        if not path.is_absolute():
            path = project_root / path
        datasets["Custom (Llama 3.1)"] = load_comparison_results(path)

    if not datasets:
        # Auto-detect from results directory
        results_base = project_root / "experiments/results"
        for d in sorted(results_base.glob("controlled_comparison_*")):
            if (d / "comparison_results.json").exists():
                results = load_comparison_results(d)
                ds_name = results.get("dataset", d.name)
                gen = results.get("generator", "gpt-4o-mini")
                label = f"{ds_name} ({gen})"
                datasets[label] = results
                print(f"  Auto-detected: {label} from {d.name}")

    if not datasets:
        print("ERROR: No results found. Provide --custom-results, --hotpotqa-results, etc.")
        return

    # Load stats if available
    stats = None
    if args.stats_file:
        stats_path = Path(args.stats_file)
        if not stats_path.is_absolute():
            stats_path = project_root / stats_path
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

    # Print and create figures
    master_data = print_master_table(datasets, stats)
    create_cross_dataset_figure(datasets, output_dir)
    create_retrieval_benefit_figure(datasets, output_dir)

    # Save summary
    summary = {
        "datasets": list(datasets.keys()),
        "master_data": master_data
    }
    with open(output_dir / "cross_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir / 'cross_dataset_summary.json'}")


if __name__ == "__main__":
    main()
