"""
Per-Question TopK Analysis

Analyzes the RL agent's per-question topk decisions from the controlled comparison:
1. TopK vs Question Features (length, type, specificity)
2. Does per-question adaptation help vs fixed k=3?
3. F1 by chosen topk bucket
4. Scatter plot: question length vs chosen topk, colored by F1

Requires: Run controlled_comparison.py first to generate predictions.

Usage:
    python experiments/scripts/analysis/per_question_analysis.py \
        --results-dir experiments/results/controlled_comparison_YYYYMMDD_HHMMSS
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import argparse
import re
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


def load_predictions(path: str) -> list:
    """Load per-sample predictions from JSONL."""
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds


def classify_question_type(question: str) -> str:
    """Classify question by its leading word."""
    q = question.lower().strip()
    for prefix in ['what', 'who', 'when', 'where', 'which', 'how many', 'how much', 'how', 'why', 'is', 'are', 'does', 'do', 'can', 'could']:
        if q.startswith(prefix):
            if prefix in ('how many', 'how much'):
                return prefix
            return prefix
    return 'other'


def analyze_topk_vs_features(rl_preds: list):
    """Analysis 1: Correlate chosen topk with question features."""
    print("=" * 60)
    print("ANALYSIS 1: TopK vs Question Features")
    print("=" * 60)

    # Compute features for each question
    lengths = [len(p["question"]) for p in rl_preds]
    word_counts = [len(p["question"].split()) for p in rl_preds]
    topks = [p["topk_used"] for p in rl_preds]
    q_types = [classify_question_type(p["question"]) for p in rl_preds]

    # Correlation: question length vs topk
    corr_len = np.corrcoef(lengths, topks)[0, 1]
    corr_words = np.corrcoef(word_counts, topks)[0, 1]

    print(f"\nCorrelation (question length vs topk): {corr_len:+.4f}")
    print(f"Correlation (word count vs topk):      {corr_words:+.4f}")

    # TopK by question type
    type_topks = defaultdict(list)
    type_f1s = defaultdict(list)
    for p, qt in zip(rl_preds, q_types):
        type_topks[qt].append(p["topk_used"])
        type_f1s[qt].append(p["f1"])

    print(f"\n{'Q Type':<12}{'Count':<8}{'Avg TopK':<10}{'Avg F1':<10}{'Retr Rate'}")
    print("-" * 50)
    for qt in sorted(type_topks.keys(), key=lambda x: -len(type_topks[x])):
        n = len(type_topks[qt])
        avg_k = np.mean(type_topks[qt])
        avg_f1 = np.mean(type_f1s[qt])
        retr_rate = sum(1 for k in type_topks[qt] if k > 0) / n
        print(f"{qt:<12}{n:<8}{avg_k:<10.2f}{avg_f1:<10.4f}{retr_rate:.0%}")

    return {
        'corr_length_topk': corr_len,
        'corr_words_topk': corr_words,
        'type_summary': {qt: {'n': len(type_topks[qt]),
                              'avg_topk': float(np.mean(type_topks[qt])),
                              'avg_f1': float(np.mean(type_f1s[qt]))}
                         for qt in type_topks}
    }


def analyze_adaptation_benefit(rl_preds: list, fixed_k3_preds: list):
    """Analysis 2: Does per-question adaptation help vs fixed k=3?"""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-Question Adaptation Benefit")
    print("=" * 60)

    # Build lookup by question ID
    k3_by_id = {p["id"]: p for p in fixed_k3_preds}

    # Find questions where RL chose k != 3
    different_k = []
    same_k = []
    for p in rl_preds:
        pid = p["id"]
        if pid not in k3_by_id:
            continue
        k3_f1 = k3_by_id[pid]["f1"]
        rl_f1 = p["f1"]
        rl_k = p["topk_used"]

        if rl_k != 3:
            different_k.append({
                "id": pid,
                "question": p["question"][:80],
                "rl_topk": rl_k,
                "rl_f1": rl_f1,
                "k3_f1": k3_f1,
                "f1_diff": rl_f1 - k3_f1
            })
        else:
            same_k.append({
                "id": pid,
                "rl_f1": rl_f1,
                "k3_f1": k3_f1
            })

    n_different = len(different_k)
    n_total = len(rl_preds)
    pct_different = n_different / n_total * 100

    print(f"\nQueries where RL chose k != 3: {n_different}/{n_total} ({pct_different:.1f}%)")
    print(f"Queries where RL chose k = 3:  {len(same_k)}/{n_total} ({100-pct_different:.1f}%)")

    if different_k:
        # On the queries where RL diverged from k=3, how did it do?
        rl_f1_on_diff = np.mean([d["rl_f1"] for d in different_k])
        k3_f1_on_diff = np.mean([d["k3_f1"] for d in different_k])
        wins = sum(1 for d in different_k if d["f1_diff"] > 0)
        losses = sum(1 for d in different_k if d["f1_diff"] < 0)
        ties = sum(1 for d in different_k if d["f1_diff"] == 0)

        print(f"\nOn divergent queries (k != 3):")
        print(f"  RL F1:       {rl_f1_on_diff:.4f}")
        print(f"  Fixed k=3:   {k3_f1_on_diff:.4f}")
        print(f"  Difference:  {rl_f1_on_diff - k3_f1_on_diff:+.4f}")
        print(f"  Wins/Losses/Ties: {wins}/{losses}/{ties}")

        # Show individual divergent queries
        print(f"\nDivergent queries (sorted by F1 difference):")
        print(f"{'ID':<10}{'RL k':<6}{'RL F1':<8}{'k3 F1':<8}{'Diff':<8}Question")
        print("-" * 90)
        for d in sorted(different_k, key=lambda x: -x["f1_diff"]):
            print(f"{d['id']:<10}{d['rl_topk']:<6}{d['rl_f1']:<8.4f}"
                  f"{d['k3_f1']:<8.4f}{d['f1_diff']:+.4f}  {d['question']}")

    # Overall comparison
    all_rl_f1 = np.mean([p["f1"] for p in rl_preds])
    all_k3_f1 = np.mean([p["f1"] for p in fixed_k3_preds])
    print(f"\nOverall F1 comparison:")
    print(f"  RL Dynamic TopK: {all_rl_f1:.4f}")
    print(f"  Fixed k=3:       {all_k3_f1:.4f}")
    print(f"  Difference:      {all_rl_f1 - all_k3_f1:+.4f}")

    return {
        'n_different': n_different,
        'pct_different': pct_different,
        'rl_f1_on_divergent': float(np.mean([d["rl_f1"] for d in different_k])) if different_k else None,
        'k3_f1_on_divergent': float(np.mean([d["k3_f1"] for d in different_k])) if different_k else None,
    }


def analyze_f1_by_topk_bucket(rl_preds: list):
    """Analysis 3: F1 grouped by chosen topk bucket."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: F1 by Chosen TopK Bucket")
    print("=" * 60)

    buckets = defaultdict(list)
    for p in rl_preds:
        buckets[p["topk_used"]].append(p["f1"])

    print(f"\n{'TopK':<8}{'Count':<8}{'Mean F1':<10}{'Std F1':<10}{'Min':<8}{'Max'}")
    print("-" * 52)
    for k in sorted(buckets.keys()):
        f1s = buckets[k]
        print(f"k={k:<5}{len(f1s):<8}{np.mean(f1s):<10.4f}{np.std(f1s):<10.4f}"
              f"{np.min(f1s):<8.4f}{np.max(f1s):.4f}")

    return {k: {'count': len(v), 'mean_f1': float(np.mean(v)), 'std_f1': float(np.std(v))}
            for k, v in buckets.items()}


def analyze_hotpotqa_breakdowns(rl_preds: list, all_config_preds: dict):
    """Analysis 5: HotpotQA difficulty-level and question-type breakdowns."""
    # Check if metadata is present
    has_level = any(p.get("metadata", {}).get("level") for p in rl_preds)
    has_type = any(p.get("metadata", {}).get("type") for p in rl_preds)

    if not has_level and not has_type:
        print("\n(No HotpotQA metadata found — skipping difficulty/type breakdowns)")
        return {}

    results = {}

    if has_level:
        print("\n" + "=" * 60)
        print("ANALYSIS 5a: HotpotQA Difficulty Level Breakdown")
        print("=" * 60)

        # RL breakdown by difficulty
        level_data = defaultdict(lambda: {"topks": [], "f1s": []})
        for p in rl_preds:
            level = p.get("metadata", {}).get("level", "unknown")
            level_data[level]["topks"].append(p["topk_used"])
            level_data[level]["f1s"].append(p["f1"])

        print(f"\nRL Dynamic TopK — Breakdown by Difficulty:")
        print(f"{'Level':<12}{'Count':<8}{'Avg TopK':<10}{'Avg F1':<10}{'Retr Rate'}")
        print("-" * 50)
        level_summary = {}
        for level in ["easy", "medium", "hard", "unknown"]:
            if level not in level_data:
                continue
            d = level_data[level]
            n = len(d["topks"])
            avg_k = np.mean(d["topks"])
            avg_f1 = np.mean(d["f1s"])
            retr_rate = sum(1 for k in d["topks"] if k > 0) / n
            print(f"{level:<12}{n:<8}{avg_k:<10.2f}{avg_f1:<10.4f}{retr_rate:.0%}")
            level_summary[level] = {"n": n, "avg_topk": float(avg_k), "avg_f1": float(avg_f1)}

        # Cross-config comparison by difficulty level
        if all_config_preds:
            print(f"\nCross-config F1 by Difficulty Level:")
            configs = sorted(all_config_preds.keys())
            header = f"{'Level':<12}" + "".join(f"{c:<18}" for c in configs)
            print(header)
            print("-" * (12 + 18 * len(configs)))
            for level in ["easy", "medium", "hard"]:
                row = f"{level:<12}"
                for config in configs:
                    preds = all_config_preds[config]
                    level_f1s = [p["f1"] for p in preds if p.get("metadata", {}).get("level") == level]
                    if level_f1s:
                        row += f"{np.mean(level_f1s):<18.4f}"
                    else:
                        row += f"{'N/A':<18}"
                print(row)

        results["level_breakdown"] = level_summary

    if has_type:
        print("\n" + "=" * 60)
        print("ANALYSIS 5b: HotpotQA Question Type Breakdown")
        print("=" * 60)

        type_data = defaultdict(lambda: {"topks": [], "f1s": []})
        for p in rl_preds:
            qtype = p.get("metadata", {}).get("type", "unknown")
            type_data[qtype]["topks"].append(p["topk_used"])
            type_data[qtype]["f1s"].append(p["f1"])

        print(f"\nRL Dynamic TopK — Breakdown by Question Type:")
        print(f"{'Type':<15}{'Count':<8}{'Avg TopK':<10}{'Avg F1':<10}{'Retr Rate'}")
        print("-" * 53)
        type_summary = {}
        for qtype in sorted(type_data.keys(), key=lambda x: -len(type_data[x]["topks"])):
            d = type_data[qtype]
            n = len(d["topks"])
            avg_k = np.mean(d["topks"])
            avg_f1 = np.mean(d["f1s"])
            retr_rate = sum(1 for k in d["topks"] if k > 0) / n
            print(f"{qtype:<15}{n:<8}{avg_k:<10.2f}{avg_f1:<10.4f}{retr_rate:.0%}")
            type_summary[qtype] = {"n": n, "avg_topk": float(avg_k), "avg_f1": float(avg_f1)}

        # Cross-config comparison by type
        if all_config_preds:
            print(f"\nCross-config F1 by Question Type:")
            configs = sorted(all_config_preds.keys())
            header = f"{'Type':<15}" + "".join(f"{c:<18}" for c in configs)
            print(header)
            print("-" * (15 + 18 * len(configs)))
            for qtype in sorted(type_data.keys()):
                row = f"{qtype:<15}"
                for config in configs:
                    preds = all_config_preds[config]
                    type_f1s = [p["f1"] for p in preds if p.get("metadata", {}).get("type") == qtype]
                    if type_f1s:
                        row += f"{np.mean(type_f1s):<18.4f}"
                    else:
                        row += f"{'N/A':<18}"
                print(row)

        results["type_breakdown"] = type_summary

    return results


def create_difficulty_figure(rl_preds: list, all_config_preds: dict, output_dir: Path):
    """Create a figure showing F1 by difficulty level across configs."""
    has_level = any(p.get("metadata", {}).get("level") for p in rl_preds)
    if not has_level:
        return

    levels = ["easy", "medium", "hard"]
    configs = sorted(all_config_preds.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1 by difficulty level across configs
    ax1 = axes[0]
    x = np.arange(len(levels))
    width = 0.8 / len(configs)
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    for i, config in enumerate(configs):
        preds = all_config_preds[config]
        f1_by_level = []
        for level in levels:
            level_f1s = [p["f1"] for p in preds if p.get("metadata", {}).get("level") == level]
            f1_by_level.append(np.mean(level_f1s) * 100 if level_f1s else 0)
        ax1.bar(x + i * width - 0.4 + width / 2, f1_by_level, width,
                label=config, color=colors[i], edgecolor='gray', linewidth=0.5)

    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("F1 (%)")
    ax1.set_title("F1 by HotpotQA Difficulty Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.legend(fontsize=7, ncol=2)

    # Right: RL TopK distribution by difficulty
    ax2 = axes[1]
    level_topks = defaultdict(list)
    for p in rl_preds:
        level = p.get("metadata", {}).get("level", "unknown")
        if level in levels:
            level_topks[level].append(p["topk_used"])

    topk_options = sorted(set(p["topk_used"] for p in rl_preds))
    x2 = np.arange(len(topk_options))
    width2 = 0.8 / len(levels)
    level_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}

    for i, level in enumerate(levels):
        if level not in level_topks:
            continue
        from collections import Counter
        counts = Counter(level_topks[level])
        total = len(level_topks[level])
        pcts = [counts.get(k, 0) / total * 100 for k in topk_options]
        ax2.bar(x2 + i * width2 - 0.4 + width2 / 2, pcts, width2,
                label=level, color=level_colors[level], edgecolor='gray', linewidth=0.5)

    ax2.set_xlabel("Chosen TopK")
    ax2.set_ylabel("% of Queries")
    ax2.set_title("RL TopK Distribution by Difficulty")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(k) for k in topk_options])
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "hotpotqa_difficulty_analysis.pdf")
    fig.savefig(output_dir / "hotpotqa_difficulty_analysis.png")
    plt.close(fig)
    print(f"Saved: hotpotqa_difficulty_analysis.pdf")


def create_scatter_plot(rl_preds: list, output_dir: Path):
    """Analysis 4: Scatter plot of question length vs topk, colored by F1."""
    lengths = [len(p["question"]) for p in rl_preds]
    topks = [p["topk_used"] for p in rl_preds]
    f1s = [p["f1"] for p in rl_preds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Question length vs TopK, colored by F1
    ax1 = axes[0]
    scatter = ax1.scatter(lengths, topks, c=f1s, cmap='RdYlGn', s=60,
                          edgecolors='gray', linewidths=0.5, alpha=0.8)
    ax1.set_xlabel('Question Length (chars)')
    ax1.set_ylabel('Chosen TopK')
    ax1.set_title('TopK Decision vs Question Length')
    ax1.set_yticks(sorted(set(topks)))
    plt.colorbar(scatter, ax=ax1, label='F1 Score')

    # Add jitter to topk for visibility
    jittered_topks = [t + np.random.uniform(-0.2, 0.2) for t in topks]
    ax1.clear()
    scatter = ax1.scatter(lengths, jittered_topks, c=f1s, cmap='RdYlGn', s=60,
                          edgecolors='gray', linewidths=0.5, alpha=0.8)
    ax1.set_xlabel('Question Length (chars)')
    ax1.set_ylabel('Chosen TopK')
    ax1.set_title('TopK Decision vs Question Length')
    ax1.set_yticks(sorted(set(topks)))
    plt.colorbar(scatter, ax=ax1, label='F1 Score')

    # Right: TopK distribution bar chart with F1 overlay
    ax2 = axes[1]
    buckets = defaultdict(list)
    for p in rl_preds:
        buckets[p["topk_used"]].append(p["f1"])

    ks = sorted(buckets.keys())
    counts = [len(buckets[k]) for k in ks]
    mean_f1s = [np.mean(buckets[k]) for k in ks]

    bars = ax2.bar([str(k) for k in ks], counts, color='steelblue', alpha=0.7, label='Count')
    ax2.set_xlabel('Chosen TopK')
    ax2.set_ylabel('Number of Queries', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    ax2b = ax2.twinx()
    ax2b.plot([str(k) for k in ks], [f * 100 for f in mean_f1s], 'o-', color='red',
              linewidth=2, markersize=8, label='Mean F1')
    ax2b.set_ylabel('Mean F1 (%)', color='red')
    ax2b.tick_params(axis='y', labelcolor='red')
    ax2.set_title('TopK Distribution and F1 by Bucket')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    plt.tight_layout()
    fig.savefig(output_dir / 'per_question_topk_analysis.pdf')
    fig.savefig(output_dir / 'per_question_topk_analysis.png')
    plt.close(fig)
    print(f"\nSaved: per_question_topk_analysis.pdf")


def main():
    parser = argparse.ArgumentParser(description='Per-question TopK analysis')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to controlled comparison results directory')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PER-QUESTION TOPK ANALYSIS")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")

    # Load predictions — auto-detect RL prediction file
    rl_pred_path = results_dir / "predictions_RL_Dynamic_TopK.jsonl"
    if not rl_pred_path.exists():
        # Try Phase 2 filename (with learned rewrite)
        rl_pred_path = results_dir / "predictions_RL_Dynamic_TopK_+_Learned_Rewrite.jsonl"
    k3_pred_path = results_dir / "predictions_k3_fixed.jsonl"

    if not rl_pred_path.exists():
        print(f"ERROR: No RL predictions found in {results_dir}. Run controlled_comparison.py first.")
        return
    if not k3_pred_path.exists():
        print(f"ERROR: {k3_pred_path} not found. Run controlled_comparison.py first.")
        return

    rl_preds = load_predictions(str(rl_pred_path))
    k3_preds = load_predictions(str(k3_pred_path))
    print(f"Loaded {len(rl_preds)} RL predictions and {len(k3_preds)} fixed-k3 predictions")

    # Load all config predictions for cross-config analysis
    all_config_preds = {}
    for pred_file in sorted(results_dir.glob("predictions_*.jsonl")):
        config_name = pred_file.stem.replace("predictions_", "")
        all_config_preds[config_name] = load_predictions(str(pred_file))
    print(f"Loaded {len(all_config_preds)} config prediction files")

    # Run analyses
    features_analysis = analyze_topk_vs_features(rl_preds)
    adaptation_analysis = analyze_adaptation_benefit(rl_preds, k3_preds)
    bucket_analysis = analyze_f1_by_topk_bucket(rl_preds)
    create_scatter_plot(rl_preds, output_dir)

    # HotpotQA-specific analyses
    difficulty_analysis = analyze_hotpotqa_breakdowns(rl_preds, all_config_preds)
    if difficulty_analysis:
        create_difficulty_figure(rl_preds, all_config_preds, output_dir)

    # Save analysis results
    analysis_results = {
        'features': features_analysis,
        'adaptation': adaptation_analysis,
        'buckets': bucket_analysis,
        'difficulty': difficulty_analysis
    }
    with open(output_dir / 'per_question_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print(f"\nSaved analysis results to {output_dir / 'per_question_analysis.json'}")


if __name__ == '__main__':
    main()
