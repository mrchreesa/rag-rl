"""
Statistical Analysis for RAG Experiment Results

Provides rigorous statistical tests for comparing configurations:
1. Paired Bootstrap Confidence Intervals (95%)
2. Wilcoxon Signed-Rank Test (paired non-parametric)
3. Permutation Test
4. Cohen's d Effect Size

Operates on existing predictions_*.jsonl files — no API calls needed.

Usage:
    python experiments/scripts/analysis/statistical_analysis.py \
        --results-dir experiments/results/controlled_comparison_YYYYMMDD_HHMMSS

    # Compare specific configs
    python experiments/scripts/analysis/statistical_analysis.py \
        --results-dir experiments/results/controlled_comparison_YYYYMMDD_HHMMSS \
        --baseline "k10_fixed" --treatment "RL_Dynamic_TopK"

    # Multi-seed aggregation
    python experiments/scripts/analysis/statistical_analysis.py \
        --results-dirs dir1 dir2 dir3 --multi-seed
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats


def load_predictions(path: str) -> list:
    """Load per-sample predictions from JSONL."""
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Compute paired bootstrap confidence interval for mean difference.

    Args:
        scores_a: Per-sample scores for system A (e.g., baseline)
        scores_b: Per-sample scores for system B (e.g., treatment)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 0.95)
        seed: Random seed

    Returns:
        Dict with mean_diff, ci_lower, ci_upper, p_value
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = scores_b - scores_a
    observed_diff = diffs.mean()

    # Bootstrap
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_diffs[i] = diffs[indices].mean()

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Bootstrap p-value (two-sided)
    p_value = np.mean(bootstrap_diffs <= 0) * 2
    p_value = min(p_value, 2 - p_value)  # Two-sided

    return {
        "mean_diff": float(observed_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_level": confidence,
        "n_bootstrap": n_bootstrap,
        "p_value": float(p_value)
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test (paired, non-parametric).

    Tests whether the paired differences have a symmetric distribution around zero.

    Args:
        scores_a: Per-sample scores for system A
        scores_b: Per-sample scores for system B

    Returns:
        Dict with statistic, p_value
    """
    diffs = scores_b - scores_a

    # Remove zero differences (ties)
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 5:
        return {
            "statistic": None,
            "p_value": None,
            "n_non_zero": len(non_zero),
            "note": "Too few non-zero differences for Wilcoxon test"
        }

    stat, p_value = scipy_stats.wilcoxon(non_zero, alternative='two-sided')

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_non_zero": len(non_zero)
    }


def permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42
) -> dict:
    """
    Permutation test for paired differences.

    Under the null hypothesis (no difference), the sign of each paired
    difference is equally likely to be positive or negative.

    Args:
        scores_a: Per-sample scores for system A
        scores_b: Per-sample scores for system B
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dict with observed_diff, p_value
    """
    rng = np.random.RandomState(seed)
    diffs = scores_b - scores_a
    observed_diff = diffs.mean()

    # Permutation: randomly flip signs
    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = (diffs * signs).mean()
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_permutations

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        scores_a: Per-sample scores for system A
        scores_b: Per-sample scores for system B

    Returns:
        Dict with d value and interpretation
    """
    diffs = scores_b - scores_a
    d = diffs.mean() / (diffs.std(ddof=1) + 1e-10)

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "d": float(d),
        "interpretation": interpretation,
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1))
    }


def compare_configs(preds_a: list, preds_b: list, name_a: str, name_b: str) -> dict:
    """
    Run all statistical tests comparing two configurations.

    Args:
        preds_a: Predictions for config A (baseline)
        preds_b: Predictions for config B (treatment)
        name_a: Name of config A
        name_b: Name of config B

    Returns:
        Dict with all test results
    """
    # Align predictions by ID
    a_by_id = {p["id"]: p for p in preds_a}
    b_by_id = {p["id"]: p for p in preds_b}

    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    if len(common_ids) < len(preds_a):
        print(f"  Warning: {len(preds_a) - len(common_ids)} samples not matched")

    scores_a = np.array([a_by_id[i]["f1"] for i in common_ids])
    scores_b = np.array([b_by_id[i]["f1"] for i in common_ids])

    print(f"\n{'='*60}")
    print(f"COMPARISON: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"  Samples: {len(common_ids)}")
    print(f"  {name_a} mean F1: {scores_a.mean():.4f} (+/- {scores_a.std():.4f})")
    print(f"  {name_b} mean F1: {scores_b.mean():.4f} (+/- {scores_b.std():.4f})")
    print(f"  Mean difference: {(scores_b - scores_a).mean():+.4f}")

    # Run tests
    bootstrap = paired_bootstrap_ci(scores_a, scores_b)
    print(f"\n  Paired Bootstrap (10K resamples):")
    print(f"    95% CI: [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]")
    print(f"    p-value: {bootstrap['p_value']:.4f}")

    wilcoxon = wilcoxon_test(scores_a, scores_b)
    print(f"\n  Wilcoxon Signed-Rank Test:")
    if wilcoxon["p_value"] is not None:
        print(f"    Statistic: {wilcoxon['statistic']:.1f}")
        print(f"    p-value: {wilcoxon['p_value']:.4f}")
        print(f"    Non-zero pairs: {wilcoxon['n_non_zero']}")
    else:
        print(f"    {wilcoxon['note']}")

    perm = permutation_test(scores_a, scores_b)
    print(f"\n  Permutation Test (10K permutations):")
    print(f"    p-value: {perm['p_value']:.4f}")

    effect = cohens_d(scores_a, scores_b)
    print(f"\n  Cohen's d: {effect['d']:+.4f} ({effect['interpretation']})")

    # Significance summary
    sig_level = 0.05
    bootstrap_sig = bootstrap["p_value"] < sig_level
    wilcoxon_sig = wilcoxon["p_value"] is not None and wilcoxon["p_value"] < sig_level
    perm_sig = perm["p_value"] < sig_level

    print(f"\n  Significant at p<{sig_level}?")
    print(f"    Bootstrap: {'YES' if bootstrap_sig else 'NO'}")
    print(f"    Wilcoxon:  {'YES' if wilcoxon_sig else 'N/A' if wilcoxon['p_value'] is None else 'NO'}")
    print(f"    Permutation: {'YES' if perm_sig else 'NO'}")

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "n_samples": len(common_ids),
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "mean_diff": float((scores_b - scores_a).mean()),
        "bootstrap": bootstrap,
        "wilcoxon": wilcoxon,
        "permutation": perm,
        "cohens_d": effect
    }


def aggregate_multi_seed(results_dirs: list, output_dir: Path):
    """
    Aggregate results across multiple seeds.

    Args:
        results_dirs: List of result directories (one per seed)
        output_dir: Output directory
    """
    print("\n" + "=" * 60)
    print("MULTI-SEED AGGREGATION")
    print("=" * 60)

    # Collect per-config results across seeds
    config_f1s = defaultdict(list)
    config_ems = defaultdict(list)
    config_topks = defaultdict(list)

    for results_dir in results_dirs:
        summary_path = Path(results_dir) / "comparison_results.json"
        if not summary_path.exists():
            print(f"  Warning: {summary_path} not found, skipping")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        configs = summary.get("configs", summary)  # Handle both formats
        if isinstance(configs, list):
            for r in configs:
                config_f1s[r["config"]].append(r["avg_f1"])
                config_ems[r["config"]].append(r["avg_em"])
                config_topks[r["config"]].append(r["avg_topk"])

    print(f"\nSeeds: {len(results_dirs)}")
    print(f"\n{'Config':<25}{'Mean F1':<12}{'Std F1':<12}{'Mean EM':<12}{'Std EM':<12}{'Mean TopK'}")
    print("-" * 80)

    aggregated = {}
    for config in sorted(config_f1s.keys()):
        f1s = config_f1s[config]
        ems = config_ems[config]
        topks = config_topks[config]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_em = np.mean(ems)
        std_em = np.std(ems)
        mean_topk = np.mean(topks)
        print(f"{config:<25}{mean_f1:<12.4f}{std_f1:<12.4f}{mean_em:<12.4f}{std_em:<12.4f}{mean_topk:.1f}")
        aggregated[config] = {
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_em": float(mean_em),
            "std_em": float(std_em),
            "mean_topk": float(mean_topk),
            "n_seeds": len(f1s)
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of RAG experiment results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to controlled comparison results directory")
    parser.add_argument("--results-dirs", type=str, nargs="+", default=None,
                        help="Multiple results dirs for multi-seed aggregation")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline config name (default: best fixed-k)")
    parser.add_argument("--treatment", type=str, default=None,
                        help="Treatment config name (default: RL_Dynamic_TopK)")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Compare all pairs of configurations")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Aggregate across multiple seed runs")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/results/figures",
                        help="Output directory for results")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Multi-seed aggregation
    if args.multi_seed and args.results_dirs:
        dirs = [project_root / d if not Path(d).is_absolute() else Path(d) for d in args.results_dirs]
        aggregated = aggregate_multi_seed(dirs, output_dir)
        with open(output_dir / "multi_seed_aggregation.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nSaved: {output_dir / 'multi_seed_aggregation.json'}")
        return

    if args.results_dir is None:
        print("ERROR: --results-dir is required (unless using --multi-seed)")
        return

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"Results dir: {results_dir}")

    # Load all prediction files
    pred_files = sorted(results_dir.glob("predictions_*.jsonl"))
    if not pred_files:
        print(f"ERROR: No prediction files found in {results_dir}")
        return

    all_preds = {}
    for pf in pred_files:
        config_name = pf.stem.replace("predictions_", "")
        all_preds[config_name] = load_predictions(str(pf))
        print(f"  Loaded {len(all_preds[config_name])} predictions for {config_name}")

    all_results = []

    if args.all_pairs:
        # Compare all pairs
        configs = sorted(all_preds.keys())
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                result = compare_configs(
                    all_preds[configs[i]], all_preds[configs[j]],
                    configs[i], configs[j]
                )
                all_results.append(result)
    else:
        # Compare specific baseline vs treatment
        baseline_name = args.baseline
        treatment_name = args.treatment

        # Auto-detect if not specified
        if baseline_name is None:
            # Find best fixed-k config
            fixed_configs = {k: v for k, v in all_preds.items() if "fixed" in k}
            if fixed_configs:
                baseline_name = max(fixed_configs.keys(),
                                   key=lambda k: np.mean([p["f1"] for p in fixed_configs[k]]))
            else:
                baseline_name = sorted(all_preds.keys())[0]

        if treatment_name is None:
            treatment_name = "RL_Dynamic_TopK" if "RL_Dynamic_TopK" in all_preds else sorted(all_preds.keys())[-1]

        if baseline_name not in all_preds:
            print(f"ERROR: Baseline '{baseline_name}' not found. Available: {list(all_preds.keys())}")
            return
        if treatment_name not in all_preds:
            print(f"ERROR: Treatment '{treatment_name}' not found. Available: {list(all_preds.keys())}")
            return

        # Main comparison: best fixed-k vs RL
        result = compare_configs(
            all_preds[baseline_name], all_preds[treatment_name],
            baseline_name, treatment_name
        )
        all_results.append(result)

        # Also compare RL vs all fixed-k configs
        print("\n\n" + "=" * 60)
        print("SUMMARY: RL vs All Fixed-K Configurations")
        print("=" * 60)

        rl_preds = all_preds.get(treatment_name, [])
        if rl_preds:
            for config_name, preds in sorted(all_preds.items()):
                if config_name == treatment_name:
                    continue
                result = compare_configs(preds, rl_preds, config_name, treatment_name)
                all_results.append(result)

    # Save results
    results_path = output_dir / "statistical_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Print summary table
    print("\n\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Comparison':<45}{'Diff':<10}{'95% CI':<25}{'p (boot)':<10}{'Cohen d'}")
    print("-" * 100)
    for r in all_results:
        ci = r["bootstrap"]
        d = r["cohens_d"]
        ci_str = f"[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]"
        print(f"{r['comparison']:<45}{r['mean_diff']:+.4f}    {ci_str:<25}{ci['p_value']:<10.4f}{d['d']:+.4f} ({d['interpretation']})")


if __name__ == "__main__":
    main()
