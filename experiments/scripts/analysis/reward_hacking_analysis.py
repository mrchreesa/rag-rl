"""
Reward Hacking & Confidence Calibration Analysis for RL-RAG Dissertation

Analyzes the trained policy for potential reward exploitation:
1. TopK-cost gaming: Does the policy prefer low k where F1 is worse?
2. Lazy agent residual: Verification that the fix works
3. Format bonus exploitation: Quantify the always-collected +0.05
4. Confidence calibration: Entropy vs F1 correlation

Usage:
    python experiments/scripts/analysis/reward_hacking_analysis.py \
        [--results PATH] [--output-dir PATH]
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


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_topk_cost_gaming(history: dict, config: dict):
    """
    Analysis 1: TopK-Cost Gaming Detection

    If the policy is gaming costs, it would prefer lower k values
    even when F1 is worse. We check by looking at the relationship
    between the policy's topk preferences and F1 outcomes.
    """
    print("=" * 60)
    print("ANALYSIS 1: TopK-Cost Gaming Detection")
    print("=" * 60)

    topk_dists = history['train_topk_distribution']
    train_f1 = history['train_f1']
    train_avg_topk = history['train_avg_topk']
    val_f1 = history['val_f1']
    val_avg_topk = history['val_avg_topk']

    base_cost = config.get('base_retrieval_cost', 0.05)
    per_doc_cost = config.get('per_doc_cost', 0.01)

    print(f"\nCost structure: base={base_cost}, per_doc={per_doc_cost}")
    print(f"Cost by k: k=1: {base_cost + per_doc_cost:.3f}, "
          f"k=3: {base_cost + 3*per_doc_cost:.3f}, "
          f"k=5: {base_cost + 5*per_doc_cost:.3f}, "
          f"k=10: {base_cost + 10*per_doc_cost:.3f}")

    print(f"\n{'Epoch':<8}{'Avg TopK':<10}{'Train F1':<12}{'Val F1':<10}{'Est. Cost':<12}{'Net Reward Est.'}")
    print("-" * 62)

    for i in range(len(train_f1)):
        avg_k = train_avg_topk[i]
        est_cost = base_cost + per_doc_cost * avg_k if avg_k > 0 else 0
        net_est = train_f1[i] - est_cost + 0.05  # +0.05 format bonus
        print(f"{i+1:<8}{avg_k:<10.2f}{train_f1[i]:<12.4f}{val_f1[i]:<10.4f}"
              f"{est_cost:<12.4f}{net_est:.4f}")

    # Check for gaming: Does lowering k hurt F1 disproportionately?
    # Compare val F1 when avg_topk is low vs high
    sorted_by_topk = sorted(zip(val_avg_topk, val_f1))
    low_k_f1 = [f for k, f in sorted_by_topk if k <= 3]
    high_k_f1 = [f for k, f in sorted_by_topk if k > 3]

    print(f"\n--- Gaming Assessment ---")
    if low_k_f1 and high_k_f1:
        avg_low = np.mean(low_k_f1)
        avg_high = np.mean(high_k_f1)
        print(f"Val F1 when avg_topk <= 3: {avg_low:.4f} (n={len(low_k_f1)} epochs)")
        print(f"Val F1 when avg_topk >  3: {avg_high:.4f} (n={len(high_k_f1)} epochs)")
        diff = avg_high - avg_low
        print(f"F1 difference: {diff:+.4f}")

        if avg_low >= avg_high * 0.95:
            print("VERDICT: NO gaming detected. Lower k achieves comparable F1.")
            print("  The policy learned genuine efficiency - fewer docs with similar quality.")
        else:
            pct_loss = (1 - avg_low / avg_high) * 100
            print(f"VERDICT: Potential mild gaming. {pct_loss:.1f}% F1 loss at lower k.")
            print("  But this may be acceptable if cost savings outweigh the F1 drop.")
    else:
        print("Insufficient data to compare low vs high k epochs.")

    # Final topk distribution analysis
    final_dist = topk_dists[-1]
    print(f"\n--- Final TopK Distribution (Epoch {len(topk_dists)}) ---")
    for k in sorted(final_dist.keys(), key=lambda x: int(x)):
        pct = final_dist[k] * 100
        cost = base_cost + per_doc_cost * int(k) if int(k) > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  k={k:>2}: {pct:5.1f}% {bar:<45} (cost: {cost:.3f})")

    return {
        'low_k_f1': np.mean(low_k_f1) if low_k_f1 else None,
        'high_k_f1': np.mean(high_k_f1) if high_k_f1 else None,
        'final_distribution': final_dist
    }


def analyze_lazy_agent_residual(history: dict):
    """
    Analysis 2: Lazy Agent Residual Verification

    Verify that the wrong_no_retrieval_penalty fix successfully
    eliminated the lazy agent problem.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Lazy Agent Residual Check")
    print("=" * 60)

    failures = history['lazy_agent_failures']
    retrieval_rates = history['train_retrieval_rate']
    val_retrieval = history['val_retrieval_rate']

    print(f"\n{'Epoch':<8}{'Failures':<12}{'Train Retr Rate':<18}{'Val Retr Rate'}")
    print("-" * 50)
    for i in range(len(failures)):
        status = " [ISSUE]" if failures[i] > 5 else " [OK]" if failures[i] == 0 else ""
        print(f"{i+1:<8}{failures[i]:<12}{retrieval_rates[i]:<18.1%}"
              f"{val_retrieval[i]:.1%}{status}")

    total_failures = sum(failures)
    peak_failures = max(failures)
    peak_epoch = failures.index(peak_failures) + 1
    resolved_epoch = next((i + 1 for i, f in enumerate(failures) if f == 0), None)

    print(f"\n--- Summary ---")
    print(f"Total failures across training: {total_failures}")
    print(f"Peak failures: {peak_failures} (epoch {peak_epoch})")
    if resolved_epoch:
        print(f"First zero-failure epoch: {resolved_epoch}")
        remaining_failures = sum(failures[resolved_epoch-1:])
        print(f"Failures after resolution: {remaining_failures}")
    print(f"Final retrieval rate (train): {retrieval_rates[-1]:.1%}")
    print(f"Final retrieval rate (val):   {val_retrieval[-1]:.1%}")

    # Check for complete resolution
    late_failures = sum(failures[len(failures)//2:])
    if late_failures == 0:
        print("\nVERDICT: Lazy agent problem FULLY RESOLVED in second half of training.")
    elif late_failures <= 2:
        print(f"\nVERDICT: Lazy agent problem MOSTLY resolved ({late_failures} minor occurrences).")
    else:
        print(f"\nVERDICT: Lazy agent problem PERSISTS ({late_failures} failures in late training).")

    return {
        'total_failures': total_failures,
        'peak_failures': peak_failures,
        'peak_epoch': peak_epoch,
        'resolved_epoch': resolved_epoch,
    }


def analyze_format_bonus_exploitation(history: dict, config: dict):
    """
    Analysis 3: Format Bonus Exploitation

    The +0.05 format bonus is always collected (answer is always
    non-empty and < 500 chars). Quantify its impact on rewards.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Format Bonus Impact")
    print("=" * 60)

    format_bonus = 0.05  # From RAGRewardCalculator default
    train_rewards = history['train_rewards']
    train_f1 = history['train_f1']

    print(f"\nFormat bonus: +{format_bonus} per answer (always collected)")
    print(f"Condition: non-empty answer < 500 chars (always met by GPT-4o-mini)")

    print(f"\n{'Epoch':<8}{'Reward':<12}{'F1':<10}{'Format Bonus':<15}{'% of Reward'}")
    print("-" * 55)
    for i in range(len(train_rewards)):
        reward = train_rewards[i]
        pct = (format_bonus / reward * 100) if reward > 0 else 0
        print(f"{i+1:<8}{reward:<12.4f}{train_f1[i]:<10.4f}{format_bonus:<15.2f}{pct:.1f}%")

    avg_reward = np.mean(train_rewards)
    avg_contribution = format_bonus / avg_reward * 100 if avg_reward > 0 else 0

    print(f"\n--- Summary ---")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Format bonus contribution: {avg_contribution:.1f}% of average reward")
    print(f"Format bonus as fraction of F1: {format_bonus / np.mean(train_f1) * 100:.1f}%")

    if avg_contribution > 30:
        print("\nVERDICT: Format bonus is a SIGNIFICANT fraction of total reward.")
        print("  This is a form of reward exploitation, but benign - it doesn't")
        print("  change the relative ordering of actions since all actions receive it.")
    else:
        print(f"\nVERDICT: Format bonus is a minor component ({avg_contribution:.1f}%).")
        print("  It provides a small baseline reward but doesn't distort policy learning.")

    print("\nNOTE: The format bonus is a constant additive term across all actions.")
    print("  It does not create a gradient difference between retrieve/no-retrieve,")
    print("  so it cannot cause reward hacking. It serves as a baseline reward floor.")

    return {
        'avg_contribution_pct': avg_contribution,
        'format_bonus': format_bonus
    }


def analyze_confidence_calibration(history: dict, output_dir: Path):
    """
    Analysis 4: Confidence Calibration (Entropy vs F1)

    Correlate policy entropy with answer quality.
    High entropy = uncertain policy -> should correlate with lower F1.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Confidence Calibration")
    print("=" * 60)

    train_entropy = history['train_entropy']
    train_f1 = history['train_f1']
    val_f1 = history['val_f1']
    val_avg_topk = history['val_avg_topk']

    # Max entropy for 6-action space
    max_entropy = np.log(6)

    print(f"\nMax possible entropy: {max_entropy:.4f} (uniform over 6 actions)")

    print(f"\n{'Epoch':<8}{'Entropy':<12}{'Norm. Entropy':<16}{'Train F1':<12}{'Val F1'}")
    print("-" * 56)
    for i in range(len(train_entropy)):
        norm_ent = train_entropy[i] / max_entropy
        confidence = 1 - norm_ent
        print(f"{i+1:<8}{train_entropy[i]:<12.4f}{norm_ent:<16.4f}"
              f"{train_f1[i]:<12.4f}{val_f1[i]:.4f}")

    # Correlation analysis
    correlation = np.corrcoef(train_entropy, val_f1)[0, 1]
    print(f"\nCorrelation (entropy vs val F1): {correlation:+.4f}")

    if correlation < -0.3:
        print("  Negative correlation: Higher confidence -> better F1 (well calibrated)")
    elif correlation > 0.3:
        print("  Positive correlation: Higher uncertainty -> better F1 (exploration helps)")
    else:
        print("  Weak correlation: Confidence is not strongly predictive of F1")

    # Check if k=0 (no retrieval) epochs have lower F1
    no_retr_epochs = [i for i, k in enumerate(val_avg_topk) if k <= 1.0]
    retr_epochs = [i for i, k in enumerate(val_avg_topk) if k > 1.0]

    if no_retr_epochs and retr_epochs:
        no_retr_f1 = np.mean([val_f1[i] for i in no_retr_epochs])
        retr_f1 = np.mean([val_f1[i] for i in retr_epochs])
        print(f"\nVal F1 when avg_topk <= 1 (low/no retrieval): {no_retr_f1:.4f}")
        print(f"Val F1 when avg_topk >  1 (substantial retrieval): {retr_f1:.4f}")

    # --- Generate calibration plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Entropy vs F1 scatter + trajectory
    epochs = np.arange(1, len(train_entropy) + 1)
    scatter = ax1.scatter(train_entropy, [f * 100 for f in val_f1],
                          c=epochs, cmap='viridis', s=100, zorder=3, edgecolors='white')
    # Connect points chronologically
    ax1.plot(train_entropy, [f * 100 for f in val_f1], '--', alpha=0.3, color='gray')

    # Add epoch labels
    for i, (e, f) in enumerate(zip(train_entropy, val_f1)):
        ax1.annotate(f'{i+1}', (e, f*100), fontsize=8, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

    ax1.set_xlabel('Policy Entropy')
    ax1.set_ylabel('Validation F1 (%)')
    ax1.set_title(f'Confidence Calibration (r={correlation:+.3f})')
    ax1.axvline(x=max_entropy, color='red', linestyle=':', alpha=0.3,
                label=f'Max entropy ({max_entropy:.2f})')
    plt.colorbar(scatter, ax=ax1, label='Epoch')
    ax1.legend(frameon=True)

    # Right: Entropy + F1 over epochs (dual axis)
    color1 = '#1f77b4'
    color2 = '#ff7f0e'
    ax2.plot(epochs, train_entropy, 'o-', color=color1, linewidth=2,
             markersize=7, label='Entropy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Policy Entropy', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.axhline(y=max_entropy, color=color1, linestyle=':', alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(epochs, [f * 100 for f in val_f1], 's-', color=color2,
             linewidth=2, markersize=7, label='Val F1')
    ax3.set_ylabel('Validation F1 (%)', color=color2)
    ax3.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
    ax2.set_title('Entropy and F1 Over Training')

    plt.tight_layout()
    fig.savefig(output_dir / 'confidence_calibration.pdf')
    fig.savefig(output_dir / 'confidence_calibration.png')
    plt.close(fig)
    print(f"\nSaved: confidence_calibration.pdf")

    return {
        'entropy_f1_correlation': correlation,
    }


def print_summary(gaming, lazy, format_bonus, calibration):
    """Print final summary table for dissertation inclusion."""
    print("\n" + "=" * 60)
    print("SUMMARY: Reward Hacking Analysis")
    print("=" * 60)

    print("""
+-------------------------------+------------+------------------+
| Check                         | Status     | Evidence         |
+-------------------------------+------------+------------------+""")

    # TopK gaming
    if gaming['low_k_f1'] is not None and gaming['high_k_f1'] is not None:
        ratio = gaming['low_k_f1'] / gaming['high_k_f1'] if gaming['high_k_f1'] > 0 else 0
        if ratio >= 0.95:
            gaming_status = "No gaming"
            gaming_evidence = f"{ratio:.0%} F1 retention"
        else:
            gaming_status = "Mild"
            gaming_evidence = f"{ratio:.0%} F1 retention"
    else:
        gaming_status = "N/A"
        gaming_evidence = "Insufficient data"

    print(f"| TopK-Cost Gaming              | {gaming_status:<10} | {gaming_evidence:<16} |")

    # Lazy agent
    if lazy['total_failures'] == 0:
        lazy_status = "Never occurred"
    elif lazy.get('resolved_epoch'):
        lazy_status = f"Fixed ep {lazy['resolved_epoch']}"
    else:
        lazy_status = "Persists"
    lazy_evidence = f"{lazy['total_failures']} total failures"
    print(f"| Lazy Agent Problem            | {lazy_status:<10} | {lazy_evidence:<16} |")

    # Format bonus
    fb_pct = format_bonus['avg_contribution_pct']
    fb_status = "Benign" if fb_pct < 30 else "Notable"
    fb_evidence = f"{fb_pct:.0f}% of reward"
    print(f"| Format Bonus Exploitation     | {fb_status:<10} | {fb_evidence:<16} |")

    # Calibration
    corr = calibration['entropy_f1_correlation']
    if abs(corr) < 0.3:
        cal_status = "Weak"
    elif corr < 0:
        cal_status = "Good"
    else:
        cal_status = "Inverted"
    cal_evidence = f"r = {corr:+.3f}"
    print(f"| Confidence Calibration        | {cal_status:<10} | {cal_evidence:<16} |")

    print("+-------------------------------+------------+------------------+")


def main():
    parser = argparse.ArgumentParser(description='Reward hacking and confidence calibration analysis')
    parser.add_argument('--results', type=str,
                        default='experiments/results/rl_enhanced_20260208_223508/training_results.json',
                        help='Path to training_results.json')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    results_path = project_root / args.results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    print(f"Saving figures to: {output_dir}")

    data = load_results(results_path)
    history = data['history']
    config = data.get('config', {})

    # Run all analyses
    gaming = analyze_topk_cost_gaming(history, config)
    lazy = analyze_lazy_agent_residual(history)
    format_bonus = analyze_format_bonus_exploitation(history, config)
    calibration = analyze_confidence_calibration(history, output_dir)

    # Print summary table
    print_summary(gaming, lazy, format_bonus, calibration)


if __name__ == '__main__':
    main()
