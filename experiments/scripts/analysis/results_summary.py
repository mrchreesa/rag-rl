"""
Results Summary Figure: Bar chart of F1 for all configurations.

Reads comparison_results.json from the controlled comparison and produces
a publication-quality bar chart annotated with avg-k and efficiency callout.

Usage:
    python experiments/scripts/analysis/results_summary.py \
        --results experiments/results/controlled_comparison_YYYYMMDD_HHMMSS/comparison_results.json
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
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def main():
    parser = argparse.ArgumentParser(description='Generate comparison bar chart')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to comparison_results.json')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = project_root / results_path

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Extract data
    configs = [r["config"] for r in results]
    f1_scores = [r["avg_f1"] * 100 for r in results]
    avg_topks = [r["avg_topk"] for r in results]

    # Short labels for x-axis
    short_labels = []
    for c in configs:
        if "parametric" in c:
            short_labels.append("k=0\n(no retrieval)")
        elif "k=1" in c:
            short_labels.append("k=1")
        elif "k=3" in c:
            short_labels.append("k=3")
        elif "k=5" in c:
            short_labels.append("k=5")
        elif "k=10" in c:
            short_labels.append("k=10")
        elif "RL" in c:
            short_labels.append("RL\nDynamic TopK")
        else:
            short_labels.append(c)

    # Colors
    colors = ['#bdc3c7'] * len(configs)  # Gray for fixed-k
    colors[0] = '#95a5a6'  # Darker gray for parametric
    colors[-1] = '#2ecc71'  # Green for RL

    # Find best fixed-k and RL indices
    best_fixed_idx = max(range(len(results) - 1), key=lambda i: f1_scores[i])
    rl_idx = len(results) - 1

    # Highlight best fixed-k
    colors[best_fixed_idx] = '#3498db'  # Blue for best fixed-k

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(configs)), f1_scores, color=colors,
                  edgecolor='white', linewidth=1.5, width=0.7)

    # Add F1 value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                f'{f1_scores[i]:.1f}%', ha='center', va='bottom',
                fontweight='bold' if i in (best_fixed_idx, rl_idx) else 'normal',
                fontsize=11)

    # Annotate RL bar with avg k
    rl_bar = bars[rl_idx]
    rl_avg_k = avg_topks[rl_idx]
    ax.annotate(f'avg k={rl_avg_k:.1f}',
                xy=(rl_bar.get_x() + rl_bar.get_width() / 2, f1_scores[rl_idx] / 2),
                ha='center', va='center', fontsize=10, color='white',
                fontweight='bold')

    # Add efficiency callout comparing RL to k=10
    k10_idx = next(i for i, c in enumerate(configs) if "k=10" in c)
    k10_topk = avg_topks[k10_idx]
    docs_saved = (1 - rl_avg_k / k10_topk) * 100

    if docs_saved > 0:
        ax.annotate(
            f'{docs_saved:.0f}% fewer\nretrievals\nvs k=10',
            xy=(rl_bar.get_x() + rl_bar.get_width() + 0.1, f1_scores[rl_idx] * 0.7),
            fontsize=9, color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60', alpha=0.8),
        )

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(short_labels)
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Controlled Comparison: Fixed-k Baselines vs RL Dynamic TopK')
    ax.set_ylim(0, max(f1_scores) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', label='No retrieval'),
        Patch(facecolor='#bdc3c7', label='Fixed-k baselines'),
        Patch(facecolor='#3498db', label='Best fixed-k'),
        Patch(facecolor='#2ecc71', label='RL Dynamic TopK'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)

    plt.tight_layout()
    fig.savefig(output_dir / 'controlled_comparison.pdf')
    fig.savefig(output_dir / 'controlled_comparison.png')
    plt.close(fig)
    print(f"Saved: {output_dir / 'controlled_comparison.pdf'}")
    print(f"Saved: {output_dir / 'controlled_comparison.png'}")


if __name__ == '__main__':
    main()
