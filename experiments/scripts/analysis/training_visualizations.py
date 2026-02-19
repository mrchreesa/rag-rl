"""
Training Dynamics Visualizations for RL-RAG Dissertation

Generates publication-quality figures from training_results.json:
1. TopK distribution evolution (stacked bar chart per epoch)
2. Policy loss + entropy over training steps (dual-axis)
3. Lazy agent failures per epoch (bar chart)
4. Reward component analysis (grouped bars)
5. Train vs Validation F1 learning curve

Usage:
    python experiments/scripts/analysis/training_visualizations.py \
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


# Publication-quality defaults
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
    """Load training_results.json."""
    with open(path) as f:
        return json.load(f)


def plot_topk_distribution_evolution(history: dict, output_dir: Path):
    """
    Plot 1: TopK distribution evolution across epochs.

    Stacked bar chart showing how the policy's topk preferences
    change over training. Shows the U-shaped learning curve from
    high k -> low k -> k=3 convergence.
    """
    topk_dists = history['train_topk_distribution']
    n_epochs = len(topk_dists)

    # Get all k values and sort
    k_values = sorted([int(k) for k in topk_dists[0].keys()])

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_epochs)
    width = 0.65

    # Color palette: darker = more documents
    colors = ['#e8e8e8', '#a8d8ea', '#61b3de', '#3182bd', '#1a5c8a', '#08306b']

    bottom = np.zeros(n_epochs)
    for i, k in enumerate(k_values):
        values = [topk_dists[epoch].get(str(k), 0) for epoch in range(n_epochs)]
        label = f'k={k}' if k > 0 else 'k=0 (no retrieval)'
        ax.bar(x, values, width, bottom=bottom, label=label, color=colors[i],
               edgecolor='white', linewidth=0.5)
        bottom += np.array(values)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion of Queries')
    ax.set_title('TopK Distribution Evolution During Training')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n_epochs)])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)

    # Annotate key phases
    ax.annotate('Exploration\n(uniform)', xy=(0, 1.01), fontsize=8,
                ha='center', color='gray')
    ax.annotate('Convergence\nto k=3', xy=(n_epochs-1, 1.01), fontsize=8,
                ha='center', color='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'topk_distribution_evolution.pdf')
    fig.savefig(output_dir / 'topk_distribution_evolution.png')
    plt.close(fig)
    print(f"  Saved: topk_distribution_evolution.pdf")


def plot_policy_loss_and_entropy(history: dict, output_dir: Path):
    """
    Plot 2: Policy loss and entropy over training steps.

    Dual-axis plot showing how policy loss and exploration
    (entropy) evolve during training.
    """
    policy_losses = history['policy_losses']
    train_entropy = history['train_entropy']

    # Smooth policy losses with a rolling average (lots of zeros and noise)
    window = 20
    if len(policy_losses) > window:
        smoothed_losses = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
    else:
        smoothed_losses = policy_losses

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # Policy loss (left axis)
    color1 = '#d62728'
    steps = np.arange(len(smoothed_losses))
    ax1.plot(steps, smoothed_losses, color=color1, alpha=0.8, linewidth=1.2,
             label=f'Policy Loss (smoothed, window={window})')
    ax1.set_xlabel('Training Step (batch updates)')
    ax1.set_ylabel('Policy Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color=color1, linestyle=':', alpha=0.3)

    # Entropy (right axis) - per epoch
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    epochs = np.arange(len(train_entropy))
    # Scale epoch positions to match step x-axis
    n_steps = len(policy_losses)
    n_epochs = len(train_entropy)
    epoch_positions = [int(i * n_steps / n_epochs) for i in range(n_epochs)]

    ax2.plot(epoch_positions, train_entropy, color=color2, linewidth=2.0,
             marker='o', markersize=6, label='Policy Entropy (per epoch)')
    ax2.set_ylabel('Entropy', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Max entropy reference line (ln(6) for 6 actions)
    max_entropy = np.log(6)
    ax2.axhline(y=max_entropy, color=color2, linestyle='--', alpha=0.3)
    ax2.annotate(f'Max entropy (ln 6 = {max_entropy:.2f})',
                 xy=(0, max_entropy), fontsize=8, color=color2, alpha=0.6)

    ax1.set_title('Policy Loss and Entropy During Training')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    plt.tight_layout()
    fig.savefig(output_dir / 'policy_loss_entropy.pdf')
    fig.savefig(output_dir / 'policy_loss_entropy.png')
    plt.close(fig)
    print(f"  Saved: policy_loss_entropy.pdf")


def plot_lazy_agent_failures(history: dict, output_dir: Path):
    """
    Plot 3: Lazy agent failures per epoch.

    Bar chart showing the number of times the "wrong no-retrieval
    penalty" was applied per epoch. Demonstrates the lazy agent
    problem being solved.
    """
    failures = history['lazy_agent_failures']
    n_epochs = len(failures)

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(n_epochs)
    colors = ['#d62728' if f > 0 else '#2ca02c' for f in failures]

    bars = ax.bar(x, failures, color=colors, edgecolor='white', width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, failures):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lazy Agent Penalty Count')
    ax.set_title('Lazy Agent Failures Per Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n_epochs)])
    ax.set_ylim(0, max(failures) * 1.3 if max(failures) > 0 else 1)

    # Annotate phases
    # Find where failures drop to zero
    first_zero = next((i for i, f in enumerate(failures) if f == 0), n_epochs)
    if first_zero < n_epochs:
        ax.axvline(x=first_zero - 0.5, color='green', linestyle='--', alpha=0.5)
        ax.annotate('Problem resolved',
                    xy=(first_zero, 0), xytext=(first_zero + 0.5, max(failures) * 0.6),
                    fontsize=9, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'lazy_agent_failures.pdf')
    fig.savefig(output_dir / 'lazy_agent_failures.png')
    plt.close(fig)
    print(f"  Saved: lazy_agent_failures.pdf")


def plot_reward_decomposition(history: dict, config: dict, output_dir: Path):
    """
    Plot 4: Reward component decomposition across epochs.

    Shows how quality (F1), retrieval cost, and the net reward
    relate across training. Also shows the format bonus contribution.
    """
    train_f1 = history['train_f1']
    train_rewards = history['train_rewards']
    train_avg_topk = history['train_avg_topk']
    n_epochs = len(train_f1)

    # Reconstruct approximate reward components per epoch
    base_cost = config.get('base_retrieval_cost', 0.05)
    per_doc_cost = config.get('per_doc_cost', 0.01)
    format_bonus = 0.05  # From RAGRewardCalculator default

    # Estimated retrieval cost per epoch = base_cost + per_doc_cost * avg_topk
    est_retrieval_costs = [base_cost + per_doc_cost * k for k in train_avg_topk]

    # Estimated quality reward = F1 (quality_score component)
    quality_rewards = train_f1

    # Estimated net = quality - cost + format_bonus (~0.05 always collected)
    # The actual reward also includes lazy agent penalties, etc.

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_epochs)
    width = 0.25

    ax.bar(x - width, quality_rewards, width, label='F1 (Quality)',
           color='#2ca02c', alpha=0.85, edgecolor='white')
    ax.bar(x, est_retrieval_costs, width, label='Retrieval Cost (est.)',
           color='#d62728', alpha=0.85, edgecolor='white')
    ax.bar(x + width, train_rewards, width, label='Net Reward',
           color='#1f77b4', alpha=0.85, edgecolor='white')

    # Add format bonus reference line
    ax.axhline(y=format_bonus, color='orange', linestyle=':', alpha=0.5,
               label=f'Format Bonus ({format_bonus})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Reward Decomposition Across Training')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n_epochs)])
    ax.legend(frameon=True)

    # Add avg topk annotation on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x, train_avg_topk, 'k--', marker='s', markersize=5,
             alpha=0.5, label='Avg TopK')
    ax2.set_ylabel('Average TopK', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    fig.savefig(output_dir / 'reward_decomposition.pdf')
    fig.savefig(output_dir / 'reward_decomposition.png')
    plt.close(fig)
    print(f"  Saved: reward_decomposition.pdf")


def plot_f1_learning_curve(history: dict, output_dir: Path):
    """
    Plot 5: Train vs Validation F1 learning curve.

    Shows model performance over training epochs with
    the baseline reference line.
    """
    train_f1 = history['train_f1']
    val_f1 = history['val_f1']
    n_epochs = len(train_f1)

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, n_epochs + 1)

    ax.plot(epochs, [f * 100 for f in train_f1], 'o-', color='#1f77b4',
            linewidth=2, markersize=7, label='Train F1')
    ax.plot(epochs, [f * 100 for f in val_f1], 's-', color='#ff7f0e',
            linewidth=2, markersize=7, label='Validation F1')

    # Baseline reference (Dense E5 + GPT-4o-mini topk=10)
    baseline_f1 = 31.10
    ax.axhline(y=baseline_f1, color='red', linestyle='--', alpha=0.6,
               label=f'Baseline (always k=10): {baseline_f1}%')

    # Mark best validation epoch
    best_val_idx = np.argmax(val_f1)
    best_val = val_f1[best_val_idx] * 100
    ax.annotate(f'Best: {best_val:.1f}%',
                xy=(best_val_idx + 1, best_val),
                xytext=(best_val_idx + 1.5, best_val + 2),
                fontsize=10, fontweight='bold', color='#ff7f0e',
                arrowprops=dict(arrowstyle='->', color='#ff7f0e'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('F1 Learning Curve: RL Agent vs Baseline')
    ax.set_xticks(epochs)
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(train_f1), max(val_f1)) * 100 + 10)

    # Add validation reward on secondary axis for context
    if 'val_rewards' in history:
        ax2 = ax.twinx()
        val_rewards = history['val_rewards']
        ax2.plot(epochs, val_rewards, 'x--', color='gray', alpha=0.4,
                 linewidth=1, label='Val Reward')
        ax2.set_ylabel('Reward', color='gray', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    fig.savefig(output_dir / 'f1_learning_curve.pdf')
    fig.savefig(output_dir / 'f1_learning_curve.png')
    plt.close(fig)
    print(f"  Saved: f1_learning_curve.pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate training dynamics visualizations')
    parser.add_argument('--results', type=str,
                        default='experiments/results/rl_enhanced_20260208_223508/training_results.json',
                        help='Path to training_results.json')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[3]
    results_path = project_root / args.results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    print(f"Saving figures to: {output_dir}")
    print()

    data = load_results(results_path)
    history = data['history']
    config = data.get('config', {})

    print("Generating figures:")
    plot_topk_distribution_evolution(history, output_dir)
    plot_policy_loss_and_entropy(history, output_dir)
    plot_lazy_agent_failures(history, output_dir)
    plot_reward_decomposition(history, config, output_dir)
    plot_f1_learning_curve(history, output_dir)

    print(f"\nAll 5 figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
