"""
K-Fold Cross-Validation for RAG Configurations

Runs 5-fold CV on all custom dataset samples (combined train+test = 579)
to produce more robust estimates with standard deviations.

Each fold:
1. Trains RL policy on 80% of data
2. Evaluates all configs on remaining 20%

Usage:
    python experiments/scripts/eval/kfold_evaluation.py --n-folds 5 --wandb
    python experiments/scripts/eval/kfold_evaluation.py --n-folds 5 --configs-only  # Skip RL training
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
import random
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.env_loader import load_env_file
load_env_file(PROJECT_ROOT)

import torch
from agents.flashrag_components import DenseRetrieverWrapper, GeneratorWrapper
from agents.enhanced_pipeline import EnhancedRAGPipeline, RLTrainer, DynamicTopKPolicyNetwork
from agents.reward import compute_f1, compute_exact_match


def load_all_custom_data():
    """Load all custom dataset samples (train + test)."""
    data = []
    for split in ["train", "test"]:
        path = PROJECT_ROOT / f"data/datasets/custom_dataset/{split}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
    return data


def create_folds(data: list, n_folds: int, seed: int = 42):
    """Create stratified folds from data."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    folds = []
    fold_size = len(indices) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            end = len(indices)  # Last fold gets remainder
        else:
            end = start + fold_size
        test_indices = set(indices[start:end])
        train_indices = [j for j in indices if j not in test_indices]
        folds.append({
            "train": [data[j] for j in train_indices],
            "test": [data[j] for j in sorted(test_indices)]
        })

    return folds


def evaluate_fixed_k(retriever, generator, test_data, topk):
    """Evaluate a fixed-k configuration."""
    f1_scores = []
    em_scores = []

    for sample in test_data:
        question = sample["question"]
        golden_answers = sample["golden_answers"]

        if topk == 0:
            answer = generator.generate_direct(question)
        else:
            docs = retriever.retrieve([question], topk=topk)[0]
            answer = generator.generate_with_retrieval(question, docs)

        f1_scores.append(compute_f1(answer, golden_answers))
        em_scores.append(compute_exact_match(answer, golden_answers))

    return {
        "avg_f1": float(np.mean(f1_scores)),
        "avg_em": float(np.mean(em_scores)),
        "per_sample_f1": f1_scores
    }


def train_and_evaluate_rl(
    retriever, generator, train_data, test_data,
    epochs=10, topk_options=None, use_ollama=False,
    algorithm="reinforce", group_size=8
):
    """Train RL policy on train_data, evaluate on test_data."""
    topk_options = topk_options or [0, 1, 3, 5, 7, 10]

    pipeline = EnhancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        use_query_rewriter=False,
        use_learned_retrieval=True,
        use_dynamic_topk=True,
        topk_options=topk_options
    )

    trainer = RLTrainer(
        pipeline=pipeline,
        use_dynamic_cost=True,
        base_retrieval_cost=0.05,
        per_doc_cost=0.01,
        wrong_no_retrieval_penalty=0.3,
        entropy_coef=0.01,
        eval_temperature=0.7,
        use_wandb=False
    )

    # Train
    results = trainer.train(
        train_data=train_data,
        val_data=test_data[:20],  # Small val set
        epochs=epochs,
        use_curriculum=True,
        algorithm=algorithm,
        group_size=group_size
    )

    # Evaluate on full test set
    f1_scores = []
    em_scores = []
    topk_values = []

    for sample in test_data:
        question = sample["question"]
        golden_answers = sample["golden_answers"]

        answer, docs, metadata = pipeline.answer(
            question, deterministic=True, temperature=0.7
        )

        f1_scores.append(compute_f1(answer, golden_answers))
        em_scores.append(compute_exact_match(answer, golden_answers))
        topk_values.append(metadata.get("topk_used", 0))

    return {
        "avg_f1": float(np.mean(f1_scores)),
        "avg_em": float(np.mean(em_scores)),
        "avg_topk": float(np.mean(topk_values)),
        "per_sample_f1": f1_scores,
        "best_train_f1": results["best_f1"]
    }


def main():
    parser = argparse.ArgumentParser(description="K-Fold CV for RAG configurations")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="RL training epochs per fold")
    parser.add_argument("--configs-only", action="store_true",
                        help="Only evaluate fixed-k configs (skip RL training)")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama instead of OpenAI")
    parser.add_argument("--algorithm", type=str, default="reinforce",
                        choices=["reinforce", "grpo"],
                        help="RL algorithm")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"experiments/results/kfold_{args.n_folds}fold_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"{args.n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    # Load data
    all_data = load_all_custom_data()
    print(f"Total samples: {len(all_data)}")

    folds = create_folds(all_data, args.n_folds, seed=args.seed)
    print(f"Created {len(folds)} folds")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: train={len(fold['train'])}, test={len(fold['test'])}")

    # Initialize shared retriever (reused across folds)
    print("\nInitializing retriever...")
    retriever = DenseRetrieverWrapper()
    generator = GeneratorWrapper(use_ollama=args.use_ollama)

    # W&B
    if args.wandb:
        import wandb
        gen_tag = "ollama" if args.use_ollama else "gpt4omini"
        wandb.init(
            project="rl-rag-enhanced",
            name=f"kfold_{args.n_folds}fold_{gen_tag}_{datetime.now().strftime('%m%d_%H%M')}",
            tags=["kfold", f"{args.n_folds}-fold", gen_tag],
            config=vars(args)
        )

    # Configs to evaluate
    fixed_configs = [("k=0", 0), ("k=1", 1), ("k=3", 3), ("k=5", 5), ("k=10", 10)]

    # Results accumulator
    all_fold_results = []

    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        fold_results = {"fold": fold_idx + 1}

        # Evaluate fixed-k configs
        for name, k in fixed_configs:
            print(f"\n  Evaluating {name}...")
            result = evaluate_fixed_k(retriever, generator, fold["test"], k)
            fold_results[name] = result
            print(f"    F1: {result['avg_f1']:.4f}, EM: {result['avg_em']:.4f}")

        # Train and evaluate RL (unless --configs-only)
        if not args.configs_only:
            print(f"\n  Training RL ({args.algorithm})...")
            rl_result = train_and_evaluate_rl(
                retriever, generator,
                fold["train"], fold["test"],
                epochs=args.epochs,
                use_ollama=args.use_ollama,
                algorithm=args.algorithm,
                group_size=args.group_size
            )
            fold_results["RL Dynamic TopK"] = rl_result
            print(f"    RL F1: {rl_result['avg_f1']:.4f}, Avg K: {rl_result['avg_topk']:.1f}")

        all_fold_results.append(fold_results)

        # Log to W&B
        if args.wandb:
            import wandb
            log_data = {f"fold_{fold_idx+1}/{name}": fold_results[name]["avg_f1"]
                       for name in fold_results if name != "fold" and isinstance(fold_results[name], dict)}
            wandb.log(log_data)

    # Aggregate results
    print("\n\n" + "=" * 60)
    print(f"{args.n_folds}-FOLD CV SUMMARY")
    print("=" * 60)

    config_names = [name for name, _ in fixed_configs]
    if not args.configs_only:
        config_names.append("RL Dynamic TopK")

    print(f"\n{'Config':<25}{'Mean F1':<12}{'Std F1':<12}{'Mean EM':<12}{'Std EM'}")
    print("-" * 65)

    summary = {}
    for config in config_names:
        f1s = [fr[config]["avg_f1"] for fr in all_fold_results if config in fr]
        ems = [fr[config]["avg_em"] for fr in all_fold_results if config in fr]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_em = np.mean(ems)
        std_em = np.std(ems)
        print(f"{config:<25}{mean_f1:<12.4f}{std_f1:<12.4f}{mean_em:<12.4f}{std_em:.4f}")
        summary[config] = {
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_em": float(mean_em),
            "std_em": float(std_em),
            "fold_f1s": [float(f) for f in f1s],
            "fold_ems": [float(e) for e in ems]
        }

    # Save results
    results = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "total_samples": len(all_data),
        "generator": "ollama" if args.use_ollama else "gpt-4o-mini",
        "algorithm": args.algorithm,
        "summary": summary,
        "fold_details": all_fold_results
    }

    results_path = output_dir / "kfold_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    if args.wandb:
        import wandb
        table_data = [[config, s["mean_f1"], s["std_f1"], s["mean_em"], s["std_em"]]
                      for config, s in summary.items()]
        table = wandb.Table(
            columns=["Config", "Mean F1", "Std F1", "Mean EM", "Std EM"],
            data=table_data
        )
        wandb.log({"kfold_summary": table})
        wandb.finish()

    # Print cost
    generator.print_usage_summary()


if __name__ == "__main__":
    main()
