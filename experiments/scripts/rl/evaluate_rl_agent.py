#!/usr/bin/env python3
"""
Evaluation Script for RL-RAG Agent vs Baselines

This script evaluates the trained RL-RAG agent against the established
baselines on the custom academic dataset (87 test samples).

Usage:
    python evaluate_rl_agent.py --mode baseline  # Run baseline comparison
    python evaluate_rl_agent.py --mode rl        # Run RL agent evaluation
    python evaluate_rl_agent.py --mode compare   # Compare RL vs baseline

To enable wandb logging:
    python evaluate_rl_agent.py --wandb --mode compare
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Global wandb run reference
_wandb_run = None

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/rl/agent-lightning'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Load .env file if it exists
from utils.env_loader import load_env_file
load_env_file(PROJECT_ROOT)

# Suppress verbose warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL-RAG Agent (with trained policy)")

    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "rl", "compare", "efficiency", "trained"],
                        help="Evaluation mode ('trained' uses actual trained policy)")
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["custom", "hotpotqa"],
                        help="Dataset to evaluate on (default: custom for dissertation)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples (None = all)")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of documents to retrieve")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use local Ollama instead of OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Generator model (e.g. gpt-4o-mini, claude-3-5-haiku-20241022)")

    # Trained policy arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained policy checkpoint (best_model.pt)")
    parser.add_argument("--eval-temperature", type=float, default=0.7,
                        help="Temperature for soft sampling during evaluation")

    # Dynamic TopK arguments
    parser.add_argument("--analyze-topk", action="store_true",
                        help="Analyze and log topk distribution for dynamic topk policies")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="rl-rag-agent",
                        help="Wandb project name (default: rl-rag-agent)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[],
                        help="Tags for the wandb run")

    return parser.parse_args()


def init_wandb(args, dataset_name: str):
    """Initialize Weights & Biases logging."""
    global _wandb_run
    
    try:
        import wandb
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        print("   Continuing without wandb logging...")
        args.wandb = False
        return
    
    # Generate run name if not provided
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_name = f"eval_{args.mode}_{dataset_name}_k{args.topk}_{timestamp}"
    
    # Build config dict
    config = {
        "mode": args.mode,
        "dataset": dataset_name,
        "split": args.split,
        "samples": args.samples,
        "topk": args.topk,
        "use_ollama": args.use_ollama,
    }
    
    # Build tags
    tags = list(args.wandb_tags) if args.wandb_tags else []
    tags.append("evaluation")
    tags.append(dataset_name)
    tags.append(f"mode={args.mode}")
    
    print(f"\n📊 Initializing Wandb...")
    print(f"   Project: {args.wandb_project}")
    print(f"   Run name: {args.wandb_name}")
    
    _wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        tags=tags,
        reinit=True,
    )
    
    print(f"   URL: {_wandb_run.get_url()}")


def log_results_to_wandb(results: List[Dict[str, Any]]):
    """Log evaluation results to wandb."""
    global _wandb_run
    if _wandb_run is None:
        return
    
    import wandb
    
    # Log each method's results
    for r in results:
        method = r["method"].replace(" ", "_")
        _wandb_run.log({
            f"{method}/avg_f1": r["avg_f1"],
            f"{method}/avg_em": r["avg_em"],
            f"{method}/avg_reward": r["avg_reward"],
            f"{method}/retrieval_rate": r["retrieval_rate"],
            f"{method}/num_samples": r["num_samples"],
        })
    
    # Create comparison table
    if len(results) > 0:
        table = wandb.Table(columns=["Method", "F1", "EM", "Reward", "Retrieval Rate"])
        for r in results:
            table.add_data(r["method"], r["avg_f1"], r["avg_em"], r["avg_reward"], r["retrieval_rate"])
        _wandb_run.log({"comparison_table": table})
    
    # Update summary with best results
    best_by_f1 = max(results, key=lambda x: x["avg_f1"])
    best_by_reward = max(results, key=lambda x: x["avg_reward"])
    
    _wandb_run.summary.update({
        "best_f1_method": best_by_f1["method"],
        "best_f1": best_by_f1["avg_f1"],
        "best_reward_method": best_by_reward["method"],
        "best_reward": best_by_reward["avg_reward"],
    })


def evaluate_baseline(dataset, topk: int, use_ollama: bool, generator_model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Evaluate baseline (always retrieve) performance.

    Args:
        dataset: Dataset to evaluate on
        topk: Number of documents to retrieve
        use_ollama: Whether to use Ollama
        generator_model: Model name for generator

    Returns:
        Dictionary with evaluation results
    """
    from tqdm import tqdm
    from agents.flashrag_components import RAGPipeline
    from agents.reward import RAGRewardCalculator, compute_f1, compute_exact_match

    print("\n📊 Evaluating BASELINE (always retrieve)...")

    pipeline = RAGPipeline(topk=topk, generator_model=generator_model)
    reward_calculator = RAGRewardCalculator(retrieval_cost=0.1)
    
    results = []
    
    for task in tqdm(list(dataset), desc="Baseline"):
        question = task["question"]
        golden_answers = task["golden_answers"]
        
        # Always retrieve
        answer, docs = pipeline.answer(question, should_retrieve=True)
        
        # Calculate metrics
        reward, metrics = reward_calculator.compute_reward(
            prediction=answer,
            ground_truths=golden_answers,
            did_retrieve=True
        )
        
        results.append({
            "id": task["id"],
            "question": question,
            "golden_answers": golden_answers,
            "prediction": answer,
            "reward": reward,
            "f1": metrics["f1"],
            "em": metrics["em"],
            "did_retrieve": True
        })
    
    # Aggregate metrics
    return {
        "method": "baseline_always_retrieve",
        "num_samples": len(results),
        "avg_reward": sum(r["reward"] for r in results) / len(results),
        "avg_f1": sum(r["f1"] for r in results) / len(results),
        "avg_em": sum(r["em"] for r in results) / len(results),
        "retrieval_rate": 1.0,
        "results": results
    }


def evaluate_rl_agent(dataset, topk: int, use_ollama: bool, 
                      retrieve_rate: float = 0.7) -> Dict[str, Any]:
    """
    Evaluate RL agent with learned retrieval decisions.
    
    For now, simulates learned policy with probabilistic retrieval.
    In full implementation, this would load a trained policy.
    
    Args:
        dataset: Dataset to evaluate on
        topk: Number of documents to retrieve
        use_ollama: Whether to use Ollama
        retrieve_rate: Simulated retrieval rate (for testing)
        
    Returns:
        Dictionary with evaluation results
    """
    import random
    from tqdm import tqdm
    from agents.flashrag_components import RAGPipeline
    from agents.reward import RAGRewardCalculator
    
    print(f"\n📊 Evaluating RL AGENT (simulated {retrieve_rate:.0%} retrieval rate)...")
    
    pipeline = RAGPipeline(topk=topk)
    reward_calculator = RAGRewardCalculator(retrieval_cost=0.1)
    
    results = []
    random.seed(42)  # For reproducibility
    
    for task in tqdm(list(dataset), desc="RL Agent"):
        question = task["question"]
        golden_answers = task["golden_answers"]
        
        # Simulate learned retrieval decision
        # In real implementation, this would come from trained policy
        should_retrieve = random.random() < retrieve_rate
        
        # Execute
        answer, docs = pipeline.answer(question, should_retrieve=should_retrieve)
        
        # Calculate metrics
        reward, metrics = reward_calculator.compute_reward(
            prediction=answer,
            ground_truths=golden_answers,
            did_retrieve=should_retrieve
        )
        
        results.append({
            "id": task["id"],
            "question": question,
            "golden_answers": golden_answers,
            "prediction": answer,
            "reward": reward,
            "f1": metrics["f1"],
            "em": metrics["em"],
            "did_retrieve": should_retrieve
        })
    
    # Aggregate metrics
    actual_retrieval_rate = sum(1 for r in results if r["did_retrieve"]) / len(results)
    
    return {
        "method": "rl_agent",
        "num_samples": len(results),
        "avg_reward": sum(r["reward"] for r in results) / len(results),
        "avg_f1": sum(r["f1"] for r in results) / len(results),
        "avg_em": sum(r["em"] for r in results) / len(results),
        "retrieval_rate": actual_retrieval_rate,
        "results": results
    }


def evaluate_no_retrieval(dataset, topk: int, use_ollama: bool) -> Dict[str, Any]:
    """
    Evaluate performance without any retrieval (pure LLM).

    Args:
        dataset: Dataset to evaluate on
        topk: Number of documents to retrieve (unused)
        use_ollama: Whether to use Ollama

    Returns:
        Dictionary with evaluation results
    """
    from tqdm import tqdm
    from agents.flashrag_components import RAGPipeline
    from agents.reward import RAGRewardCalculator

    print("\n📊 Evaluating NO RETRIEVAL (pure LLM)...")

    pipeline = RAGPipeline(topk=topk)
    reward_calculator = RAGRewardCalculator(retrieval_cost=0.1)

    results = []

    for task in tqdm(list(dataset), desc="No Retrieval"):
        question = task["question"]
        golden_answers = task["golden_answers"]

        # Never retrieve
        answer, docs = pipeline.answer(question, should_retrieve=False)

        # Calculate metrics
        reward, metrics = reward_calculator.compute_reward(
            prediction=answer,
            ground_truths=golden_answers,
            did_retrieve=False
        )

        results.append({
            "id": task["id"],
            "question": question,
            "golden_answers": golden_answers,
            "prediction": answer,
            "reward": reward,
            "f1": metrics["f1"],
            "em": metrics["em"],
            "did_retrieve": False
        })

    # Aggregate metrics
    return {
        "method": "no_retrieval",
        "num_samples": len(results),
        "avg_reward": sum(r["reward"] for r in results) / len(results),
        "avg_f1": sum(r["f1"] for r in results) / len(results),
        "avg_em": sum(r["em"] for r in results) / len(results),
        "retrieval_rate": 0.0,
        "results": results
    }


def evaluate_trained_policy(
    dataset,
    topk: int,
    use_ollama: bool,
    checkpoint_path: str,
    eval_temperature: float = 0.7,
    analyze_topk: bool = False
) -> Dict[str, Any]:
    """
    Evaluate with actual trained policy network.

    Supports both binary and dynamic topk policies.

    Args:
        dataset: Dataset to evaluate on
        topk: Number of documents to retrieve (default for binary mode)
        use_ollama: Whether to use Ollama
        checkpoint_path: Path to trained checkpoint (best_model.pt)
        eval_temperature: Temperature for soft sampling
        analyze_topk: Whether to analyze topk distribution

    Returns:
        Dictionary with evaluation results
    """
    import torch
    from tqdm import tqdm
    from collections import Counter
    from agents.enhanced_pipeline import EnhancedRAGPipeline, DynamicTopKPolicyNetwork
    from agents.reward import RAGRewardCalculator

    print(f"\n📊 Evaluating TRAINED POLICY from {checkpoint_path}...")
    print(f"   Eval temperature: {eval_temperature}")

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    use_dynamic_topk = config.get("use_dynamic_topk", False)
    topk_options = config.get("topk_options", [0, 1, 3, 5, 7, 10])
    use_difficulty_features = config.get("use_difficulty_features", False)

    print(f"   Policy Mode: {'Dynamic TopK' if use_dynamic_topk else 'Binary'}")
    if use_dynamic_topk:
        print(f"   TopK Options: {topk_options}")

    # Initialize pipeline with matching config
    pipeline = EnhancedRAGPipeline(
        use_query_rewriter=False,
        use_learned_retrieval=True,
        use_ollama=use_ollama,
        generator_model=args.model,
        topk=topk,
        use_dynamic_topk=use_dynamic_topk,
        topk_options=topk_options if use_dynamic_topk else None,
        use_difficulty_features=use_difficulty_features
    )

    # Load checkpoint weights
    pipeline.policy_network.load_state_dict(checkpoint["policy_network"])
    pipeline.policy_network.eval()
    print(f"   Loaded checkpoint successfully!")

    # Configure reward calculator
    if use_dynamic_topk:
        reward_calculator = RAGRewardCalculator(
            retrieval_cost=0.1,
            use_dynamic_cost=True,
            base_retrieval_cost=0.05,
            per_doc_cost=0.01
        )
    else:
        reward_calculator = RAGRewardCalculator(retrieval_cost=0.1)

    results = []
    retrieval_probs = []
    topk_values = []

    for task in tqdm(list(dataset), desc="Trained Policy"):
        question = task["question"]
        golden_answers = task["golden_answers"]

        # Use trained policy
        with torch.no_grad():
            answer, docs, metadata = pipeline.answer(
                question,
                deterministic=True,
                temperature=eval_temperature
            )

        did_retrieve = metadata["did_retrieve"]
        topk_used = metadata.get("topk_used", 0)
        prob = metadata.get("retrieval_probability", 0.5)
        retrieval_probs.append(prob)
        topk_values.append(topk_used)

        # Calculate metrics
        reward, metrics = reward_calculator.compute_reward(
            prediction=answer,
            ground_truths=golden_answers,
            did_retrieve=did_retrieve,
            topk_used=topk_used
        )

        result_entry = {
            "id": task["id"],
            "question": question,
            "golden_answers": golden_answers,
            "prediction": answer,
            "reward": reward,
            "f1": metrics["f1"],
            "em": metrics["em"],
            "did_retrieve": did_retrieve,
            "topk_used": topk_used
        }

        if not use_dynamic_topk:
            result_entry["retrieval_probability"] = prob

        results.append(result_entry)

    # Aggregate metrics
    actual_retrieval_rate = sum(1 for r in results if r["did_retrieve"]) / len(results)
    avg_prob = sum(retrieval_probs) / len(retrieval_probs) if retrieval_probs else 0.5
    avg_topk = sum(topk_values) / len(topk_values) if topk_values else 0.0

    output = {
        "method": "trained_policy",
        "num_samples": len(results),
        "avg_reward": sum(r["reward"] for r in results) / len(results),
        "avg_f1": sum(r["f1"] for r in results) / len(results),
        "avg_em": sum(r["em"] for r in results) / len(results),
        "retrieval_rate": actual_retrieval_rate,
        "results": results
    }

    if use_dynamic_topk:
        output["avg_topk"] = avg_topk
        output["method"] = "trained_policy_dynamic_topk"

        # Compute topk distribution
        topk_counts = Counter(topk_values)
        topk_dist = {k: topk_counts.get(k, 0) / len(topk_values) for k in topk_options}
        output["topk_distribution"] = topk_dist

        print(f"\n   Average TopK: {avg_topk:.2f}")
        print(f"   TopK Distribution:")
        for k in topk_options:
            count = topk_counts.get(k, 0)
            pct = topk_dist[k] * 100
            print(f"      k={k}: {count} ({pct:.1f}%)")
    else:
        output["avg_retrieval_prob"] = avg_prob
        print(f"\n   Average retrieval probability: {avg_prob:.3f}")

    print(f"   Actual retrieval rate: {actual_retrieval_rate:.1%}")

    return output


def print_comparison(results: List[Dict[str, Any]]):
    """Print comparison table of results."""
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)

    # Check if any result has avg_topk
    has_topk = any("avg_topk" in r for r in results)

    if has_topk:
        print(f"{'Method':<35} {'F1':>10} {'EM':>10} {'Reward':>10} {'Retr.Rate':>10} {'AvgK':>8}")
        print("-" * 90)
        for r in results:
            avg_topk = r.get("avg_topk", "-")
            topk_str = f"{avg_topk:.1f}" if isinstance(avg_topk, (int, float)) else avg_topk
            print(f"{r['method']:<35} {r['avg_f1']:>10.2%} {r['avg_em']:>10.2%} "
                  f"{r['avg_reward']:>10.4f} {r['retrieval_rate']:>10.0%} {topk_str:>8}")
    else:
        print(f"{'Method':<30} {'F1':>10} {'EM':>10} {'Reward':>10} {'Retr.Rate':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['method']:<30} {r['avg_f1']:>10.2%} {r['avg_em']:>10.2%} "
                  f"{r['avg_reward']:>10.4f} {r['retrieval_rate']:>12.0%}")

    print("-" * 90 if has_topk else "-" * 80)

    # Calculate efficiency metrics
    if len(results) >= 2:
        baseline = next((r for r in results if "baseline" in r["method"]), results[0])

        print("\n📈 Efficiency Analysis:")
        for r in results:
            if r["method"] != baseline["method"]:
                f1_diff = r["avg_f1"] - baseline["avg_f1"]
                retrieval_saved = baseline["retrieval_rate"] - r["retrieval_rate"]
                reward_diff = r["avg_reward"] - baseline["avg_reward"]

                print(f"\n  {r['method']} vs {baseline['method']}:")
                print(f"    F1 difference: {f1_diff:+.2%}")
                print(f"    Retrieval saved: {retrieval_saved:.0%}")
                print(f"    Reward improvement: {reward_diff:+.4f}")

                # Show topk distribution if available
                if "topk_distribution" in r:
                    print(f"    TopK distribution: {r['topk_distribution']}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("RL-RAG Agent Evaluation")
    print("=" * 70)
    
    # Import dataset loader
    from agents.dataset import load_hotpotqa, load_custom_dataset
    
    # Load dataset
    if args.dataset == "custom":
        print(f"\nLoading custom dataset ({args.split})...")
        dataset = load_custom_dataset(args.split, sample_size=args.samples)
    else:
        print(f"\nLoading HotpotQA ({args.split})...")
        dataset = load_hotpotqa(args.split, sample_size=args.samples)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Check API key
    if not args.use_ollama and not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set!")
        print("\n   Options to set it:")
        print("   1. Export for this session:")
        print("      export OPENAI_API_KEY='your-key'")
        print("   2. Create .env file in project root:")
        print("      echo 'OPENAI_API_KEY=your-key' > .env")
        print("   3. Add to your shell config (~/.zshrc):")
        print("      echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.zshrc")
        print("   4. Use local Ollama instead:")
        print("      python evaluate_rl_agent.py --use-ollama")
        print("\n   After setting, run this script again.")
        return
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / f"experiments/results/agent_lightning/eval_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.wandb:
        init_wandb(args, args.dataset)
    
    # Run evaluation based on mode
    all_results = []

    if args.mode == "baseline":
        results = evaluate_baseline(dataset, args.topk, args.use_ollama, args.model)
        all_results.append(results)

    elif args.mode == "rl":
        results = evaluate_rl_agent(dataset, args.topk, args.use_ollama)
        all_results.append(results)

    elif args.mode == "trained":
        # Evaluate trained policy
        if args.checkpoint is None:
            print("\n❌ --checkpoint required for 'trained' mode!")
            print("   Example: --checkpoint experiments/results/rl_enhanced_xxx/best_model.pt")
            return
        if not Path(args.checkpoint).exists():
            print(f"\n❌ Checkpoint not found: {args.checkpoint}")
            return

        results = evaluate_trained_policy(
            dataset, args.topk, args.use_ollama,
            args.checkpoint, args.eval_temperature,
            analyze_topk=args.analyze_topk
        )
        all_results.append(results)

        # Also run baseline for comparison
        all_results.append(evaluate_baseline(dataset, args.topk, args.use_ollama, args.model))
        all_results.append(evaluate_no_retrieval(dataset, args.topk, args.use_ollama))

    elif args.mode == "compare":
        # Run all three evaluation modes
        all_results.append(evaluate_baseline(dataset, args.topk, args.use_ollama, args.model))
        all_results.append(evaluate_no_retrieval(dataset, args.topk, args.use_ollama))

        # If checkpoint provided, use trained policy
        if args.checkpoint and Path(args.checkpoint).exists():
            results = evaluate_trained_policy(
                dataset, args.topk, args.use_ollama,
                args.checkpoint, args.eval_temperature,
                analyze_topk=args.analyze_topk
            )
            all_results.append(results)
        else:
            # Simulate RL agent at different retrieval rates
            for rate in [0.8, 0.6, 0.4]:
                results = evaluate_rl_agent(dataset, args.topk, args.use_ollama, retrieve_rate=rate)
                results["method"] = f"rl_agent_{rate:.0%}_retrieve"
                all_results.append(results)

    elif args.mode == "efficiency":
        # Focus on efficiency analysis
        all_results.append(evaluate_baseline(dataset, args.topk, args.use_ollama, args.model))
        all_results.append(evaluate_no_retrieval(dataset, args.topk, args.use_ollama))
    
    # Print comparison
    print_comparison(all_results)
    
    # Log to wandb if enabled
    if args.wandb:
        log_results_to_wandb(all_results)
    
    # Save results
    output_file = Path(args.output_dir) / "evaluation_results.json"
    
    # Prepare serializable results (remove detailed results for smaller file)
    summary_results = []
    for r in all_results:
        summary = {k: v for k, v in r.items() if k != "results"}
        summary_results.append(summary)
    
    with open(output_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "summary": summary_results
        }, f, indent=2)
    
    # Save detailed results separately
    detailed_file = Path(args.output_dir) / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    
    # Save wandb run URL and close
    if args.wandb and _wandb_run is not None:
        wandb_info_file = Path(args.output_dir) / "wandb_info.txt"
        with open(wandb_info_file, 'w') as f:
            f.write(f"Wandb Run URL: {_wandb_run.get_url()}\n")
            f.write(f"Project: {args.wandb_project}\n")
            f.write(f"Run Name: {args.wandb_name}\n")
        print(f"📊 Wandb URL saved to: {wandb_info_file}")
        _wandb_run.finish()
        print("📊 Wandb run finished")


if __name__ == "__main__":
    main()

