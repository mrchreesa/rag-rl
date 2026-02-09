#!/usr/bin/env python3
"""
Training Script for RL-RAG Agent on HotpotQA

This script performs a "Hello World" training run to verify the RL setup.
It trains the retrieval decision agent on a subset of HotpotQA.

Usage:
    python train_hotpotqa.py --samples 500 --epochs 3
    
To run in dev mode (fast sanity check):
    python train_hotpotqa.py --dev --samples 50

To enable wandb logging:
    python train_hotpotqa.py --wandb --samples 100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Global wandb run reference
_wandb_run = None

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/rl/agent-lightning'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Load .env file if it exists
from utils.env_loader import load_env_file
if load_env_file(PROJECT_ROOT):
    print(f"✅ Loaded environment variables from {PROJECT_ROOT / '.env'}")

# Suppress verbose warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL-RAG Agent on HotpotQA")
    
    # Data arguments
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of training samples (default: 500)")
    parser.add_argument("--val-samples", type=int, default=100,
                        help="Number of validation samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--n-runners", type=int, default=1,
                        help="Number of parallel runners (default: 1)")
    
    # Model arguments
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--retrieval-cost", type=float, default=0.1,
                        help="Retrieval cost penalty (default: 0.1)")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use local Ollama instead of OpenAI API")
    
    # Mode arguments
    parser.add_argument("--dev", action="store_true",
                        help="Run in dev mode (fast sanity check)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run without actual training")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    
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


def main():
    args = parse_args()
    
    print("=" * 70)
    print("RL-RAG Agent Training on HotpotQA")
    print("=" * 70)
    
    # Import after path setup - handle missing dependencies gracefully
    try:
        from agents.dataset import load_hotpotqa, dataset_stats
    except ImportError as e:
        print(f"Error importing dataset module: {e}")
        return
    
    try:
        from agents.rl_rag_agent import RLRAGAgent
    except ImportError as e:
        print(f"Warning: Could not import RLRAGAgent: {e}")
        RLRAGAgent = None
    
    # Adjust samples for dev mode
    if args.dev:
        args.samples = min(args.samples, 50)
        args.val_samples = min(args.val_samples, 20)
        print(f"\n🔧 DEV MODE: Using {args.samples} train, {args.val_samples} val samples")
    
    # Load datasets
    print(f"\nLoading HotpotQA dataset...")
    train_dataset = load_hotpotqa("dev", sample_size=args.samples, seed=args.seed)
    val_dataset = load_hotpotqa("dev", sample_size=args.val_samples, seed=args.seed + 1)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Show dataset stats
    train_stats = dataset_stats(train_dataset)
    print(f"Question types: {train_stats.get('question_types', {})}")
    
    if args.dry_run:
        print("\n🏃 DRY RUN: Skipping actual training")
        print("Dataset loaded successfully!")
        return
    
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
        print("      python train_hotpotqa.py --use-ollama")
        print("\n   After setting, run this script again.")
        return
    
    # Initialize agent
    print(f"\nInitializing RL-RAG Agent...")
    print(f"  - Top-k documents: {args.topk}")
    print(f"  - Retrieval cost: {args.retrieval_cost}")
    print(f"  - Using Ollama: {args.use_ollama}")
    
    agent = RLRAGAgent(
        topk=args.topk,
        retrieval_cost=args.retrieval_cost,
        use_ollama=args.use_ollama
    )
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / f"experiments/results/agent_lightning/hotpotqa_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {args.output_dir}")
    
    # Initialize wandb if enabled
    if args.wandb:
        init_wandb(args)
    
    # For initial development/testing, use standalone evaluation
    # Agent Lightning's multiprocessing can have issues on macOS
    # Once the basic pipeline is verified, we can enable full training
    
    print(f"\n🚀 Running evaluation on {len(train_dataset)} samples...")
    print("   (Using standalone mode for reliable execution)")
    
    try:
        run_standalone_evaluation(agent, train_dataset, args)
    finally:
        # Close wandb if enabled
        if args.wandb and _wandb_run is not None:
            _wandb_run.finish()
            print("📊 Wandb run finished")


def init_wandb(args):
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
        args.wandb_name = f"hotpotqa_s{args.samples}_k{args.topk}_c{args.retrieval_cost}_{timestamp}"
    
    # Build config dict
    config = {
        "dataset": "hotpotqa",
        "samples": args.samples,
        "val_samples": args.val_samples,
        "seed": args.seed,
        "epochs": args.epochs,
        "topk": args.topk,
        "retrieval_cost": args.retrieval_cost,
        "use_ollama": args.use_ollama,
        "dev_mode": args.dev,
    }
    
    # Build tags
    tags = list(args.wandb_tags) if args.wandb_tags else []
    tags.append("hotpotqa")
    tags.append(f"topk={args.topk}")
    if args.dev:
        tags.append("dev")
    
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


def log_to_wandb(metrics: dict, step: int = None):
    """Log metrics to wandb if enabled."""
    global _wandb_run
    if _wandb_run is not None:
        _wandb_run.log(metrics, step=step)


def run_standalone_evaluation(agent, dataset, args):
    """
    Run standalone evaluation without Agent Lightning trainer.
    
    Useful for debugging or when Agent Lightning is not available.
    """
    print("\n🔄 Running standalone evaluation...")
    
    from tqdm import tqdm
    import json
    
    results = []
    running_f1 = 0.0
    running_em = 0.0
    running_reward = 0.0
    
    for i, task in enumerate(tqdm(dataset, desc="Processing")):
        # Process task
        agent._ensure_initialized()
        
        question = task["question"]
        golden_answers = task["golden_answers"]
        
        # Always retrieve for baseline
        answer, docs = agent._pipeline.answer(question, should_retrieve=True)
        
        # Calculate reward
        reward, metrics = agent.reward_calculator.compute_reward(
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
            "metrics": metrics
        })
        
        # Update running averages
        running_f1 += metrics["f1"]
        running_em += metrics["em"]
        running_reward += reward
        
        # Log to wandb every 10 samples (for live monitoring)
        if args.wandb and (i + 1) % 10 == 0:
            log_to_wandb({
                "sample": i + 1,
                "running_avg_f1": running_f1 / (i + 1),
                "running_avg_em": running_em / (i + 1),
                "running_avg_reward": running_reward / (i + 1),
                "current_f1": metrics["f1"],
                "current_em": metrics["em"],
                "current_reward": reward,
            }, step=i + 1)
    
    # Calculate aggregate metrics
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
    avg_em = sum(r["metrics"]["em"] for r in results) / len(results)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Samples evaluated: {len(results)}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average EM: {avg_em:.4f}")
    
    # Log final metrics to wandb
    if args.wandb:
        log_to_wandb({
            "final/avg_reward": avg_reward,
            "final/avg_f1": avg_f1,
            "final/avg_em": avg_em,
            "final/num_samples": len(results),
        })
        
        # Log summary metrics
        if _wandb_run is not None:
            _wandb_run.summary.update({
                "avg_reward": avg_reward,
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "num_samples": len(results),
                "topk": args.topk,
                "retrieval_cost": args.retrieval_cost,
            })
    
    # Save results
    output_file = Path(args.output_dir) / "standalone_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "aggregate_metrics": {
                "avg_reward": avg_reward,
                "avg_f1": avg_f1,
                "avg_em": avg_em,
                "num_samples": len(results)
            },
            "results": results
        }, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save wandb run URL if available
    if args.wandb and _wandb_run is not None:
        wandb_info_file = Path(args.output_dir) / "wandb_info.txt"
        with open(wandb_info_file, 'w') as f:
            f.write(f"Wandb Run URL: {_wandb_run.get_url()}\n")
            f.write(f"Project: {args.wandb_project}\n")
            f.write(f"Run Name: {args.wandb_name}\n")
        print(f"📊 Wandb URL saved to: {wandb_info_file}")


if __name__ == "__main__":
    main()

