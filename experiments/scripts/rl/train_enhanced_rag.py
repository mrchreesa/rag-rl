#!/usr/bin/env python3
"""
Train RL-Enhanced RAG Pipeline

This script trains the RL-enhanced RAG pipeline with:
1. Neural policy network for retrieval decisions
2. RL-based query rewriting
3. REINFORCE training

Usage:
    python train_enhanced_rag.py --samples 500 --epochs 5
    python train_enhanced_rag.py --dev  # Quick test with 50 samples
    python train_enhanced_rag.py --no-rewriter  # Policy only, no query rewriting

For full experiment:
    python train_enhanced_rag.py --samples 1000 --epochs 10 --wandb
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Load environment
from utils.env_loader import load_env_file
if load_env_file(PROJECT_ROOT):
    print(f"✅ Loaded environment from {PROJECT_ROOT / '.env'}")

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL-Enhanced RAG")

    # Data args
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["hotpotqa", "custom", "combined"],
                        help="Dataset to train on (default: custom for dissertation)")
    parser.add_argument("--samples", type=int, default=492,
                        help="Training samples (default: 492 = full custom train set)")
    parser.add_argument("--val-samples", type=int, default=87,
                        help="Validation samples (default: 87 = full custom test set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Training args
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--retrieval-cost", type=float, default=0.1,
                        help="Retrieval cost penalty (binary mode)")
    parser.add_argument("--wrong-no-retrieval-penalty", type=float, default=0.3,
                        help="Penalty for bad answers without retrieval")
    parser.add_argument("--update-every", type=int, default=5,
                        help="Policy update frequency")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Starting exploration rate")

    # Entropy and temperature
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient (higher = more exploration)")
    parser.add_argument("--eval-temperature", type=float, default=0.7,
                        help="Temperature for soft sampling during evaluation")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--curriculum-phases", type=int, default=3,
                        help="Number of curriculum learning phases")
    parser.add_argument("--use-difficulty-features", action="store_true",
                        help="Use question difficulty features for policy")

    # Dynamic TopK arguments
    parser.add_argument("--dynamic-topk", action="store_true",
                        help="Enable dynamic topk selection (learns optimal k per question)")
    parser.add_argument("--topk-options", type=str, default="0,1,3,5,7,10",
                        help="Comma-separated topk options (default: 0,1,3,5,7,10)")
    parser.add_argument("--base-retrieval-cost", type=float, default=0.05,
                        help="Fixed cost for any retrieval (dynamic topk mode)")
    parser.add_argument("--per-doc-cost", type=float, default=0.01,
                        help="Cost per document retrieved (dynamic topk mode)")

    # Feature flags
    parser.add_argument("--no-rewriter", action="store_true",
                        help="Disable query rewriting")
    parser.add_argument("--no-policy", action="store_true",
                        help="Disable learned retrieval policy")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama instead of OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Generator model (e.g. gpt-4o-mini, claude-3-5-haiku-20241022)")

    # Mode args
    parser.add_argument("--dev", action="store_true",
                        help="Dev mode (50 samples, 2 epochs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run without training")

    # Output args
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")

    # Wandb args
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rl-rag-enhanced",
                        help="W&B project name")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    import torch
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("=" * 70)
    print("RL-Enhanced RAG Training")
    print("=" * 70)
    print(f"🔧 Random seed: {args.seed}")
    
    # Adjust for dev mode
    if args.dev:
        args.samples = min(args.samples, 50)
        args.val_samples = min(args.val_samples, 20)
        args.epochs = 2
        print(f"\n🔧 DEV MODE: {args.samples} train, {args.val_samples} val, {args.epochs} epochs")
    
    # Load dataset
    from agents.dataset import load_hotpotqa, load_custom_dataset, load_combined_dataset
    
    print(f"\nLoading {args.dataset} dataset...")
    
    if args.dataset == "hotpotqa":
        train_data = list(load_hotpotqa("dev", sample_size=args.samples, seed=args.seed))
        val_data = list(load_hotpotqa("dev", sample_size=args.val_samples, seed=args.seed + 1))
    elif args.dataset == "custom":
        train_data = list(load_custom_dataset("train", sample_size=args.samples, seed=args.seed))
        val_data = list(load_custom_dataset("test", sample_size=args.val_samples, seed=args.seed))
    else:
        combined = list(load_combined_dataset(args.samples, args.val_samples // 2, seed=args.seed))
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    
    if args.dry_run:
        print("\n🏃 DRY RUN - Skipping training")
        return
    
    # Check API key
    if not args.use_ollama and not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print("   Or use: python train_enhanced_rag.py --use-ollama")
        return
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / f"experiments/results/rl_enhanced_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output: {args.output_dir}")
    
    # Parse topk options
    topk_options = [int(x.strip()) for x in args.topk_options.split(",")]

    # Initialize pipeline
    print("\n🔧 Initializing Enhanced RAG Pipeline...")

    from agents.enhanced_pipeline import EnhancedRAGPipeline, RLTrainer

    pipeline = EnhancedRAGPipeline(
        use_query_rewriter=not args.no_rewriter,
        use_learned_retrieval=not args.no_policy,
        use_ollama=args.use_ollama,
        generator_model=args.model,
        topk=5,
        use_difficulty_features=args.use_difficulty_features,
        use_dynamic_topk=args.dynamic_topk,
        topk_options=topk_options if args.dynamic_topk else None
    )

    print(f"   Query Rewriter: {'Enabled' if not args.no_rewriter else 'Disabled'}")
    print(f"   Learned Policy: {'Enabled' if not args.no_policy else 'Disabled'}")
    print(f"   Generator: {'Ollama' if args.use_ollama else 'GPT-4o-mini'}")
    print(f"   Difficulty Features: {'Enabled' if args.use_difficulty_features else 'Disabled'}")
    print(f"   Policy Mode: {'Dynamic TopK' if args.dynamic_topk else 'Binary'}")
    if args.dynamic_topk:
        print(f"   TopK Options: {topk_options}")

    # Initialize trainer
    trainer = RLTrainer(
        pipeline=pipeline,
        retrieval_cost=args.retrieval_cost,
        wrong_no_retrieval_penalty=args.wrong_no_retrieval_penalty,
        entropy_coef=args.entropy_coef,
        eval_temperature=args.eval_temperature,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        # Dynamic TopK cost parameters
        use_dynamic_cost=args.dynamic_topk,
        base_retrieval_cost=args.base_retrieval_cost,
        per_doc_cost=args.per_doc_cost
    )

    print(f"   Wrong no-retrieval penalty: {args.wrong_no_retrieval_penalty}")
    print(f"   Entropy coefficient: {args.entropy_coef}")
    print(f"   Eval temperature: {args.eval_temperature}")
    print(f"   Curriculum learning: {'Disabled' if args.no_curriculum else 'Enabled'}")
    if args.dynamic_topk:
        print(f"   Base retrieval cost: {args.base_retrieval_cost}")
        print(f"   Per-doc cost: {args.per_doc_cost}")

    
    # Initialize wandb if enabled
    if args.wandb:
        try:
            import wandb

            # Create descriptive run name
            timestamp = datetime.now().strftime("%m%d_%H%M")
            mode_tag = "dyn-topk" if args.dynamic_topk else "binary"
            run_name = f"{args.dataset}_{mode_tag}_e{args.epochs}_{timestamp}"

            # Comprehensive config for experiment tracking
            wandb_config = {
                # Data
                "dataset": args.dataset,
                "train_samples": args.samples,
                "val_samples": args.val_samples,
                "seed": args.seed,
                # Training
                "epochs": args.epochs,
                "update_every": args.update_every,
                "start_epsilon": args.epsilon,
                # Cost parameters
                "retrieval_cost": args.retrieval_cost,
                "wrong_no_retrieval_penalty": args.wrong_no_retrieval_penalty,
                "entropy_coef": args.entropy_coef,
                "eval_temperature": args.eval_temperature,
                # Curriculum
                "use_curriculum": not args.no_curriculum,
                "curriculum_phases": args.curriculum_phases,
                # Features
                "use_query_rewriter": not args.no_rewriter,
                "use_learned_retrieval": not args.no_policy,
                "use_difficulty_features": args.use_difficulty_features,
                "use_ollama": args.use_ollama,
                "generator_model": "ollama" if args.use_ollama else "gpt-4o-mini",
                # Dynamic TopK
                "use_dynamic_topk": args.dynamic_topk,
                "topk_options": topk_options if args.dynamic_topk else None,
                "base_retrieval_cost": args.base_retrieval_cost if args.dynamic_topk else None,
                "per_doc_cost": args.per_doc_cost if args.dynamic_topk else None,
            }

            tags = [
                args.dataset,
                mode_tag,
                f"epochs-{args.epochs}",
                "curriculum" if not args.no_curriculum else "no-curriculum",
            ]

            wandb.init(
                project=args.wandb_project,
                config=wandb_config,
                name=run_name,
                tags=tags
            )
            print(f"\n📊 W&B: {wandb.run.get_url()}")
        except ImportError:
            print("⚠️  wandb not installed, continuing without")
            args.wandb = False
    
    # Train with curriculum learning
    print("\n" + "=" * 70)
    results = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        update_every=args.update_every,
        start_epsilon=args.epsilon,
        use_curriculum=not args.no_curriculum,
        curriculum_phases=args.curriculum_phases
    )

    
    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best F1: {results['best_f1']:.4f} (epoch {results['best_epoch']})")

    history = results['history']
    if history['val_f1']:
        print(f"Final Val F1: {history['val_f1'][-1]:.4f}")
        print(f"Final Val Retrieval Rate: {history['val_retrieval_rate'][-1]:.1%}")

    # Print cost summary
    usage_stats = results.get('usage_stats', {})
    if usage_stats:
        print(f"\n💰 API Cost Summary:")
        print(f"   Model: {usage_stats.get('model', 'unknown')}")
        print(f"   Total API calls: {usage_stats.get('total_calls', 0):,}")
        print(f"   Total tokens: {usage_stats.get('total_tokens', 0):,}")
        if not usage_stats.get('is_local', False):
            print(f"   Estimated cost: ${usage_stats.get('total_cost_usd', 0):.4f}")
        else:
            print(f"   Cost: $0.00 (local model)")

    # Log to wandb
    if args.wandb:
        log_data = {
            "best_f1": results['best_f1'],
            "best_epoch": results['best_epoch'],
            "final_retrieval_rate": history['val_retrieval_rate'][-1] if history['val_retrieval_rate'] else 1.0
        }
        # Add cost tracking to wandb
        if usage_stats:
            log_data["total_api_calls"] = usage_stats.get('total_calls', 0)
            log_data["total_tokens"] = usage_stats.get('total_tokens', 0)
            log_data["total_cost_usd"] = usage_stats.get('total_cost_usd', 0)
        wandb.log(log_data)
        wandb.finish()
    
    print(f"\n✅ Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
