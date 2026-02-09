#!/usr/bin/env python3
"""
Retroactively log all existing experiment results to Weights & Biases (W&B).

This script:
1. Scans experiments/results/baselines/ for all experiment directories
2. Parses experiment metadata from directory names and config files
3. Loads metrics from metric_score.txt
4. Logs everything to W&B for visualization and comparison
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

try:
    import yaml
except ImportError:
    print("❌ pyyaml not installed. Install with: pip install pyyaml")
    exit(1)

try:
    import wandb
except ImportError:
    print("❌ wandb not installed. Install with: pip install wandb")
    exit(1)

# Project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments/results/baselines"


def parse_experiment_name(dir_name: str) -> Dict[str, Any]:
    """
    Parse experiment directory name to extract metadata.
    
    Formats:
    - custom_dataset_YYYY_MM_DD_HH_MM_method_name
    - hotpotqa_YYYY_MM_DD_HH_MM_method_name
    """
    metadata = {
        "dataset": "unknown",
        "method": "unknown",
        "generator": "unknown",
        "retrieval_method": "unknown",
        "topk": None,
        "timestamp": None,
    }
    
    # Extract dataset
    if dir_name.startswith("custom_dataset"):
        metadata["dataset"] = "custom"
        parts = dir_name.replace("custom_dataset_", "").split("_")
    elif dir_name.startswith("hotpotqa"):
        metadata["dataset"] = "hotpotqa"
        parts = dir_name.replace("hotpotqa_", "").split("_")
    else:
        return metadata
    
    # Extract timestamp (YYYY_MM_DD_HH_MM)
    if len(parts) >= 5:
        try:
            timestamp_str = "_".join(parts[:5])
            metadata["timestamp"] = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M")
            method_parts = parts[5:]
        except:
            method_parts = parts
    else:
        method_parts = parts
    
    # Parse method name for generator, retrieval method, topk
    method_str = "_".join(method_parts)
    metadata["method"] = method_str
    
    # Extract generator
    if "openai" in method_str or "gpt" in method_str:
        if "gpt-4o-mini" in method_str:
            metadata["generator"] = "gpt-4o-mini"
        elif "gpt-4o" in method_str:
            metadata["generator"] = "gpt-4o"
        elif "gpt-3.5" in method_str:
            metadata["generator"] = "gpt-3.5-turbo"
        else:
            metadata["generator"] = "openai"
    elif "ollama" in method_str or "llama" in method_str:
        metadata["generator"] = "ollama"
    
    # Extract retrieval method
    if "dense" in method_str or "e5" in method_str:
        metadata["retrieval_method"] = "dense_e5"
    elif "bm25" in method_str:
        metadata["retrieval_method"] = "bm25"
    else:
        metadata["retrieval_method"] = "bm25"  # default
    
    # Extract topk
    topk_match = re.search(r"topk(\d+)", method_str)
    if topk_match:
        metadata["topk"] = int(topk_match.group(1))
    
    return metadata


def load_config_yaml(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load config.yaml file if it exists."""
    if not config_path.exists():
        return None
    
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load config.yaml: {e}")
        return None


def load_metrics(metric_path: Path) -> Optional[Dict[str, float]]:
    """Load metrics from metric_score.txt."""
    if not metric_path.exists():
        return None
    
    metrics = {}
    try:
        with open(metric_path) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = value.strip()
    except Exception as e:
        print(f"  ⚠️  Failed to load metrics: {e}")
        return None
    
    return metrics


def get_num_samples(intermediate_path: Path) -> Optional[int]:
    """Get number of samples from intermediate_data.json if available."""
    if not intermediate_path.exists():
        return None
    
    try:
        with open(intermediate_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
    except:
        pass
    
    return None


def log_experiment_to_wandb(exp_dir: Path, dry_run: bool = False) -> bool:
    """
    Log a single experiment to W&B.
    
    Returns:
        True if successful, False otherwise
    """
    dir_name = exp_dir.name
    print(f"\n📊 Processing: {dir_name}")
    
    # Parse metadata from directory name
    metadata = parse_experiment_name(dir_name)
    
    # Load config.yaml if available
    config_path = exp_dir / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Merge config with metadata (config takes precedence)
    if config:
        metadata["retrieval_method"] = config.get("retrieval_method", metadata["retrieval_method"])
        metadata["topk"] = config.get("retrieval_topk", metadata["topk"])
        metadata["generator"] = config.get("generator_model", metadata["generator"])
        if "openai_setting" in config:
            if config["openai_setting"].get("api_key") == "ollama":
                metadata["generator"] = "ollama"
            elif "gpt" in str(config.get("generator_model", "")):
                metadata["generator"] = config.get("generator_model", metadata["generator"])
    
    # Load metrics
    metric_path = exp_dir / "metric_score.txt"
    metrics = load_metrics(metric_path)
    
    if not metrics:
        print(f"  ⚠️  No metrics found, skipping...")
        return False
    
    # Get number of samples
    intermediate_path = exp_dir / "intermediate_data.json"
    num_samples = get_num_samples(intermediate_path)
    
    # Build W&B config
    wandb_config = {
        "dataset": metadata["dataset"],
        "method": metadata["method"],
        "generator": metadata["generator"],
        "retrieval_method": metadata["retrieval_method"],
        "topk": metadata["topk"],
        "num_samples": num_samples,
    }
    
    # Add config.yaml values if available
    if config:
        wandb_config.update({
            "framework": config.get("framework", "unknown"),
            "split": config.get("split", ["test"])[0] if isinstance(config.get("split"), list) else config.get("split", "test"),
            "seed": config.get("seed", None),
        })
    
    # Build tags
    tags = [
        metadata["dataset"],
        metadata["method"].split("_")[0] if "_" in metadata["method"] else metadata["method"],
        metadata["generator"],
        metadata["retrieval_method"],
    ]
    
    # Create run name (avoid duplication if topk already in method name)
    run_name = f"{metadata['dataset']}_{metadata['method']}"
    if metadata["topk"] and f"topk{metadata['topk']}" not in metadata['method']:
        run_name += f"_topk{metadata['topk']}"
    
    if dry_run:
        print(f"  [DRY RUN] Would log:")
        print(f"    Run name: {run_name}")
        print(f"    Config: {wandb_config}")
        print(f"    Metrics: {metrics}")
        print(f"    Tags: {tags}")
        return True
    
    # Initialize W&B run
    try:
        wandb.init(
            project="rag-baselines",
            name=run_name,
            config=wandb_config,
            tags=tags,
            reinit=True,  # Allow re-running this script
            id=None,  # Generate new run ID
        )
        
        # Log final metrics
        wandb.log({
            "final/em": metrics.get("em", 0),
            "final/f1": metrics.get("f1", 0),
            "final/retrieval_recall": metrics.get("retrieval_recall_top5", 
                                                   metrics.get("retrieval_recall", 0)),
        })
        
        # Update summary (shown in main table)
        wandb.summary.update({
            "EM": metrics.get("em", 0),
            "F1": metrics.get("f1", 0),
            "Retrieval_Recall": metrics.get("retrieval_recall_top5", 
                                            metrics.get("retrieval_recall", 0)),
            "Num_Samples": num_samples or 0,
        })
        
        wandb.finish()
        print(f"  ✅ Logged to W&B")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to log to W&B: {e}")
        return False


def main():
    """Main function to log all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Log existing experiment results to Weights & Biases"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be logged without actually logging"
    )
    parser.add_argument(
        "--project",
        default="rag-baselines",
        help="W&B project name (default: rag-baselines)"
    )
    parser.add_argument(
        "--filter",
        help="Only process experiments matching this pattern (e.g., 'custom_dataset')"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("W&B RETROACTIVE LOGGING")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"W&B project: {args.project}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No data will be logged")
    else:
        # Check if wandb is logged in
        if not wandb.api.api_key:
            print("\n❌ W&B not logged in. Run: wandb login")
            return
        
        print(f"\n✅ Logged in to W&B as: {wandb.api.viewer()}")
    
    if not RESULTS_DIR.exists():
        print(f"\n❌ Results directory not found: {RESULTS_DIR}")
        return
    
    # Find all experiment directories
    exp_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    if args.filter:
        exp_dirs = [d for d in exp_dirs if args.filter in d.name]
    
    if not exp_dirs:
        print(f"\n❌ No experiment directories found")
        return
    
    print(f"\nFound {len(exp_dirs)} experiment(s) to process")
    
    # Process each experiment
    success_count = 0
    failed_count = 0
    
    for exp_dir in sorted(exp_dirs):
        success = log_experiment_to_wandb(exp_dir, dry_run=args.dry_run)
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed/Skipped: {failed_count}")
    print(f"📊 Total: {len(exp_dirs)}")
    
    if not args.dry_run and success_count > 0:
        print(f"\n🌐 View results at: https://wandb.ai/{wandb.api.viewer()['username']}/{args.project}")
        print("=" * 70)


if __name__ == "__main__":
    main()

