"""
Controlled Comparison: Fair Evaluation of All Configurations

Runs all configurations through EnhancedRAGPipeline with the SAME F1 function
(reward.py:compute_f1 with normalize_answer) to produce a fair comparison.

Configurations:
  k=0  (parametric baseline)  - generate_direct(), no retrieval
  k=1  (fixed)                - always retrieve 1 doc
  k=3  (fixed)                - always retrieve 3 docs
  k=5  (fixed)                - always retrieve 5 docs
  k=10 (fixed)                - always retrieve 10 docs
  RL Dynamic TopK             - learned policy from checkpoint

Usage:
    python experiments/scripts/eval/controlled_comparison.py [--wandb] [--checkpoint PATH]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

# Load .env file (API keys)
from utils.env_loader import load_env_file
load_env_file(PROJECT_ROOT)

import torch

from agents.flashrag_components import DenseRetrieverWrapper, GeneratorWrapper
from agents.enhanced_pipeline import EnhancedRAGPipeline, DynamicTopKPolicyNetwork, QueryRewritePolicyNetwork
from agents.query_rewriter import StrategyRewriter
from agents.reward import compute_f1, compute_exact_match


def load_dataset(path: str):
    """Load JSONL dataset."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_config(
    name: str,
    pipeline: EnhancedRAGPipeline,
    test_data: list,
    topk: int,
    use_rl: bool = False,
    verbose: bool = True
) -> dict:
    """
    Evaluate a single configuration on all test samples.

    Returns dict with aggregate metrics and per-sample predictions.
    """
    predictions = []
    f1_scores = []
    em_scores = []
    topk_values = []

    for i, sample in enumerate(test_data):
        question = sample["question"]
        golden_answers = sample["golden_answers"]

        metadata = {}
        if use_rl:
            # RL Dynamic TopK: let the policy decide
            answer, docs, metadata = pipeline.answer(
                question,
                deterministic=True,
                temperature=0.7
            )
            topk_used = metadata.get("topk_used", 0)
            did_retrieve = metadata["did_retrieve"]
        elif topk == 0:
            # Parametric baseline: no retrieval
            answer = pipeline.generator.generate_direct(question)
            docs = None
            topk_used = 0
            did_retrieve = False
        else:
            # Fixed-k: always retrieve with this topk
            docs = pipeline.retriever.retrieve([question], topk=topk)[0]
            answer = pipeline.generator.generate_with_retrieval(question, docs)
            topk_used = topk
            did_retrieve = True

        f1 = compute_f1(answer, golden_answers)
        em = compute_exact_match(answer, golden_answers)

        f1_scores.append(f1)
        em_scores.append(em)
        topk_values.append(topk_used)

        # Build retrieved context string from actual docs
        retrieved_text = ""
        if docs and did_retrieve:
            if isinstance(docs, list):
                retrieved_text = "\n\n".join(
                    d.get("contents", d) if isinstance(d, dict) else str(d)
                    for d in docs[:topk_used]
                )

        pred_entry = {
            "id": sample.get("id", f"sample_{i}"),
            "question": question,
            "golden_answers": golden_answers,
            "prediction": answer,
            "f1": f1,
            "em": em,
            "topk_used": topk_used,
            "did_retrieve": did_retrieve,
            "config": name,
            "retrieved_text": retrieved_text,
        }
        # Add rewrite metadata if present
        if use_rl and metadata:
            if metadata.get("rewrite_strategy") is not None:
                pred_entry["rewrite_strategy"] = metadata["rewrite_strategy"]
                pred_entry["rewrite_strategy_name"] = metadata.get("rewrite_strategy_name", "")
            if metadata.get("rewritten_question") is not None:
                pred_entry["rewritten_question"] = metadata["rewritten_question"]
        # Propagate metadata (e.g., HotpotQA level/type)
        if "metadata" in sample:
            pred_entry["metadata"] = sample["metadata"]
        predictions.append(pred_entry)

        if verbose and (i + 1) % 10 == 0:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            print(f"  [{i+1}/{len(test_data)}] Running F1: {avg_f1:.4f}")

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    avg_topk = sum(topk_values) / len(topk_values)
    retrieval_rate = sum(1 for t in topk_values if t > 0) / len(topk_values)

    results = {
        "config": name,
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "avg_topk": avg_topk,
        "retrieval_rate": retrieval_rate,
        "n_samples": len(test_data),
        "predictions": predictions
    }

    if verbose:
        print(f"  {name}: F1={avg_f1:.4f}, EM={avg_em:.4f}, AvgK={avg_topk:.1f}, Retr={retrieval_rate:.1%}")

    return results


def load_rl_checkpoint(pipeline: EnhancedRAGPipeline, checkpoint_path: str):
    """Load a trained RL checkpoint into the pipeline."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    topk_options = config.get("topk_options", [0, 1, 3, 5, 7, 10])

    # Verify network architecture matches
    if isinstance(pipeline.policy_network, DynamicTopKPolicyNetwork):
        pipeline.policy_network.load_state_dict(checkpoint["policy_network"])
        if pipeline.device:
            pipeline.policy_network.to(pipeline.device)
        pipeline.policy_network.eval()
        print(f"  Loaded topk policy from {checkpoint_path}")
        print(f"  TopK options: {topk_options}")
    else:
        raise ValueError("Pipeline policy network type doesn't match checkpoint")

    # Load rewrite policy if present
    if "rewrite_policy_network" in checkpoint and pipeline.rewrite_policy_network is not None:
        pipeline.rewrite_policy_network.load_state_dict(checkpoint["rewrite_policy_network"])
        if pipeline.device:
            pipeline.rewrite_policy_network.to(pipeline.device)
        pipeline.rewrite_policy_network.eval()
        print(f"  Loaded rewrite policy (strategies: {QueryRewritePolicyNetwork.STRATEGY_NAMES})")


def print_results_table(all_results: list):
    """Print a markdown table of results."""
    print("\n## Controlled Comparison Results")
    print()
    print("| Configuration | F1 (%) | EM (%) | Avg K | Retr Rate |")
    print("|--------------|--------|--------|-------|-----------|")

    for r in all_results:
        f1_pct = r["avg_f1"] * 100
        em_pct = r["avg_em"] * 100
        name = r["config"]
        # Bold the best F1
        print(f"| {name} | {f1_pct:.2f}% | {em_pct:.2f}% | {r['avg_topk']:.1f} | {r['retrieval_rate']:.0%} |")

    # Find best
    best = max(all_results, key=lambda r: r["avg_f1"])
    print(f"\nBest: **{best['config']}** (F1: {best['avg_f1']*100:.2f}%)")


def get_dataset_paths(dataset_name: str):
    """Get dataset, index, and corpus paths for a given dataset."""
    if dataset_name == "hotpotqa":
        return {
            "test_path": "data/benchmarks/hotpotqa/dev.jsonl",
            "index_path": str(PROJECT_ROOT / "data/indexes/wiki_hotpotqa_e5/e5_Flat.index"),
            "corpus_path": str(PROJECT_ROOT / "data/corpus/wiki/wiki_hotpotqa_subset.jsonl"),
        }
    else:  # custom
        return {
            "test_path": "data/datasets/custom_dataset/test.jsonl",
            "index_path": None,  # Use defaults
            "corpus_path": None,
        }


def main():
    parser = argparse.ArgumentParser(description="Controlled comparison of all RAG configurations")
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/results/rl_enhanced_20260208_223508/best_model.pt",
                        help="Path to RL checkpoint (relative to project root)")
    parser.add_argument("--dataset", type=str,
                        default="custom",
                        choices=["custom", "hotpotqa"],
                        help="Dataset to evaluate on (default: custom)")
    parser.add_argument("--test-path", type=str, default=None,
                        help="Override path to test dataset (relative to project root)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max test samples to evaluate (default: all)")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama instead of OpenAI for generation")
    parser.add_argument("--output-dir", type=str,
                        default=None,
                        help="Output directory for results")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (when --max-samples is used)")
    args = parser.parse_args()

    # Resolve dataset paths
    ds_paths = get_dataset_paths(args.dataset)
    test_path_str = args.test_path or ds_paths["test_path"]

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    dataset_path = PROJECT_ROOT / test_path_str

    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_suffix = "_ollama" if args.use_ollama else ""
        output_dir = PROJECT_ROOT / f"experiments/results/controlled_comparison_{args.dataset}{gen_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONTROLLED COMPARISON: Fair Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({dataset_path})")
    print(f"Generator: {'Ollama' if args.use_ollama else 'GPT-4o-mini'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"F1 function: reward.py:compute_f1 (normalize_answer)")
    print()

    # Load test data
    test_data = load_dataset(str(dataset_path))
    print(f"Loaded {len(test_data)} test samples")

    # Optionally subsample
    if args.max_samples and args.max_samples < len(test_data):
        import random
        random.seed(args.seed)
        test_data = random.sample(test_data, args.max_samples)
        print(f"Subsampled to {len(test_data)} test samples (seed={args.seed})")

    # Initialize shared components (ONE retriever, ONE generator)
    print("\nInitializing shared components...")
    if ds_paths["index_path"] is not None:
        retriever = DenseRetrieverWrapper(
            index_path=ds_paths["index_path"],
            corpus_path=ds_paths["corpus_path"]
        )
    else:
        retriever = DenseRetrieverWrapper()

    generator = GeneratorWrapper(
        model="gpt-4o-mini" if not args.use_ollama else "gpt-4o-mini",
        use_ollama=args.use_ollama
    )

    # Initialize W&B if requested
    if args.wandb:
        import wandb
        gen_tag = "ollama" if args.use_ollama else "gpt4omini"
        wandb.init(
            project="rl-rag-enhanced",
            name=f"controlled_comparison_{args.dataset}_{gen_tag}_{datetime.now().strftime('%m%d_%H%M')}",
            tags=["controlled-comparison", "evaluation", args.dataset, gen_tag],
            config={
                "dataset": args.dataset,
                "n_test_samples": len(test_data),
                "checkpoint": str(checkpoint_path),
                "use_ollama": args.use_ollama,
                "configs": ["k=0", "k=1", "k=3", "k=5", "k=10", "RL Dynamic TopK"],
                "f1_function": "reward.py:compute_f1"
            }
        )

    # Initialize energy tracking
    emissions_tracker = None
    try:
        from codecarbon import EmissionsTracker
        emissions_tracker = EmissionsTracker(
            project_name="rl-rag-evaluation",
            output_dir=str(output_dir),
            output_file="emissions.csv",
            log_level="warning",
            tracking_mode="process",
        )
        emissions_tracker.start()
        print("⚡ Energy tracking: Enabled (CodeCarbon)")
    except ImportError:
        print("⚡ Energy tracking: Disabled (install codecarbon)")
    except Exception as e:
        print(f"⚡ Energy tracking: Failed to start ({e})")

    all_results = []

    # --- Fixed-k configurations ---
    fixed_k_configs = [
        ("k=0 (parametric)", 0),
        ("k=1 (fixed)", 1),
        ("k=3 (fixed)", 3),
        ("k=5 (fixed)", 5),
        ("k=10 (fixed)", 10),
    ]

    # Create a simple pipeline for fixed-k (no policy network needed)
    fixed_pipeline = EnhancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        use_learned_retrieval=False,
        use_query_rewriter=False,
        use_dynamic_topk=False,
        topk=5  # default, overridden per config
    )

    for name, k in fixed_k_configs:
        print(f"\nEvaluating: {name}")
        start = time.time()
        results = evaluate_config(
            name=name,
            pipeline=fixed_pipeline,
            test_data=test_data,
            topk=k,
            use_rl=False,
            verbose=args.verbose
        )
        elapsed = time.time() - start
        results["elapsed_seconds"] = elapsed
        all_results.append(results)

        if args.wandb:
            import wandb
            wandb.log({
                f"comparison/{name}/f1": results["avg_f1"],
                f"comparison/{name}/em": results["avg_em"],
                f"comparison/{name}/avg_topk": results["avg_topk"],
            })

    # --- RL Dynamic TopK ---
    # Load checkpoint config to detect features
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    ckpt_config = ckpt.get("config", {})
    topk_options = ckpt_config.get("topk_options", [0, 1, 3, 5, 7, 10])
    use_difficulty = ckpt_config.get("use_difficulty_features", False)
    has_rewrite_policy = "rewrite_policy_network" in ckpt

    if has_rewrite_policy:
        config_name = "RL Dynamic TopK + Learned Rewrite"
    else:
        config_name = "RL Dynamic TopK"

    print(f"\nEvaluating: {config_name}")

    rl_pipeline = EnhancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        use_learned_retrieval=True,
        use_query_rewriter=False,
        use_dynamic_topk=True,
        topk_options=topk_options,
        use_difficulty_features=use_difficulty,
        use_learned_rewrite=has_rewrite_policy,
        use_ollama=args.use_ollama
    )

    load_rl_checkpoint(rl_pipeline, str(checkpoint_path))

    start = time.time()
    rl_results = evaluate_config(
        name=config_name,
        pipeline=rl_pipeline,
        test_data=test_data,
        topk=0,  # not used for RL
        use_rl=True,
        verbose=args.verbose
    )
    elapsed = time.time() - start
    rl_results["elapsed_seconds"] = elapsed

    # Add topk distribution for RL
    from collections import Counter
    rl_topks = [p["topk_used"] for p in rl_results["predictions"]]
    topk_counts = Counter(rl_topks)
    rl_results["topk_distribution"] = {
        k: topk_counts.get(k, 0) / len(rl_topks) for k in topk_options
    }

    # Add rewrite strategy distribution if applicable
    if has_rewrite_policy:
        rewrite_strategies = [p.get("rewrite_strategy") for p in rl_results["predictions"] if p.get("rewrite_strategy") is not None]
        if rewrite_strategies:
            strategy_counts = Counter(rewrite_strategies)
            rl_results["rewrite_strategy_distribution"] = {
                StrategyRewriter.STRATEGY_NAMES[s]: strategy_counts.get(s, 0) / len(rewrite_strategies)
                for s in range(QueryRewritePolicyNetwork.NUM_STRATEGIES)
            }

    all_results.append(rl_results)

    if args.wandb:
        import wandb
        log_dict = {
            f"comparison/{config_name}/f1": rl_results["avg_f1"],
            f"comparison/{config_name}/em": rl_results["avg_em"],
            f"comparison/{config_name}/avg_topk": rl_results["avg_topk"],
        }
        if has_rewrite_policy and "rewrite_strategy_distribution" in rl_results:
            for name, frac in rl_results["rewrite_strategy_distribution"].items():
                log_dict[f"comparison/{config_name}/rewrite_dist/{name}"] = frac
        wandb.log(log_dict)

    # --- Print results ---
    print_results_table(all_results)

    # Print RL topk distribution
    print(f"\nRL TopK Distribution:")
    for k in sorted(rl_results["topk_distribution"].keys()):
        pct = rl_results["topk_distribution"][k] * 100
        bar = "#" * int(pct / 2)
        print(f"  k={k:>2}: {pct:5.1f}% {bar}")

    # Print rewrite strategy distribution if applicable
    if has_rewrite_policy and "rewrite_strategy_distribution" in rl_results:
        print(f"\nRewrite Strategy Distribution:")
        for name, frac in sorted(rl_results["rewrite_strategy_distribution"].items(), key=lambda x: -x[1]):
            pct = frac * 100
            bar = "#" * int(pct / 2)
            print(f"  {name:>15s}: {pct:5.1f}% {bar}")

    # Save per-sample predictions for each config
    for r in all_results:
        safe_name = r["config"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        pred_path = output_dir / f"predictions_{safe_name}.jsonl"
        with open(pred_path, "w") as f:
            for p in r["predictions"]:
                f.write(json.dumps(p) + "\n")

    print(f"Saved per-sample predictions to {output_dir}/predictions_*.jsonl")

    # Print API usage
    generator.print_usage_summary()

    # Stop energy tracking and report
    emissions_data = None
    if emissions_tracker is not None:
        try:
            total_emissions = emissions_tracker.stop()
            emissions_data = {
                "total_emissions_kg_co2": total_emissions,
                "total_energy_kwh": emissions_tracker.final_emissions_data.energy_consumed,
                "duration_seconds": emissions_tracker.final_emissions_data.duration,
                "cpu_energy_kwh": emissions_tracker.final_emissions_data.cpu_energy,
                "ram_energy_kwh": emissions_tracker.final_emissions_data.ram_energy,
                "gpu_energy_kwh": emissions_tracker.final_emissions_data.gpu_energy,
                "country_iso_code": emissions_tracker.final_emissions_data.country_iso_code,
            }

            # Per-config energy estimate (proportional to elapsed time)
            total_elapsed = sum(r.get("elapsed_seconds", 0) for r in all_results)
            if total_elapsed > 0:
                for r in all_results:
                    frac = r.get("elapsed_seconds", 0) / total_elapsed
                    r["energy_kwh"] = emissions_data["total_energy_kwh"] * frac
                    r["co2_kg"] = emissions_data["total_emissions_kg_co2"] * frac

            print(f"\n⚡ Energy & Carbon Summary:")
            print(f"   Total energy: {emissions_data['total_energy_kwh']:.6f} kWh")
            print(f"   CO2 emissions: {emissions_data['total_emissions_kg_co2']:.6f} kg CO2eq")
            print(f"   Duration: {emissions_data['duration_seconds']:.0f}s")

            # Per-config energy breakdown
            print(f"\n   Per-configuration energy:")
            for r in all_results:
                if "energy_kwh" in r:
                    print(f"   {r['config']:>20s}: {r['energy_kwh']:.6f} kWh, {r['co2_kg']:.6f} kg CO2")

            # Save energy report
            energy_path = output_dir / "energy_report.json"
            with open(energy_path, "w") as f:
                json.dump(emissions_data, f, indent=2)
            print(f"   Saved to: {energy_path}")

        except Exception as e:
            print(f"⚠️  Energy tracking error: {e}")

    # --- Save results ---
    # Save summary (without per-sample predictions, with energy data)
    summary = {
        "dataset": args.dataset,
        "generator": "ollama" if args.use_ollama else "gpt-4o-mini",
        "n_test_samples": len(test_data),
        "f1_function": "reward.py:compute_f1",
        "configs": []
    }
    for r in all_results:
        s = {k: v for k, v in r.items() if k != "predictions"}
        summary["configs"].append(s)
    if emissions_data:
        summary["energy"] = emissions_data

    summary_path = output_dir / "comparison_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Log final summary to W&B
    if args.wandb:
        import wandb
        # Create a comparison table
        columns = ["Config", "F1", "EM", "Avg K", "Retrieval Rate"]
        data = [[r["config"], r["avg_f1"], r["avg_em"], r["avg_topk"], r["retrieval_rate"]]
                for r in all_results]

        # Add energy columns if available
        if emissions_data and any("energy_kwh" in r for r in all_results):
            columns.extend(["Energy (kWh)", "CO2 (kg)"])
            for i, r in enumerate(all_results):
                data[i].extend([r.get("energy_kwh", 0), r.get("co2_kg", 0)])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({"comparison_table": table})

        # Log aggregate energy metrics
        if emissions_data:
            wandb.log({
                "energy/total_kwh": emissions_data["total_energy_kwh"],
                "energy/co2_kg": emissions_data["total_emissions_kg_co2"],
                "energy/duration_s": emissions_data["duration_seconds"],
            })
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
