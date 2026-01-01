#!/usr/bin/env python3
"""
Run baseline RAG methods with OpenAI API (gpt-4o-mini).
Supports Naive RAG and FLARE methods.
"""

import argparse
import os
import sys

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_openai_config, get_retrieval_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset


def run_naive_rag_openai(split="test", model="gpt-4o-mini", topk=5):
    """Run Naive RAG with OpenAI API."""
    from flashrag.pipeline import SequentialPipeline
    
    method_name = f"Naive RAG (OpenAI {model}, topk={topk})"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config(f"naive_rag_openai_{model}_topk{topk}", split)
    config_dict.update(get_openai_config(model))
    config_dict.update(get_retrieval_config(topk))
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split.get(split)
    
    if test_data is None:
        print(f"❌ Failed to load {split} data.")
        return None
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Generator: {model}")
    print(f"Retrieval topk: {topk}")
    print(f"Running pipeline...")
    
    pipeline = SequentialPipeline(config)
    results = pipeline.run(test_data, do_eval=True)
    
    print_completion(method_name, RESULTS_DIR)
    return results


def run_flare_openai(split="test", model="gpt-4o-mini", topk=5):
    """Run FLARE with OpenAI API."""
    from flashrag.pipeline import FLAREPipeline
    
    method_name = f"FLARE (OpenAI {model}, topk={topk})"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config(f"flare_openai_{model}_topk{topk}", split)
    config_dict.update(get_openai_config(model))
    config_dict.update(get_retrieval_config(topk))
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split.get(split)
    
    if test_data is None:
        print(f"❌ Failed to load {split} data.")
        return None
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Generator: {model}")
    print(f"Retrieval topk: {topk}")
    print(f"Running pipeline...")
    
    try:
        pipeline = FLAREPipeline(
            config,
            threshold=0.3,
            look_ahead_steps=1,
            max_generation_length=20,
            max_iter_num=5,
        )
        
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ FLARE failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run baselines with OpenAI API")
    parser.add_argument("--method", default="naive_rag", choices=["naive_rag", "flare", "both"],
                       help="Method to run")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--model", default="gpt-4o-mini", 
                       choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                       help="OpenAI model to use")
    parser.add_argument("--topk", type=int, default=5,
                       help="Number of documents to retrieve")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    if args.method == "naive_rag" or args.method == "both":
        run_naive_rag_openai(args.split, args.model, args.topk)
    
    if args.method == "flare" or args.method == "both":
        run_flare_openai(args.split, args.model, args.topk)


if __name__ == "__main__":
    main()

