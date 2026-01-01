#!/usr/bin/env python3
"""
Run Naive RAG baseline on custom dataset.
Standard RAG: retrieve -> generate
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


def run_naive_rag(split="test"):
    """Run Naive RAG baseline."""
    method_name = "Naive RAG"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("naive_rag", split)
    config_dict.update(get_ollama_config())
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split.get(split)
    
    if test_data is None:
        print(f"❌ Failed to load {split} data. Available splits: {list(all_split.keys())}")
        return None
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    pipeline = SequentialPipeline(config)
    results = pipeline.run(test_data, do_eval=True)
    
    print_completion(method_name, RESULTS_DIR)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Naive RAG baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_naive_rag(split=args.split)

