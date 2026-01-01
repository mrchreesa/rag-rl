#!/usr/bin/env python3
"""
Run Iter-RetGen baseline on custom dataset.
Iterative Retrieval-Generation: Multiple rounds of retrieve-generate.
Uses previous generation to augment retrieval query.
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import IterativePipeline


def run_iterretgen(split="test", iter_num=3):
    """Run Iter-RetGen baseline."""
    method_name = "Iter-RetGen"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("iterretgen", split)
    config_dict.update(get_ollama_config())
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline with {iter_num} iterations...")
    
    try:
        pipeline = IterativePipeline(
            config,
            iter_num=iter_num,  # Number of retrieval-generation iterations
        )
        
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ Iter-RetGen failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Iter-RetGen baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--iter_num", type=int, default=3,
                       help="Number of iterations (default: 3)")
    args = parser.parse_args()
    
    run_iterretgen(split=args.split, iter_num=args.iter_num)

