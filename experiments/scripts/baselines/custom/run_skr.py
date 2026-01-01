#!/usr/bin/env python3
"""
Run SKR baseline on custom dataset.
SKR (Self-Knowledge Retrieval): Judges whether retrieval is needed
based on the model's own knowledge.
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import ConditionalPipeline


def run_skr(split="test"):
    """Run SKR baseline."""
    method_name = "SKR"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("skr", split)
    config_dict.update(get_ollama_config())
    
    # SKR needs judger configuration
    config_dict.update({
        "judger_name": "skr",  # Use SKR judger
    })
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    try:
        pipeline = ConditionalPipeline(config)
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ SKR failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SKR baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_skr(split=args.split)

