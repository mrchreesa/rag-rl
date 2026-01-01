#!/usr/bin/env python3
"""
Run FLARE baseline on custom dataset.
FLARE: Forward-Looking Active REtrieval augmented generation
Iteratively retrieves based on model's uncertainty.
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import FLAREPipeline


def run_flare(split="test"):
    """Run FLARE baseline."""
    method_name = "FLARE"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("flare", split)
    config_dict.update(get_ollama_config())
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    try:
        pipeline = FLAREPipeline(
            config,
            threshold=0.3,           # Confidence threshold for retrieval
            look_ahead_steps=1,      # Steps to look ahead
            max_generation_length=20, # Max tokens per generation step
            max_iter_num=5,          # Max iterations
        )
        
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ FLARE failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FLARE baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_flare(split=args.split)

