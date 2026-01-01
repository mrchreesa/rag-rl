#!/usr/bin/env python3
"""
Run IRCoT baseline on custom dataset.
IRCoT (Interleaving Retrieval with Chain-of-Thought):
Combines retrieval with step-by-step reasoning.
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import IRCOTPipeline


def run_ircot(split="test"):
    """Run IRCoT baseline."""
    method_name = "IRCoT"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("ircot", split)
    config_dict.update(get_ollama_config())
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    try:
        pipeline = IRCOTPipeline(
            config,
            max_iter=5,  # Maximum iterations
        )
        
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ IRCoT failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IRCoT baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_ircot(split=args.split)

