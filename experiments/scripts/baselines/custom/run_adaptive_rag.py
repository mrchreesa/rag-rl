#!/usr/bin/env python3
"""
Run Adaptive-RAG baseline on custom dataset.
Adaptive-RAG: Adaptively chooses between different RAG strategies
based on query complexity (no-rag, single-hop, multi-hop).
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config, get_ollama_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import AdaptivePipeline


def run_adaptive_rag(split="test"):
    """Run Adaptive-RAG baseline."""
    method_name = "Adaptive-RAG"
    print_header(method_name)
    
    # Build config
    config_dict = get_base_config("adaptive_rag", split)
    config_dict.update(get_ollama_config())
    
    # Adaptive-RAG needs judger configuration
    # Use a small seq2seq model as the judger to keep compute reasonable
    config_dict.update({
        "judger_name": "adaptive",  # Use adaptive judger
        "judger_config": {
            # HuggingFace model used by AdaptiveJudger to classify query type
            # You can change this to another small seq2seq model if preferred
            "model_path": "google/flan-t5-small",
            "batch_size": 8,
            "max_length": 64,
        },
    })
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    try:
        pipeline = AdaptivePipeline(config)
        results = pipeline.run(test_data, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ Adaptive-RAG failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Adaptive-RAG baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_adaptive_rag(split=args.split)

