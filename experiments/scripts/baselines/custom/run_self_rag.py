#!/usr/bin/env python3
"""
Run Self-RAG baseline on custom dataset.
Self-RAG: Self-reflective retrieval-augmented generation
Uses special tokens to control retrieval and critique generations.

Note: Self-RAG requires a specific model trained with special tokens.
This script will skip if the model is not available.
"""

from base_config import (
    CONFIG_PATH, RESULTS_DIR,
    get_base_config,
    print_header, print_completion
)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SelfRAGPipeline


def run_self_rag(split="test"):
    """Run Self-RAG baseline."""
    method_name = "Self-RAG"
    print_header(method_name)
    
    # Build config - Self-RAG requires specific model
    config_dict = get_base_config("self_rag", split)
    config_dict.update({
        "generator_model": "selfrag-llama2-7B",
        "generator_model_path": None,
        "framework": "vllm",
        "generation_params": {
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "skip_special_tokens": False,
        },
    })
    
    config = Config(CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[split]
    
    print(f"Loaded {len(test_data.data)} {split} samples")
    print(f"Running pipeline...")
    
    try:
        pipeline = SelfRAGPipeline(
            config,
            threshold=0.2,
            max_depth=2,
            beam_width=2,
            w_rel=1.0,
            w_sup=1.0,
            w_use=1.0,
            use_grounding=True,
            use_utility=True,
            use_seqscore=True,
            ignore_cont=True,
            mode="adaptive_retrieval",
        )
        
        results = pipeline.run(test_data, long_form=False, do_eval=True)
        print_completion(method_name, RESULTS_DIR)
        return results
        
    except Exception as e:
        print(f"❌ Self-RAG failed: {e}")
        print("Note: Self-RAG requires a specific model (selfrag-llama2-7B).")
        print("This model may not be available. Skipping.")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Self-RAG baseline")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    run_self_rag(split=args.split)

