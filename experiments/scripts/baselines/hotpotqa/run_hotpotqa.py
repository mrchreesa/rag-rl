#!/usr/bin/env python3
"""
Run baseline experiments on HotpotQA benchmark.
Uses Wikipedia BM25 index for retrieval.
"""

import os
import sys
import argparse

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add FlashRAG to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../..'))
FLASHRAG_PATH = os.path.join(PROJECT_ROOT, 'src/rag/FlashRAG')
sys.path.insert(0, FLASHRAG_PATH)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

# Load .env file if it exists
try:
    from utils.env_loader import load_env_file
    from pathlib import Path
    load_env_file(Path(PROJECT_ROOT))
except ImportError:
    # Fallback: manually load .env
    from pathlib import Path
    env_file = Path(PROJECT_ROOT) / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and not os.environ.get(key):
                        os.environ[key] = value

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

# Paths
HOTPOTQA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "experiments", "configs", "hotpotqa_baseline.yaml")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "baselines")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_hotpotqa_config(method_name: str, split: str = "dev", sample_num: int = 1000) -> dict:
    """
    Get base config dictionary for HotpotQA experiments.
    
    Args:
        method_name: Name of the method (e.g., "naive_rag")
        split: Dataset split to use ("dev" or "train")
        sample_num: Number of samples to use (None for all)
    
    Returns:
        Config dictionary
    """
    return {
        "save_note": method_name,
        "save_dir": RESULTS_DIR,
        "data_dir": os.path.join(PROJECT_ROOT, "data", "benchmarks"),
        "dataset_name": "hotpotqa",
        "split": [split],
        "test_sample_num": sample_num,
        "random_sample": True,
        "seed": 2024,
        # Wikipedia BM25 index paths (absolute)
        "index_path": os.path.join(PROJECT_ROOT, "data/indexes/wiki_bm25/bm25"),
        "corpus_path": os.path.join(PROJECT_ROOT, "data/corpus/wiki/wiki_dpr.jsonl"),
    }


def get_ollama_config() -> dict:
    """Get OpenAI-compatible config for Ollama."""
    return {
        "framework": "openai",
        "generator_model": "llama3.1:8b-instruct-q4_K_M",
        "generator_model_path": None,
        "openai_setting": {
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1"
        },
        "generation_params": {
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def get_openai_config(model: str = "gpt-4o-mini") -> dict:
    """Get OpenAI API config."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Options to set it:\n"
            "  1. Export for this session: export OPENAI_API_KEY='your-key'\n"
            "  2. Create .env file in project root: echo 'OPENAI_API_KEY=your-key' > .env\n"
            "  3. Add to shell config: echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.zshrc"
        )
    
    return {
        "framework": "openai",
        "generator_model": model,
        "generator_model_path": None,
        "openai_setting": {
            "api_key": api_key,
            "base_url": "https://api.openai.com/v1"
        },
        "generation_params": {
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def run_hotpotqa(generator: str = "ollama", model: str = "gpt-4o-mini", 
                 topk: int = 5, sample_num: int = 1000):
    """
    Run Naive RAG on HotpotQA with specified generator.
    
    Args:
        generator: "ollama" or "openai"
        model: OpenAI model name (only used if generator="openai")
        topk: Number of documents to retrieve
        sample_num: Number of samples to evaluate
    """
    # Build method name for saving
    if generator == "ollama":
        method_name = f"hotpotqa_naive_rag_ollama_topk{topk}"
        gen_config = get_ollama_config()
        print("=" * 70)
        print(f"Running HotpotQA Naive RAG (Ollama llama3.1:8b, topk={topk})")
        print("=" * 70)
    else:
        method_name = f"hotpotqa_naive_rag_openai_{model}_topk{topk}"
        gen_config = get_openai_config(model)
        print("=" * 70)
        print(f"Running HotpotQA Naive RAG (OpenAI {model}, topk={topk})")
        print("=" * 70)
    
    # Build config
    config_dict = get_hotpotqa_config(method_name, split="dev", sample_num=sample_num)
    config_dict.update(gen_config)
    config_dict["retrieval_topk"] = topk
    
    print(f"Loading HotpotQA dev dataset ({sample_num} samples)...")
    config = Config(HOTPOTQA_CONFIG_PATH, config_dict)
    all_split = get_dataset(config)
    test_data = all_split.get("dev")
    
    if test_data is None:
        print(f"❌ Failed to load dev data. Available splits: {list(all_split.keys())}")
        return None
    
    print(f"Loaded {len(test_data.data)} samples")
    print(f"Index: {config_dict['index_path']}")
    print(f"Corpus: {config_dict['corpus_path']}")
    print(f"Running pipeline...")
    
    pipeline = SequentialPipeline(config)
    results = pipeline.run(test_data, do_eval=True)
    
    print(f"✅ HotpotQA experiment completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run HotpotQA baseline experiments")
    parser.add_argument("--generator", default="ollama", choices=["ollama", "openai"],
                       help="Generator to use")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model (only used with --generator openai)")
    parser.add_argument("--topk", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to evaluate")
    args = parser.parse_args()
    
    run_hotpotqa(
        generator=args.generator,
        model=args.model,
        topk=args.topk,
        sample_num=args.samples
    )


if __name__ == "__main__":
    main()

