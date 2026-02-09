#!/usr/bin/env python3
"""
Base configuration and utilities for baseline experiments.
Shared across all baseline scripts.
"""

import os
import sys
from pathlib import Path

# Fix OpenMP conflict (multiple OpenMP runtimes issue)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add FlashRAG to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))
FLASHRAG_PATH = os.path.join(PROJECT_ROOT, 'src/rag/FlashRAG')
sys.path.insert(0, FLASHRAG_PATH)

# Add experiments utils to path for env loader
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

# Load .env file if it exists
try:
    from utils.env_loader import load_env_file
    load_env_file(Path(PROJECT_ROOT))
except ImportError:
    # Fallback: manually load .env if utils not available
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

# Common paths
CONFIG_PATH = os.path.join(PROJECT_ROOT, "experiments", "configs", "custom_baseline.yaml")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "custom_dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "baselines")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_base_config(method_name: str, split: str = "test") -> dict:
    """
    Get base config dictionary for a baseline method.
    
    Args:
        method_name: Name of the method (e.g., "naive_rag", "flare")
        split: Dataset split to use ("test" or "train")
    
    Returns:
        Config dictionary with common settings
    """
    # Use absolute paths to override config file relative paths
    # Note: Config class generates dataset_path from data_dir + dataset_name
    # So we override data_dir with absolute path
    return {
        "save_note": method_name,
        "save_dir": RESULTS_DIR,
        "data_dir": os.path.join(PROJECT_ROOT, "data", "datasets"),  # Absolute path
        "dataset_name": "custom_dataset",
        "split": [split],
        "test_sample_num": None,
        "random_sample": False,
        # Override relative paths with absolute paths
        "index_path": os.path.join(PROJECT_ROOT, "data/indexes/custom_combined_bm25"),
        "corpus_path": os.path.join(PROJECT_ROOT, "data/corpus/custom/combined_corpus.jsonl"),
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
    """
    Get OpenAI API config.
    
    Args:
        model: OpenAI model to use (default: gpt-4o-mini)
               Options: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
    
    Returns:
        Config dictionary for OpenAI API
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key'"
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


def get_retrieval_config(topk: int = 5) -> dict:
    """
    Get retrieval config with adjustable topk.
    
    Args:
        topk: Number of documents to retrieve (default: 5)
    
    Returns:
        Config dictionary for retrieval settings
    """
    return {
        "retrieval_topk": topk,
    }


def print_header(method_name: str):
    """Print experiment header."""
    print("=" * 70)
    print(f"Running {method_name}")
    print("=" * 70)


def print_completion(method_name: str, results_dir: str):
    """Print experiment completion message."""
    print(f"✅ {method_name} completed!")
    print(f"Results saved to: {results_dir}")

