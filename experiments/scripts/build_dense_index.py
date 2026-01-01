#!/usr/bin/env python3
"""
Build dense FAISS index using E5 embeddings for the custom corpus.
This enables semantic/dense retrieval instead of keyword-based BM25.
"""

import os
import sys
import subprocess

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
FLASHRAG_PATH = os.path.join(PROJECT_ROOT, 'src/rag/FlashRAG')

# Add FlashRAG to path
sys.path.insert(0, FLASHRAG_PATH)

# Corpus and index paths
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/corpus/custom/combined_corpus.jsonl")
SAVE_DIR = os.path.join(PROJECT_ROOT, "data/indexes")
INDEX_NAME = "custom_e5"


def build_e5_index():
    """Build E5 dense index using FlashRAG's index builder."""
    
    print("=" * 70)
    print("Building Dense Index with E5 Embeddings")
    print("=" * 70)
    print(f"Corpus: {CORPUS_PATH}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Index name: {INDEX_NAME}")
    print()
    
    # Check corpus exists
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Corpus not found: {CORPUS_PATH}")
        return False
    
    # Count corpus size
    with open(CORPUS_PATH) as f:
        corpus_size = sum(1 for _ in f)
    print(f"Corpus size: {corpus_size:,} documents")
    print()
    
    # Create index directory
    index_dir = os.path.join(SAVE_DIR, INDEX_NAME)
    os.makedirs(index_dir, exist_ok=True)
    
    # Build command
    # Using sentence_transformers for simpler setup (handles pooling automatically)
    cmd = [
        sys.executable, "-m", "flashrag.retriever.index_builder",
        "--retrieval_method", "e5",
        "--model_path", "intfloat/e5-base-v2",  # Will be downloaded from HuggingFace
        "--corpus_path", CORPUS_PATH,
        "--save_dir", index_dir,  # Save directly to index subdirectory
        "--use_fp16",
        "--max_length", "512",
        "--batch_size", "64",  # Smaller batch for stability
        "--pooling_method", "mean",
        "--sentence_transformer",  # Use sentence-transformers for simpler setup
        "--faiss_type", "Flat",  # Exact search (most accurate)
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("This may take 10-30 minutes depending on your hardware...")
    print("=" * 70)
    
    # Run index builder
    env = os.environ.copy()
    env["PYTHONPATH"] = FLASHRAG_PATH + ":" + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, env=env, cwd=FLASHRAG_PATH)
    
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("✅ Dense index built successfully!")
        print(f"Index saved to: {os.path.join(SAVE_DIR, INDEX_NAME)}")
        print("=" * 70)
        return True
    else:
        print()
        print("❌ Index building failed!")
        return False


if __name__ == "__main__":
    build_e5_index()

