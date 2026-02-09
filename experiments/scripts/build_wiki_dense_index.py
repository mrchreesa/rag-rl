#!/usr/bin/env python3
"""
Build dense FAISS index using E5 embeddings for the HotpotQA Wikipedia subset.

Bypasses FlashRAG's index_builder (which segfaults on macOS) and uses
sentence-transformers + FAISS directly.

This indexes ~582K passages from Wikipedia articles referenced in the
HotpotQA dev set, enabling semantic retrieval for the HotpotQA benchmark.

Output structure (compatible with FlashRAG retriever):
    data/indexes/wiki_hotpotqa_e5/e5_Flat.index   - FAISS index file

Usage:
    python experiments/scripts/build_wiki_dense_index.py
    python experiments/scripts/build_wiki_dense_index.py --batch-size 32
"""

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/corpus/wiki/wiki_hotpotqa_subset.jsonl")
SAVE_DIR = os.path.join(PROJECT_ROOT, "data/indexes/wiki_hotpotqa_e5")
MODEL_NAME = "intfloat/e5-base-v2"


def load_corpus(corpus_path: str) -> list[str]:
    """Load corpus texts from JSONL file."""
    texts = []
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc['contents'])
    return texts


def encode_corpus(texts: list[str], model_name: str, batch_size: int = 64,
                  max_length: int = 512) -> np.ndarray:
    """Encode corpus texts into dense embeddings using E5."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Model: {model_name}")

    model = SentenceTransformer(model_name, device=device)

    # E5 requires "passage: " prefix for documents
    prefixed = [f"passage: {t}" for t in texts]

    print(f"  Encoding {len(prefixed):,} passages (batch_size={batch_size})...")
    start = time.time()

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    print(f"  Encoding complete: {elapsed:.0f}s ({len(texts)/elapsed:.0f} passages/sec)")
    print(f"  Embedding shape: {embeddings.shape}")

    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, save_dir: str):
    """Build and save a FAISS Flat (exact search) index."""
    import faiss

    dim = embeddings.shape[1]
    print(f"  Building FAISS Flat index (dim={dim}, n={embeddings.shape[0]:,})...")

    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since vectors are normalized)
    index.add(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    index_path = os.path.join(save_dir, "e5_Flat.index")
    faiss.write_index(index, index_path)

    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  Index saved: {index_path} ({size_mb:.1f} MB)")
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Build wiki dense index for HotpotQA")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size (lower = less memory)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per passage")
    args = parser.parse_args()

    print("=" * 70)
    print("Building Dense E5 Index for HotpotQA Wikipedia Subset")
    print("=" * 70)
    print(f"Corpus: {CORPUS_PATH}")
    print(f"Output: {SAVE_DIR}")
    print()

    if not os.path.exists(CORPUS_PATH):
        print(f"Corpus not found: {CORPUS_PATH}")
        print("Run the corpus filtering step first.")
        return

    # Load corpus
    print("Step 1: Loading corpus...")
    start = time.time()
    texts = load_corpus(CORPUS_PATH)
    print(f"  Loaded {len(texts):,} passages ({time.time()-start:.1f}s)")
    print()

    # Encode
    print("Step 2: Encoding with E5-base-v2...")
    embeddings = encode_corpus(texts, MODEL_NAME, batch_size=args.batch_size,
                               max_length=args.max_length)
    print()

    # Build index
    print("Step 3: Building FAISS index...")
    index_path = build_faiss_index(embeddings, SAVE_DIR)
    print()

    print("=" * 70)
    print("Dense index built successfully!")
    print(f"  Passages indexed: {len(texts):,}")
    print(f"  Index file: {index_path}")
    print(f"  Corpus file: {CORPUS_PATH}")
    print()
    print("To use with the RL pipeline or FlashRAG:")
    print(f"  index_path: {SAVE_DIR}/e5_Flat.index")
    print(f"  corpus_path: {CORPUS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
