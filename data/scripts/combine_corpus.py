#!/usr/bin/env python3
"""
Combine train and test corpora into a single evaluation corpus.
This ensures test questions can be answered from the corpus.
"""

import os
import json
import sys
from pathlib import Path
from tqdm import tqdm


def combine_corpora(
    train_corpus_path: str,
    test_corpus_path: str,
    output_path: str,
    preserve_ids: bool = False
) -> dict:
    """
    Combine train and test corpora into a single corpus.
    
    Args:
        train_corpus_path: Path to train corpus JSONL
        test_corpus_path: Path to test corpus JSONL
        output_path: Path to output combined corpus JSONL
        preserve_ids: If False, reassign sequential IDs starting from 0
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "train_chunks": 0,
        "test_chunks": 0,
        "total_chunks": 0,
        "train_docs": set(),
        "test_docs": set()
    }
    
    print(f"Combining corpora:")
    print(f"  Train: {train_corpus_path}")
    print(f"  Test:  {test_corpus_path}")
    print(f"  Output: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        chunk_id = 0
        
        # Process train corpus
        if os.path.exists(train_corpus_path):
            print(f"\n📚 Processing train corpus...")
            with open(train_corpus_path, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="Train chunks"):
                    item = json.loads(line.strip())
                    
                    # Track unique documents
                    if 'doc_id' in item:
                        stats["train_docs"].add(item['doc_id'])
                    
                    # Reassign ID if needed
                    if not preserve_ids:
                        item['id'] = str(chunk_id)
                        chunk_id += 1
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    stats["train_chunks"] += 1
        else:
            print(f"⚠️  Train corpus not found: {train_corpus_path}")
        
        # Process test corpus
        if os.path.exists(test_corpus_path):
            print(f"\n📚 Processing test corpus...")
            with open(test_corpus_path, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="Test chunks"):
                    item = json.loads(line.strip())
                    
                    # Track unique documents
                    if 'doc_id' in item:
                        stats["test_docs"].add(item['doc_id'])
                    
                    # Reassign ID if needed
                    if not preserve_ids:
                        item['id'] = str(chunk_id)
                        chunk_id += 1
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    stats["test_chunks"] += 1
        else:
            print(f"⚠️  Test corpus not found: {test_corpus_path}")
    
    stats["total_chunks"] = stats["train_chunks"] + stats["test_chunks"]
    stats["train_docs"] = len(stats["train_docs"])
    stats["test_docs"] = len(stats["test_docs"])
    
    return stats


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine train and test corpora for RAG evaluation"
    )
    parser.add_argument(
        "--train_corpus",
        default="data/corpus/custom/corpus.jsonl",
        help="Path to train corpus (chunked)"
    )
    parser.add_argument(
        "--test_corpus",
        default="data/corpus/custom/test_corpus_chunked.jsonl",
        help="Path to test corpus (chunked)"
    )
    parser.add_argument(
        "--output",
        default="data/corpus/custom/combined_corpus.jsonl",
        help="Output path for combined corpus"
    )
    parser.add_argument(
        "--preserve-ids",
        action="store_true",
        help="Preserve original IDs (may cause duplicates)"
    )
    
    args = parser.parse_args()
    
    # Get absolute paths - script is in data/scripts/, so go up 2 levels to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Resolve paths relative to project root
    if os.path.isabs(args.train_corpus):
        train_path = args.train_corpus
    else:
        train_path = os.path.join(project_root, args.train_corpus)
    
    if os.path.isabs(args.test_corpus):
        test_path = args.test_corpus
    else:
        test_path = os.path.join(project_root, args.test_corpus)
    
    if os.path.isabs(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(project_root, args.output)
    
    print("=" * 70)
    print("COMBINING TRAIN + TEST CORPUS")
    print("=" * 70)
    
    stats = combine_corpora(
        train_path,
        test_path,
        output_path,
        preserve_ids=args.preserve_ids
    )
    
    print("\n" + "=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)
    print(f"Train chunks:  {stats['train_chunks']:,}")
    print(f"Test chunks:   {stats['test_chunks']:,}")
    print(f"Total chunks:  {stats['total_chunks']:,}")
    print(f"Train docs:    {stats['train_docs']:,}")
    print(f"Test docs:     {stats['test_docs']:,}")
    print(f"\n✅ Combined corpus saved to: {output_path}")
    print("\nNext step: Build BM25 index from combined corpus")


if __name__ == "__main__":
    main()

