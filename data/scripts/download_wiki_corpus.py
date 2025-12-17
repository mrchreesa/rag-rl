#!/usr/bin/env python3
"""
Download pre-processed Wikipedia corpus.

Options:
1. DPR Wikipedia (21M passages, 100 words each) - Most common for RAG research
2. Wikipedia from HuggingFace datasets

The DPR corpus is based on Wikipedia 2018-12-20 dump, split into ~100 word passages.
This is the same corpus used by DPR, KILT, RAG, and many other retrieval systems.
"""

import os
import json
import gzip
import csv
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_dpr_wiki(output_dir: str = None):
    """
    Download DPR Wikipedia corpus (21M passages).
    
    Source: Facebook DPR
    URL: https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
    """
    if output_dir is None:
        base_dir = Path(__file__).parent
        output_dir = base_dir / "corpus" / "wiki"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gz_file = output_dir / "psgs_w100.tsv.gz"
    tsv_file = output_dir / "psgs_w100.tsv"
    output_file = output_dir / "wiki_dpr.jsonl"
    
    url = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
    
    print("=" * 70)
    print("DOWNLOADING DPR WIKIPEDIA CORPUS")
    print("=" * 70)
    print(f"Source: Facebook DPR")
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print()
    print("This corpus contains:")
    print("  - ~21 million Wikipedia passages")
    print("  - ~100 words per passage")
    print("  - Based on Wikipedia 2018-12-20 dump")
    print("  - Used by DPR, KILT, RAG, and many RAG systems")
    print()
    print("File size: ~3.5GB compressed, ~12GB uncompressed")
    print()
    
    # Download if not exists
    if not gz_file.exists():
        print(f"Downloading {url}...")
        print("This may take 10-30 minutes depending on your connection...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, gz_file, show_progress)
        print("\n  Download complete!")
    else:
        print(f"✓ File already downloaded: {gz_file}")
    
    # Convert to FlashRAG JSONL format
    if not output_file.exists():
        print(f"\nConverting to FlashRAG format...")
        print(f"Reading from: {gz_file}")
        print(f"Writing to: {output_file}")
        
        with gzip.open(gz_file, 'rt', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader)  # Skip header: id, text, title
            
            for row in tqdm(reader, desc="Converting", total=21015324):
                if len(row) >= 3:
                    doc_id = row[0]
                    text = row[1]
                    title = row[2]
                    
                    # FlashRAG format: {"id": str, "contents": "title\ntext"}
                    item = {
                        "id": doc_id,
                        "title": title,
                        "contents": f"{title}\n{text}"
                    }
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("  Conversion complete!")
    else:
        print(f"✓ FlashRAG corpus already exists: {output_file}")
    
    # Get file sizes
    gz_size = gz_file.stat().st_size / (1024 * 1024 * 1024) if gz_file.exists() else 0
    jsonl_size = output_file.stat().st_size / (1024 * 1024 * 1024) if output_file.exists() else 0
    
    # Count lines
    line_count = 0
    if output_file.exists():
        with open(output_file, 'r') as f:
            for _ in f:
                line_count += 1
    
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"✅ Corpus ready: {output_file}")
    print(f"   Compressed file: {gz_size:.2f} GB")
    print(f"   JSONL file: {jsonl_size:.2f} GB")
    print(f"   Total passages: {line_count:,}")
    print()
    print("Next step: Build index")
    print(f"   python -m flashrag.retriever.index_builder \\")
    print(f"     --retrieval_method bm25 \\")
    print(f"     --corpus_path {output_file} \\")
    print(f"     --bm25_backend bm25s \\")
    print(f"     --save_dir data/indexes/wiki_bm25/")
    
    return str(output_file)


def download_beir_wiki(output_dir: str = None):
    """
    Alternative: Download BEIR Wikipedia corpus using datasets library.
    Smaller but still useful for testing.
    """
    from datasets import load_dataset
    
    if output_dir is None:
        base_dir = Path(__file__).parent
        output_dir = base_dir / "corpus" / "wiki"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wiki_beir.jsonl"
    
    print("=" * 70)
    print("DOWNLOADING BEIR WIKIPEDIA CORPUS")
    print("=" * 70)
    
    # Load Natural Questions corpus which contains Wikipedia passages
    dataset = load_dataset("BeIR/nq", "corpus", split="corpus")
    
    print(f"Loaded {len(dataset)} passages")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(dataset, desc="Converting")):
            corpus_item = {
                "id": str(i),
                "title": item.get("title", ""),
                "contents": f"{item.get('title', '')}\n{item.get('text', '')}"
            }
            f.write(json.dumps(corpus_item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pre-processed Wikipedia corpus")
    parser.add_argument("--source", choices=["dpr", "beir"], default="dpr",
                        help="Corpus source: dpr (21M passages) or beir (smaller)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.source == "dpr":
        download_dpr_wiki(args.output_dir)
    else:
        download_beir_wiki(args.output_dir)

