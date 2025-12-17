#!/usr/bin/env python3
"""
Convert PDF documents to FlashRAG corpus format.

FlashRAG expects JSONL format:
{"id": "0", "contents": "..."}
For documents with title: contents = "{title}\n{text}"
For documents without title: contents = "\n{text}"
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader


def extract_pdf_title(pdf_path: str) -> str:
    """
    Extract title from PDF filename or metadata.
    Returns filename without extension as title.
    """
    filename = os.path.basename(pdf_path)
    title = os.path.splitext(filename)[0]
    return title


def process_pdf_to_corpus(
    pdf_path: str,
    doc_id: int,
    extract_title: bool = True,
    use_filename_as_title: bool = True
) -> dict:
    """
    Process a single PDF file into FlashRAG corpus format.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Unique document ID
        extract_title: Whether to extract title from PDF metadata
        use_filename_as_title: Use filename as title if metadata extraction fails
    
    Returns:
        Dictionary in FlashRAG corpus format
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Combine all pages into single text
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Extract title
        title = None
        if extract_title:
            # Try to get title from metadata
            if pages and hasattr(pages[0], 'metadata'):
                title = pages[0].metadata.get('title')
            
            # Fallback to filename
            if not title and use_filename_as_title:
                title = extract_pdf_title(pdf_path)
        
        # Clean text: remove surrogate characters and other invalid Unicode
        def clean_text(text: str) -> str:
            """Remove invalid Unicode characters."""
            return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        full_text = clean_text(full_text)
        if title:
            title = clean_text(title)
        
        # Format contents according to FlashRAG requirements
        if title:
            contents = f"{title}\n{full_text}"
        else:
            # No title - use leading newline format
            contents = f"\n{full_text}"
        
        return {
            "id": str(doc_id),
            "contents": contents,
            "source_file": os.path.basename(pdf_path)
        }
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def process_pdf_directory(
    input_dir: str,
    output_file: str,
    extract_title: bool = True,
    use_filename_as_title: bool = True
) -> dict:
    """
    Process all PDFs in a directory to FlashRAG corpus format.
    
    Args:
        input_dir: Directory containing PDF files
        output_file: Output JSONL file path
        extract_title: Whether to extract titles
        use_filename_as_title: Use filename as title fallback
    
    Returns:
        Statistics dictionary
    """
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return {"total": 0, "processed": 0, "failed": 0}
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Processing PDFs from: {input_dir}")
    print(f"Output file: {output_file}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "failed": 0
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        doc_id = 0
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            result = process_pdf_to_corpus(
                str(pdf_path),
                doc_id,
                extract_title=extract_title,
                use_filename_as_title=use_filename_as_title
            )
            
            if result:
                try:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stats["processed"] += 1
                    doc_id += 1
                except UnicodeEncodeError as e:
                    print(f"\nUnicode error writing {pdf_path}: {e}")
                    stats["failed"] += 1
            else:
                stats["failed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF documents to FlashRAG corpus format"
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output_file", "-o",
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Don't extract titles (use leading newline format)"
    )
    parser.add_argument(
        "--no-filename-title",
        action="store_true",
        help="Don't use filename as title fallback"
    )
    
    args = parser.parse_args()
    
    stats = process_pdf_directory(
        args.input_dir,
        args.output_file,
        extract_title=not args.no_title,
        use_filename_as_title=not args.no_filename_title
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total PDFs: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nOutput saved to: {args.output_file}")
    print("\nNext steps:")
    print("1. Optionally chunk the corpus using FlashRAG's chunk_doc_corpus.py")
    print("2. Build index using FlashRAG's index_builder")
    print("3. Use corpus_path in FlashRAG config")


if __name__ == "__main__":
    # Default processing for train and test PDFs
    import sys
    
    if len(sys.argv) == 1:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        conversions = [
            {
                "input_dir": os.path.join(base_dir, "docs/train"),
                "output_file": os.path.join(base_dir, "corpus/train_corpus.jsonl"),
                "name": "train"
            },
            {
                "input_dir": os.path.join(base_dir, "docs/test"),
                "output_file": os.path.join(base_dir, "corpus/test_corpus.jsonl"),
                "name": "test"
            }
        ]
        
        print("=" * 60)
        print("PREPROCESSING PDF CORPUS FOR FLASHRAG")
        print("=" * 60)
        
        all_stats = {}
        for conv in conversions:
            if os.path.exists(conv["input_dir"]):
                print(f"\nProcessing {conv['name']} PDFs...")
                stats = process_pdf_directory(
                    conv["input_dir"],
                    conv["output_file"],
                    extract_title=True,
                    use_filename_as_title=True
                )
                all_stats[conv["name"]] = stats
                print(f"  ✓ Processed {stats['processed']} PDFs")
            else:
                print(f"\n⚠ Skipping {conv['name']} - directory not found: {conv['input_dir']}")
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        
        total_processed = sum(s.get('processed', 0) for s in all_stats.values())
        total_failed = sum(s.get('failed', 0) for s in all_stats.values())
        
        print(f"\nTotal PDFs processed: {total_processed}")
        print(f"Total failed: {total_failed}")
        print("\nOutput files created in: corpus/")
        print("  - train_corpus.jsonl")
        print("  - test_corpus.jsonl")
        print("\nNext steps:")
        print("1. (Optional) Chunk corpus:")
        print("   python src/rag/FlashRAG/scripts/chunk_doc_corpus.py \\")
        print("     --input_path data/corpus/train_corpus.jsonl \\")
        print("     --output_path data/corpus/train_corpus_chunked.jsonl \\")
        print("     --chunk_by sentence --chunk_size 512")
        print("\n2. Build index:")
        print("   python -m flashrag.retriever.index_builder \\")
        print("     --retrieval_method e5 \\")
        print("     --corpus_path data/corpus/train_corpus_chunked.jsonl \\")
        print("     --save_dir data/indexes/")
    else:
        main()

