#!/usr/bin/env python3
"""
Chunk corpus documents for FlashRAG without requiring chonkie.
Uses sentence-based chunking with spacy or simple text splitting.
"""

import json
import argparse
from tqdm import tqdm
import re


def chunk_by_sentences_simple(text: str, chunk_size: int = 512) -> list:
    """
    Simple sentence-based chunking without external dependencies.
    Splits by sentences and groups them into chunks of approximately chunk_size words.
    """
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        word_count = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size, save current chunk
        if current_word_count + word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_by_sentences_spacy(text: str, nlp, chunk_size: int = 512) -> list:
    """
    Sentence-based chunking using spacy for better sentence detection.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        if not sentence:
            continue
        
        word_count = len(sentence.split())
        
        if current_word_count + word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_corpus(
    input_path: str,
    output_path: str,
    chunk_by: str = "sentence",
    chunk_size: int = 512,
    use_spacy: bool = False
):
    """
    Chunk corpus documents.
    
    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        chunk_by: Chunking method ("sentence" or "simple")
        chunk_size: Target chunk size in words
        use_spacy: Whether to use spacy for better sentence detection
    """
    # Load spacy if requested
    nlp = None
    if use_spacy:
        try:
            import spacy
            print("Loading spacy model (en_core_web_lg)...")
            nlp = spacy.load("en_core_web_lg")
            print("Spacy model loaded!")
        except Exception as e:
            print(f"Warning: Could not load spacy: {e}")
            print("Falling back to simple sentence chunking...")
            use_spacy = False
    
    # Load documents
    print(f"Loading documents from {input_path}...")
    documents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    print(f"Loaded {len(documents)} documents")
    
    # Process and chunk documents
    print(f"Chunking documents (chunk_size={chunk_size} words, method={chunk_by})...")
    chunked_documents = []
    current_chunk_id = 0
    
    for doc in tqdm(documents, desc="Chunking"):
        contents = doc.get("contents", "")
        
        # Split title and text
        if "\n" in contents:
            title, text = contents.split("\n", 1)
        else:
            title = ""
            text = contents
        
        # Chunk the text
        if chunk_by == "sentence":
            if use_spacy and nlp:
                chunks = chunk_by_sentences_spacy(text, nlp, chunk_size)
            else:
                chunks = chunk_by_sentences_simple(text, chunk_size)
        else:
            raise ValueError(f"Unsupported chunk_by method: {chunk_by}")
        
        # Create chunked documents
        for chunk_text in chunks:
            chunked_doc = {
                "id": str(current_chunk_id),
                "doc_id": doc.get("id", ""),
                "title": title,
                "contents": f"{title}\n{chunk_text}" if title else f"\n{chunk_text}",
                "source_file": doc.get("source_file", "")
            }
            chunked_documents.append(chunked_doc)
            current_chunk_id += 1
    
    # Save chunked documents
    print(f"Saving {len(chunked_documents)} chunks to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in chunked_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"\nDone! Processed {len(documents)} documents into {len(chunked_documents)} chunks.")
    print(f"Average chunks per document: {len(chunked_documents) / len(documents):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Chunk corpus documents for FlashRAG")
    parser.add_argument("--input_path", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output_path", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--chunk_by", default="sentence", choices=["sentence"], help="Chunking method")
    parser.add_argument("--chunk_size", type=int, default=512, help="Target chunk size in words")
    parser.add_argument("--use_spacy", action="store_true", help="Use spacy for better sentence detection")
    
    args = parser.parse_args()
    
    chunk_corpus(
        args.input_path,
        args.output_path,
        chunk_by=args.chunk_by,
        chunk_size=args.chunk_size,
        use_spacy=args.use_spacy
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Default: chunk both train and test
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("=" * 60)
        print("CHUNKING CORPUS FOR FLASHRAG")
        print("=" * 60)
        print()
        
        chunkings = [
            {
                "input": os.path.join(base_dir, "corpus/train_corpus.jsonl"),
                "output": os.path.join(base_dir, "corpus/train_corpus_chunked.jsonl"),
                "name": "train"
            },
            {
                "input": os.path.join(base_dir, "corpus/test_corpus.jsonl"),
                "output": os.path.join(base_dir, "corpus/test_corpus_chunked.jsonl"),
                "name": "test"
            }
        ]
        
        for chunk_info in chunkings:
            if os.path.exists(chunk_info["input"]):
                print(f"Chunking {chunk_info['name']} corpus...")
                chunk_corpus(
                    chunk_info["input"],
                    chunk_info["output"],
                    chunk_by="sentence",
                    chunk_size=512,
                    use_spacy=False  # Use simple chunking (faster, no dependencies)
                )
                print()
            else:
                print(f"⚠ Skipping {chunk_info['name']} - file not found: {chunk_info['input']}")
        
        print("=" * 60)
        print("CHUNKING COMPLETE!")
        print("=" * 60)
    else:
        main()

