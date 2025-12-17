#!/usr/bin/env python3
"""
Process already-extracted Wikipedia files into FlashRAG corpus format.

Memory-efficient streaming version that processes files in batches.
"""

import os
import json
import re
import html
from pathlib import Path
from tqdm import tqdm
import argparse
import spacy


def basic_process(title, text):
    """Clean and process Wikipedia text."""
    title = html.unescape(title)
    text = html.unescape(text)
    text = text.strip()

    if "(disambiguation)" in title.lower():
        return None, None
    if "(disambiguation page)" in title.lower():
        return None, None
    if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", title):
        return None, None
    if text.startswith("REDIRECT") or text.startswith("redirect"):
        return None, None
    if text.endswith(". References."):
        text = text[: -len(" References.")].strip()

    # Clean Wiki markup
    text = re.sub(r"\{\{cite .*?\}\}", " ", text, flags=re.DOTALL)
    text = text.replace(r"TABLETOREPLACE", " ")
    text = text.replace(r"'''", " ")
    text = text.replace(r"[[", " ")
    text = text.replace(r"]]", " ")
    text = text.replace(r"{{", " ")
    text = text.replace(r"}}", " ")
    text = text.replace("<br>", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("& amp;", "&")
    text = text.replace("nbsp;", " ")
    text = text.replace("formatnum:", "")

    text = re.sub("<math.*?</math>", "", text, flags=re.DOTALL)
    text = re.sub("<chem.*?</chem>", "", text, flags=re.DOTALL)
    text = re.sub("<score.*?</score>", "", text, flags=re.DOTALL)

    text = re.sub(r"\| ?item[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?col[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?row[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?bodystyle= ?.*? ", " ", text)
    text = re.sub(r"\| ?frame_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?data_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?label_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?headerstyle= ?.*? ", " ", text)
    text = re.sub(r"\| ?list_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?title_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?ul_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?li_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?border-style= ?.*? ", " ", text)
    text = re.sub(r'\|? ?style=".*?"', "", text)
    text = re.sub(r'\|? ?rowspan=".*?"', "", text)
    text = re.sub(r'\|? ?colspan=".*?"', "", text)
    text = re.sub(r'\|? ?scope=".*?"', "", text)
    text = re.sub(r'\|? ?align=".*?"', "", text)
    text = re.sub(r'\|? ?valign=".*?"', "", text)
    text = re.sub(r'\|? ?lang=".*?"', "", text)
    text = re.sub(r'\|? ?bgcolor=".*?"', "", text)
    text = re.sub(r"\|? ?bg=\#[a-z]+", "", text)
    text = re.sub(r'\|? ?width=".*?"', "", text)
    text = re.sub(r"\|? ?height=[0-9]+", "", text)
    text = re.sub(r"\|? ?width=[0-9]+", "", text)
    text = re.sub(r"\|? ?rowspan=[0-9]+", "", text)
    text = re.sub(r"\|? ?colspan=[0-9]+", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub("<.*?/>", "", text)
    text = re.sub(r"\|? ?align=[a-z]+", "", text)
    text = re.sub(r"\|? ?valign=[a-z]+", "", text)
    text = re.sub(r"\|? ?scope=[a-z]+", "", text)
    text = re.sub("&lt;ref&gt;.*?&lt;/ref&gt;", " ", text)
    text = re.sub("&lt;.*?&gt;", " ", text)
    text = re.sub(r"File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?", "", text)
    text = re.sub(r"Source: \[.*?\]", "", text)
    text = text.replace("Country flag|", "country:")
    text = text.replace("flag|", "country:")
    text = text.replace("flagicon|", "country:")
    text = text.replace("flagcountry|", "country:")
    text = text.replace("Flagu|", "country:")
    text = text.replace("display=inline", "")
    text = text.replace("display=it", "")
    text = text.replace("abbr=on", "")
    text = text.replace("disp=table", "")

    title = title.replace("\n", " ").replace("\t", " ")

    return title, text


def chunk_by_sentence(nlp, title, text, seg_size=6, stride=1):
    """Chunk text by sentences."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    if not sentences:
        return []
    
    chunks = []
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i : i + seg_size])
        segment = segment.replace("\n", " ").replace("\t", " ")
        chunks.append({
            "title": title,
            "text": segment
        })
        if i + seg_size >= len(sentences):
            break
    
    return chunks


def chunk_by_100w(nlp, title, text):
    """Chunk text by ~100 words."""
    doc = nlp(text)
    chunks = []
    segment_tokens = []
    word_count = 0
    
    for token in doc:
        segment_tokens.append(token.text_with_ws)
        if not token.is_space and not token.is_punct:
            word_count += 1
            if word_count == 100:
                word_count = 0
                segment = "".join(segment_tokens)
                segment = segment.replace("\n", " ").replace("\t", " ")
                chunks.append({"title": title, "text": segment})
                segment_tokens = []
    
    if segment_tokens:
        segment = "".join(segment_tokens)
        segment = segment.replace("\n", " ").replace("\t", " ")
        chunks.append({"title": title, "text": segment})
    
    return chunks


def stream_extracted_files(temp_dir):
    """Stream articles from extracted files one at a time."""
    wiki_files = sorted(Path(temp_dir).glob("**/wiki_*"))
    
    for file_path in wiki_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process extracted Wikipedia files into FlashRAG corpus")
    parser.add_argument("--temp_dir", type=str, default="data/benchmarks/wiki_corpus/temp",
                        help="Directory containing extracted Wikipedia files")
    parser.add_argument("--save_path", type=str, default="data/corpus/wiki/wiki-18.jsonl",
                        help="Output corpus file path")
    parser.add_argument("--chunk_by", default="sentence", choices=["sentence", "100w"], type=str)
    parser.add_argument("--seg_size", default=6, type=int, help="Number of sentences per chunk")
    parser.add_argument("--stride", default=1, type=int, help="Stride for chunking")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size for spacy processing")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROCESSING EXTRACTED WIKIPEDIA FILES (STREAMING)")
    print("=" * 70)
    print(f"Input directory: {args.temp_dir}")
    print(f"Output file: {args.save_path}")
    print(f"Chunk method: {args.chunk_by}")
    print()
    
    # Count files first
    wiki_files = list(Path(args.temp_dir).glob("**/wiki_*"))
    print(f"Found {len(wiki_files)} extracted files")
    
    # Load spacy model
    print("Loading spacy model...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 2000000  # Allow longer texts
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Track seen titles for deduplication
    seen_titles = set()
    chunk_id = 0
    total_articles = 0
    total_chunks = 0
    skipped_duplicates = 0
    skipped_filtered = 0
    
    print("Processing articles (streaming mode)...")
    
    with open(args.save_path, "w", encoding="utf-8") as out_file:
        batch_titles = []
        batch_texts = []
        
        for article in tqdm(stream_extracted_files(args.temp_dir), desc="Processing"):
            total_articles += 1
            
            title = article.get("title", "")
            text = article.get("text", "")
            
            # Skip empty or duplicate titles
            if not text or not title:
                skipped_filtered += 1
                continue
            
            if title in seen_titles:
                skipped_duplicates += 1
                continue
            
            seen_titles.add(title)
            
            # Process and clean
            clean_title, clean_text = basic_process(title, text)
            if clean_title is None:
                skipped_filtered += 1
                continue
            
            clean_title = f'"{clean_title}"'
            
            # Add to batch
            batch_titles.append(clean_title)
            batch_texts.append(clean_text)
            
            # Process batch when full
            if len(batch_texts) >= args.batch_size:
                # Process batch with spacy
                for i, doc in enumerate(nlp.pipe(batch_texts, batch_size=100)):
                    if args.chunk_by == "sentence":
                        chunks = []
                        sentences = [sent.text.strip() for sent in doc.sents]
                        for j in range(0, len(sentences), args.stride):
                            segment = " ".join(sentences[j : j + args.seg_size])
                            segment = segment.replace("\n", " ").replace("\t", " ")
                            chunks.append(segment)
                            if j + args.seg_size >= len(sentences):
                                break
                    else:  # 100w
                        chunks = []
                        segment_tokens = []
                        word_count = 0
                        for token in doc:
                            segment_tokens.append(token.text_with_ws)
                            if not token.is_space and not token.is_punct:
                                word_count += 1
                                if word_count == 100:
                                    word_count = 0
                                    chunks.append("".join(segment_tokens).replace("\n", " ").replace("\t", " "))
                                    segment_tokens = []
                        if segment_tokens:
                            chunks.append("".join(segment_tokens).replace("\n", " ").replace("\t", " "))
                    
                    # Write chunks
                    for chunk_text in chunks:
                        if chunk_text.strip():
                            corpus_item = {
                                "id": chunk_id,
                                "title": batch_titles[i],
                                "contents": f"{batch_titles[i]}\n{chunk_text}"
                            }
                            out_file.write(json.dumps(corpus_item, ensure_ascii=False) + "\n")
                            chunk_id += 1
                            total_chunks += 1
                
                # Clear batch
                batch_titles = []
                batch_texts = []
                
                # Progress update
                if total_articles % 100000 == 0:
                    print(f"  Processed {total_articles:,} articles -> {total_chunks:,} chunks")
        
        # Process remaining batch
        if batch_texts:
            for i, doc in enumerate(nlp.pipe(batch_texts, batch_size=100)):
                if args.chunk_by == "sentence":
                    chunks = []
                    sentences = [sent.text.strip() for sent in doc.sents]
                    for j in range(0, len(sentences), args.stride):
                        segment = " ".join(sentences[j : j + args.seg_size])
                        segment = segment.replace("\n", " ").replace("\t", " ")
                        chunks.append(segment)
                        if j + args.seg_size >= len(sentences):
                            break
                else:
                    chunks = []
                    segment_tokens = []
                    word_count = 0
                    for token in doc:
                        segment_tokens.append(token.text_with_ws)
                        if not token.is_space and not token.is_punct:
                            word_count += 1
                            if word_count == 100:
                                word_count = 0
                                chunks.append("".join(segment_tokens).replace("\n", " ").replace("\t", " "))
                                segment_tokens = []
                    if segment_tokens:
                        chunks.append("".join(segment_tokens).replace("\n", " ").replace("\t", " "))
                
                for chunk_text in chunks:
                    if chunk_text.strip():
                        corpus_item = {
                            "id": chunk_id,
                            "title": batch_titles[i],
                            "contents": f"{batch_titles[i]}\n{chunk_text}"
                        }
                        out_file.write(json.dumps(corpus_item, ensure_ascii=False) + "\n")
                        chunk_id += 1
                        total_chunks += 1
    
    # Get file size
    file_size = os.path.getsize(args.save_path) / (1024 * 1024 * 1024)  # GB
    
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✅ Corpus saved to: {args.save_path}")
    print(f"   File size: {file_size:.2f} GB")
    print(f"   Total articles processed: {total_articles:,}")
    print(f"   Unique articles: {len(seen_titles):,}")
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   Skipped (duplicates): {skipped_duplicates:,}")
    print(f"   Skipped (filtered): {skipped_filtered:,}")
    print()
    print("Next step: Build index")
    print(f"   python -m flashrag.retriever.index_builder \\")
    print(f"     --retrieval_method bm25 \\")
    print(f"     --corpus_path {args.save_path} \\")
    print(f"     --bm25_backend bm25s \\")
    print(f"     --save_dir data/indexes/wiki_bm25/")


if __name__ == "__main__":
    main()
