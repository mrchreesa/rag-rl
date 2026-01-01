#!/usr/bin/env python3
"""
Custom dense retrieval baseline that bypasses FlashRAG's pipeline issues.
Uses E5 embeddings + FAISS for retrieval, then OpenAI for generation.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Add FlashRAG to path for utils only
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
FLASHRAG_PATH = PROJECT_ROOT / 'src/rag/FlashRAG'
sys.path.insert(0, str(FLASHRAG_PATH))

from flashrag.retriever.utils import load_corpus, parse_query


class DenseRetriever:
    """Custom dense retriever using E5 + FAISS."""
    
    def __init__(self, index_path: str, corpus_path: str, model_path: str = "intfloat/e5-base-v2"):
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading E5 model from {model_path}...")
        self.model = SentenceTransformer(
            model_path, 
            trust_remote_code=True, 
            model_kwargs={"torch_dtype": torch.float16}
        )
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Index contains {self.index.ntotal} vectors")
        
        print(f"Loading corpus from {corpus_path}...")
        self.corpus = load_corpus(corpus_path)
        print(f"Corpus contains {len(self.corpus)} documents")
    
    def retrieve(self, queries: list, topk: int = 5) -> list:
        """Retrieve top-k documents for each query."""
        # Parse queries with E5 prefix
        parsed_queries = parse_query('e5', queries, 'query: ', is_query=True)
        
        # Encode queries
        emb = self.model.encode(
            parsed_queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = emb.astype(np.float32, order="C")
        
        # Search
        scores, ids = self.index.search(emb, topk)
        
        # Get documents
        results = []
        for query_ids, query_scores in zip(ids, scores):
            docs = []
            for idx, score in zip(query_ids, query_scores):
                doc = dict(self.corpus[idx])
                doc['retrieval_score'] = float(score)
                docs.append(doc)
            results.append(docs)
        
        return results


class OpenAIGenerator:
    """OpenAI API generator."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    
    def generate(self, question: str, docs: list) -> str:
        """Generate answer given question and retrieved docs."""
        # Build context from docs
        context = ""
        for i, doc in enumerate(docs):
            title = doc.get("title", "Unknown")
            contents = doc.get("contents", "")
            context += f"Doc {i+1}(Title: {title}) {contents}\n\n"
        
        # Build prompt
        system_prompt = f"""Answer the question based on the given document. Only give me the answer and do not output any other words.
The following are given documents.

{context}"""
        
        user_prompt = f"Question: {question}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=256,
            temperature=0.0,
        )
        
        return response.choices[0].message.content


def compute_em(pred: str, golden_answers: list) -> float:
    """Compute Exact Match score."""
    pred = pred.strip().lower()
    for ans in golden_answers:
        if pred == ans.strip().lower():
            return 1.0
    return 0.0


def compute_f1(pred: str, golden_answers: list) -> float:
    """Compute F1 score (best across all golden answers)."""
    def f1_single(pred_str, gold_str):
        pred_tokens = pred_str.lower().split()
        gold_tokens = gold_str.lower().split()
        common = set(pred_tokens) & set(gold_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return max(f1_single(pred, ans) for ans in golden_answers)


def compute_retrieval_recall(docs: list, golden_answers: list, threshold: float = 0.5) -> float:
    """
    Check if any retrieved doc contains the answer using token overlap.
    
    Uses token-level matching instead of exact substring to handle:
    - Formatting differences
    - Slight paraphrasing
    - Academic text variations
    
    Args:
        docs: Retrieved documents
        golden_answers: List of acceptable answers
        threshold: Minimum fraction of answer tokens that must appear in doc (default 0.5)
    
    Returns:
        1.0 if any doc contains sufficient answer overlap, 0.0 otherwise
    """
    import re
    
    def tokenize(text: str) -> set:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    for doc in docs:
        contents = doc.get("contents", "")
        doc_tokens = tokenize(contents)
        
        for ans in golden_answers:
            ans_tokens = tokenize(ans)
            if not ans_tokens:
                continue
            
            # Check token overlap
            overlap = ans_tokens & doc_tokens
            overlap_ratio = len(overlap) / len(ans_tokens)
            
            if overlap_ratio >= threshold:
                return 1.0
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run dense retrieval baseline")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--topk", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--split", default="test", help="Dataset split")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    print("=" * 70)
    print(f"Dense Retrieval Baseline (E5 + {args.model})")
    print("=" * 70)
    
    # Initialize components
    retriever = DenseRetriever(
        index_path=str(PROJECT_ROOT / "data/indexes/custom_e5/e5_Flat.index"),
        corpus_path=str(PROJECT_ROOT / "data/corpus/custom/combined_corpus.jsonl"),
    )
    generator = OpenAIGenerator(model=args.model)
    
    # Load dataset
    dataset_path = PROJECT_ROOT / f"data/datasets/custom_dataset/{args.split}.jsonl"
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} samples")
    
    # Run evaluation
    results = []
    em_scores = []
    f1_scores = []
    recall_scores = []
    
    print(f"\nRunning evaluation with topk={args.topk}...")
    for item in tqdm(dataset, desc="Processing"):
        question = item["question"]
        golden_answers = item["golden_answers"]
        
        # Retrieve
        docs = retriever.retrieve([question], topk=args.topk)[0]
        
        # Generate
        pred = generator.generate(question, docs)
        
        # Compute metrics
        em = compute_em(pred, golden_answers)
        f1 = compute_f1(pred, golden_answers)
        recall = compute_retrieval_recall(docs, golden_answers)
        
        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)
        
        results.append({
            "id": item["id"],
            "question": question,
            "golden_answers": golden_answers,
            "pred": pred,
            "retrieval_result": docs,
            "em": em,
            "f1": f1,
            "retrieval_recall": recall,
        })
    
    # Compute averages
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Exact Match:      {avg_em*100:.2f}%")
    print(f"F1 Score:         {avg_f1*100:.2f}%")
    print(f"Retrieval Recall: {avg_recall*100:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = PROJECT_ROOT / f"experiments/results/baselines/custom_dataset_{timestamp}_dense_e5_{args.model}_topk{args.topk}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "intermediate_data.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(save_dir / "metric_score.txt", 'w') as f:
        f.write(f"em: {avg_em}\n")
        f.write(f"f1: {avg_f1}\n")
        f.write(f"retrieval_recall_top5: {avg_recall}\n")
    
    print(f"\n✅ Results saved to: {save_dir}")


if __name__ == "__main__":
    main()

