#!/usr/bin/env python3
"""
Dense E5 retrieval baseline for HotpotQA benchmark.
Bypasses FlashRAG's pipeline (which segfaults on macOS) and uses E5 + FAISS directly.

Uses the same seed=2024 and 1000-sample config as the BM25 baseline for fair comparison.
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

# Add FlashRAG to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
FLASHRAG_PATH = PROJECT_ROOT / 'src/rag/FlashRAG'
sys.path.insert(0, str(FLASHRAG_PATH))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Load .env file if it exists
try:
    from utils.env_loader import load_env_file
    load_env_file(PROJECT_ROOT)
except ImportError:
    env_file = PROJECT_ROOT / '.env'
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
        parsed_queries = parse_query('e5', queries, 'query: ', is_query=True)

        emb = self.model.encode(
            parsed_queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = emb.astype(np.float32, order="C")

        scores, ids = self.index.search(emb, topk)

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
    """OpenAI API generator with cost tracking."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    PRICING = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    }

    def generate(self, question: str, docs: list) -> str:
        """Generate answer given question and retrieved docs."""
        context = ""
        for i, doc in enumerate(docs):
            title = doc.get("title", "Unknown")
            contents = doc.get("contents", "")
            context += f"Doc {i+1}(Title: {title}) {contents}\n\n"

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

        # Track usage
        self._total_calls += 1
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content

    @property
    def total_cost(self):
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (self._total_input_tokens * pricing["input"] +
                self._total_output_tokens * pricing["output"])

    def print_usage_summary(self):
        print(f"\nAPI Usage Summary:")
        print(f"  Total API calls: {self._total_calls}")
        print(f"  Input tokens:  {self._total_input_tokens:,}")
        print(f"  Output tokens: {self._total_output_tokens:,}")
        print(f"  Estimated cost: ${self.total_cost:.4f}")


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
    Uses token-level matching (>=50% threshold) for robustness.
    """
    import re

    def tokenize(text: str) -> set:
        return set(re.findall(r'\b\w+\b', text.lower()))

    for doc in docs:
        contents = doc.get("contents", "")
        doc_tokens = tokenize(contents)

        for ans in golden_answers:
            ans_tokens = tokenize(ans)
            if not ans_tokens:
                continue

            overlap = ans_tokens & doc_tokens
            overlap_ratio = len(overlap) / len(ans_tokens)

            if overlap_ratio >= threshold:
                return 1.0

    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run dense E5 baseline on HotpotQA")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--topk", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set!")
        print("\n   Options to set it:")
        print("   1. Export: export OPENAI_API_KEY='your-key'")
        print("   2. Create .env file in project root")
        return

    # W&B setup
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="rl-rag-enhanced",
                name=f"hotpotqa_dense_e5_{args.model}_topk{args.topk}",
                tags=["hotpotqa", "dense-e5", "baseline", f"topk-{args.topk}"],
                config={
                    "dataset": "hotpotqa",
                    "retrieval_method": "dense_e5",
                    "model": args.model,
                    "topk": args.topk,
                    "samples": args.samples,
                    "index": "wiki_hotpotqa_e5",
                    "corpus_size": 582445,
                    "seed": 2024,
                },
            )
        except Exception as e:
            print(f"Warning: W&B init failed: {e}")

    print("=" * 70)
    print(f"HotpotQA Dense Retrieval Baseline (E5 + {args.model}, topk={args.topk})")
    print("=" * 70)

    # Initialize components
    retriever = DenseRetriever(
        index_path=str(PROJECT_ROOT / "data/indexes/wiki_hotpotqa_e5/e5_Flat.index"),
        corpus_path=str(PROJECT_ROOT / "data/corpus/wiki/wiki_hotpotqa_subset.jsonl"),
    )
    generator = OpenAIGenerator(model=args.model)

    # Load HotpotQA dev dataset using FlashRAG (same config as BM25 baseline)
    config_path = str(PROJECT_ROOT / "experiments/configs/hotpotqa_baseline.yaml")
    config_dict = {
        "data_dir": str(PROJECT_ROOT / "data/benchmarks"),
        "dataset_name": "hotpotqa",
        "split": ["dev"],
        "test_sample_num": args.samples,
        "random_sample": True,
        "seed": 2024,
        # Override retrieval settings (not used by pipeline, but needed for Config)
        "index_path": str(PROJECT_ROOT / "data/indexes/wiki_hotpotqa_e5/e5_Flat.index"),
        "corpus_path": str(PROJECT_ROOT / "data/corpus/wiki/wiki_hotpotqa_subset.jsonl"),
    }

    print(f"\nLoading HotpotQA dev dataset ({args.samples} samples, seed=2024)...")
    config = Config(config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split.get("dev")

    if test_data is None:
        print(f"Failed to load dev data. Available splits: {list(all_split.keys())}")
        return

    dataset = test_data.data
    print(f"Loaded {len(dataset)} samples")

    # Run evaluation
    results = []
    em_scores = []
    f1_scores = []
    recall_scores = []

    print(f"\nRunning evaluation with topk={args.topk}...")
    for item in tqdm(dataset, desc="Processing"):
        question = item.question
        golden_answers = item.golden_answers

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
            "id": item.id,
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

    # Print comparison with known baselines
    print("\n" + "-" * 70)
    print("COMPARISON WITH EXISTING BASELINES")
    print("-" * 70)
    print(f"{'Method':<40} {'F1':>8} {'EM':>8} {'Recall':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'No Retrieval (GPT-4o-mini)':<40} {'38.47%':>8} {'25.60%':>8} {'—':>8}")
    print(f"{'BM25 + GPT-4o-mini (topk=5)':<40} {'41.57%':>8} {'29.60%':>8} {'32.80%':>8}")
    print(f"{'Dense E5 + ' + args.model + f' (topk={args.topk})':<40} {avg_f1*100:>7.2f}% {avg_em*100:>7.2f}% {avg_recall*100:>7.2f}%")
    print(f"\nF1 gap (dense vs no-retrieval): {(avg_f1 - 0.3847)*100:+.2f}%")
    print(f"F1 gap (dense vs BM25):         {(avg_f1 - 0.4157)*100:+.2f}%")
    print(f"Recall improvement over BM25:   {(avg_recall - 0.328)*100:+.2f}%")

    # API cost
    generator.print_usage_summary()

    # Log to W&B
    if wandb_run:
        import wandb
        wandb.log({
            "em": avg_em,
            "f1": avg_f1,
            "retrieval_recall": avg_recall,
            "api_cost": generator.total_cost,
            "api_calls": generator._total_calls,
            "input_tokens": generator._total_input_tokens,
            "output_tokens": generator._total_output_tokens,
        })
        wandb.finish()

    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = PROJECT_ROOT / f"experiments/results/baselines/hotpotqa_{timestamp}_dense_e5_{args.model}_topk{args.topk}"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "intermediate_data.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(save_dir / "metric_score.txt", 'w') as f:
        f.write(f"em: {avg_em}\n")
        f.write(f"f1: {avg_f1}\n")
        f.write(f"retrieval_recall_top{args.topk}: {avg_recall}\n")

    with open(save_dir / "config.json", 'w') as f:
        json.dump({
            "dataset": "hotpotqa",
            "split": "dev",
            "samples": args.samples,
            "seed": 2024,
            "retrieval_method": "dense_e5",
            "model": args.model,
            "topk": args.topk,
            "index_path": "data/indexes/wiki_hotpotqa_e5/e5_Flat.index",
            "corpus_path": "data/corpus/wiki/wiki_hotpotqa_subset.jsonl",
            "corpus_size": 582445,
            "results": {
                "em": avg_em,
                "f1": avg_f1,
                "retrieval_recall": avg_recall,
            },
            "api_cost": generator.total_cost,
        }, f, indent=2)

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
