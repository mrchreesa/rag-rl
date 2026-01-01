#!/usr/bin/env python3
"""
Recalculate metrics from intermediate_data.json files.
Useful when the metric calculation logic has been updated.
"""

import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments/results/baselines"


def tokenize(text: str) -> set:
    """Simple tokenization: lowercase, split on non-alphanumeric."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def compute_em(pred: str, golden_answers: list) -> float:
    """Compute Exact Match score."""
    if pred is None:
        return 0.0
    pred = pred.strip().lower()
    for ans in golden_answers:
        if pred == ans.strip().lower():
            return 1.0
    return 0.0


def compute_f1(pred: str, golden_answers: list) -> float:
    """Compute F1 score (best across all golden answers)."""
    if pred is None:
        return 0.0
    
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
    """
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


def recalculate_for_experiment(exp_dir: Path):
    """Recalculate metrics for a single experiment."""
    intermediate_path = exp_dir / "intermediate_data.json"
    if not intermediate_path.exists():
        return None
    
    with open(intermediate_path) as f:
        data = json.load(f)
    
    em_scores = []
    f1_scores = []
    recall_scores = []
    
    for item in data:
        golden_answers = item.get("golden_answers", [])
        
        # Handle both FlashRAG format (nested in 'output') and custom format (flat)
        if "output" in item:
            # FlashRAG format: {'output': {'pred': ..., 'retrieval_result': ...}}
            output = item["output"]
            pred = output.get("pred")
            docs = output.get("retrieval_result", [])
        else:
            # Custom format: {'pred': ..., 'retrieval_result': ...}
            pred = item.get("pred")
            docs = item.get("retrieval_result", [])
        
        em = compute_em(pred, golden_answers)
        f1 = compute_f1(pred, golden_answers)
        recall = compute_retrieval_recall(docs, golden_answers)
        
        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)
    
    if not em_scores:
        return None
    
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    return {
        "em": avg_em,
        "f1": avg_f1,
        "retrieval_recall_top5": avg_recall,
        "num_samples": len(em_scores),
    }


def main():
    print("=" * 70)
    print("RECALCULATING METRICS FROM INTERMEDIATE DATA")
    print("=" * 70)
    
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        intermediate_path = exp_dir / "intermediate_data.json"
        if not intermediate_path.exists():
            continue
        
        print(f"\nProcessing: {exp_dir.name}")
        
        metrics = recalculate_for_experiment(exp_dir)
        if metrics is None:
            print("  ⚠️  No data or error")
            continue
        
        # Read old metrics for comparison
        old_metrics = {}
        metric_path = exp_dir / "metric_score.txt"
        if metric_path.exists():
            with open(metric_path) as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        try:
                            old_metrics[k.strip()] = float(v.strip())
                        except:
                            pass
        
        # Print comparison
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  EM:     {metrics['em']*100:6.2f}% (was {old_metrics.get('em', 0)*100:6.2f}%)")
        print(f"  F1:     {metrics['f1']*100:6.2f}% (was {old_metrics.get('f1', 0)*100:6.2f}%)")
        print(f"  Recall: {metrics['retrieval_recall_top5']*100:6.2f}% (was {old_metrics.get('retrieval_recall_top5', 0)*100:6.2f}%)")
        
        # Save updated metrics
        with open(metric_path, 'w') as f:
            f.write(f"em: {metrics['em']}\n")
            f.write(f"f1: {metrics['f1']}\n")
            f.write(f"retrieval_recall_top5: {metrics['retrieval_recall_top5']}\n")
        
        print("  ✅ Updated metric_score.txt")
    
    print("\n" + "=" * 70)
    print("Done! Run compare_results.py to regenerate comparison table.")
    print("=" * 70)


if __name__ == "__main__":
    main()

