#!/usr/bin/env python3
"""
Diagnostic script to analyze retrieval performance.
Identifies questions where retrieval failed and analyzes patterns.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Results path
RESULTS_PATH = PROJECT_ROOT / "experiments/results/baselines/custom_dataset_2025_12_18_11_33_naive_rag/intermediate_data.json"


def load_results(results_path: Path) -> list:
    """Load intermediate results JSON."""
    with open(results_path) as f:
        return json.load(f)


def analyze_retrieval(data: list) -> dict:
    """Analyze retrieval performance across all questions."""
    stats = {
        "total": len(data),
        "retrieval_success": 0,
        "retrieval_fail": 0,
        "em_correct": 0,
        "by_source": defaultdict(lambda: {"total": 0, "retrieval_success": 0, "em_correct": 0}),
        "by_type": defaultdict(lambda: {"total": 0, "retrieval_success": 0, "em_correct": 0}),
        "failed_questions": [],
        "successful_questions": [],
    }
    
    for item in data:
        question = item["question"]
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown")
        q_type = metadata.get("type", "unknown")
        
        output = item.get("output", {})
        metric_score = output.get("metric_score", {})
        
        retrieval_recall = metric_score.get("retrieval_recall", 0)
        em = metric_score.get("em", 0)
        f1 = metric_score.get("f1", 0)
        
        # Update source stats
        stats["by_source"][source]["total"] += 1
        stats["by_type"][q_type]["total"] += 1
        
        if retrieval_recall > 0:
            stats["retrieval_success"] += 1
            stats["by_source"][source]["retrieval_success"] += 1
            stats["by_type"][q_type]["retrieval_success"] += 1
            stats["successful_questions"].append({
                "id": item["id"],
                "question": question[:100],
                "source": source,
                "type": q_type,
                "retrieval_recall": retrieval_recall,
                "em": em,
                "f1": f1,
            })
        else:
            stats["retrieval_fail"] += 1
            stats["failed_questions"].append({
                "id": item["id"],
                "question": question[:100],
                "source": source,
                "type": q_type,
                "golden_answer": item["golden_answers"][0][:100] if item["golden_answers"] else "",
                "pred": output.get("pred", "")[:100],
                "retrieval_recall": retrieval_recall,
                "em": em,
                "f1": f1,
            })
        
        if em > 0:
            stats["em_correct"] += 1
            stats["by_source"][source]["em_correct"] += 1
            stats["by_type"][q_type]["em_correct"] += 1
    
    return stats


def print_summary(stats: dict):
    """Print summary statistics."""
    print("=" * 80)
    print("RETRIEVAL DIAGNOSTIC REPORT")
    print("=" * 80)
    
    total = stats["total"]
    success = stats["retrieval_success"]
    fail = stats["retrieval_fail"]
    em = stats["em_correct"]
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total questions: {total}")
    print(f"   Retrieval success: {success} ({100*success/total:.1f}%)")
    print(f"   Retrieval failed: {fail} ({100*fail/total:.1f}%)")
    print(f"   Exact match: {em} ({100*em/total:.1f}%)")
    
    print(f"\n📁 BY SOURCE FILE")
    print("-" * 60)
    for source, data in sorted(stats["by_source"].items(), key=lambda x: -x[1]["total"]):
        t = data["total"]
        s = data["retrieval_success"]
        e = data["em_correct"]
        print(f"   {source[:40]:<40} | Total: {t:2d} | Recall: {100*s/t:5.1f}% | EM: {100*e/t:5.1f}%")
    
    print(f"\n📝 BY QUESTION TYPE")
    print("-" * 60)
    for q_type, data in sorted(stats["by_type"].items()):
        t = data["total"]
        s = data["retrieval_success"]
        e = data["em_correct"]
        print(f"   {q_type:<15} | Total: {t:2d} | Recall: {100*s/t:5.1f}% | EM: {100*e/t:5.1f}%")
    
    print(f"\n❌ FAILED RETRIEVALS (showing first 10)")
    print("-" * 80)
    for i, item in enumerate(stats["failed_questions"][:10]):
        print(f"\n   [{item['id']}] {item['question']}...")
        print(f"   Source: {item['source']}, Type: {item['type']}")
        print(f"   Golden: {item['golden_answer']}...")
        print(f"   F1: {item['f1']:.3f}")
    
    print(f"\n✅ SUCCESSFUL RETRIEVALS (showing first 5)")
    print("-" * 80)
    for i, item in enumerate(stats["successful_questions"][:5]):
        print(f"\n   [{item['id']}] {item['question']}...")
        print(f"   Source: {item['source']}, Type: {item['type']}")
        print(f"   EM: {item['em']:.1f}, F1: {item['f1']:.3f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if fail / total > 0.5:
        print("\n⚠️  HIGH FAILURE RATE: More than 50% of retrievals failed.")
        print("   Recommendations:")
        print("   1. Increase retrieval_topk from 5 to 10 or 15")
        print("   2. Consider dense retrieval (E5) for technical queries")
        print("   3. Check if corpus contains all source PDFs")
        print("   4. Try hybrid retrieval (BM25 + dense)")
    
    # Check if certain sources are problematic
    problem_sources = [s for s, d in stats["by_source"].items() 
                      if d["total"] >= 3 and d["retrieval_success"]/d["total"] < 0.3]
    if problem_sources:
        print(f"\n⚠️  PROBLEMATIC SOURCES:")
        for src in problem_sources[:5]:
            print(f"   - {src}")


def main():
    print(f"Loading results from: {RESULTS_PATH}")
    
    if not RESULTS_PATH.exists():
        print(f"❌ Results file not found: {RESULTS_PATH}")
        return
    
    data = load_results(RESULTS_PATH)
    stats = analyze_retrieval(data)
    print_summary(stats)
    
    # Save detailed results
    output_path = RESULTS_PATH.parent / "retrieval_diagnosis.json"
    with open(output_path, 'w') as f:
        # Convert defaultdicts to regular dicts for JSON
        stats_clean = dict(stats)
        stats_clean["by_source"] = dict(stats["by_source"])
        stats_clean["by_type"] = dict(stats["by_type"])
        json.dump(stats_clean, f, indent=2)
    print(f"\n📄 Detailed diagnosis saved to: {output_path}")


if __name__ == "__main__":
    main()

