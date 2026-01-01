#!/usr/bin/env python3
"""
Compare results across all baseline experiments.
Generates a summary table and identifies best performing methods.
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments/results/baselines"


def load_metric_scores(results_dir: Path) -> dict:
    """Load metric scores from all experiment directories."""
    results = {}
    
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        metric_file = exp_dir / "metric_score.txt"
        if not metric_file.exists():
            continue
        
        # Parse directory name to get method info
        # Format: custom_dataset_YYYY_MM_DD_HH_MM_method_name
        dir_name = exp_dir.name
        
        # Extract method name (last part after timestamp)
        parts = dir_name.split("_")
        if len(parts) >= 7:
            # Find where timestamp ends (after HH_MM)
            method_parts = parts[6:]  # Skip: custom_dataset_YYYY_MM_DD_HH_MM
            method_name = "_".join(method_parts)
        else:
            method_name = dir_name
        
        # Load metrics
        metrics = {}
        with open(metric_file) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = value.strip()
        
        # Extract timestamp from directory name for sorting
        try:
            timestamp_str = "_".join(parts[2:6])  # YYYY_MM_DD_HH_MM
            timestamp = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M")
        except:
            timestamp = datetime.min
        
        results[dir_name] = {
            "method": method_name,
            "metrics": metrics,
            "timestamp": timestamp,
            "path": str(exp_dir),
        }
    
    return results


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    print("=" * 100)
    print("BASELINE EXPERIMENTS COMPARISON")
    print("=" * 100)
    print()
    
    # Sort by timestamp (most recent first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["timestamp"], reverse=True)
    
    # Header
    print(f"{'Method':<40} | {'EM':>8} | {'F1':>8} | {'Recall@5':>10} | {'Date':>12}")
    print("-" * 100)
    
    best_em = 0
    best_f1 = 0
    best_recall = 0
    best_em_method = ""
    best_f1_method = ""
    best_recall_method = ""
    
    for dir_name, data in sorted_results:
        method = data["method"]
        metrics = data["metrics"]
        timestamp = data["timestamp"]
        
        em = metrics.get("em", 0)
        f1 = metrics.get("f1", 0)
        recall = metrics.get("retrieval_recall_top5", metrics.get("retrieval_recall", 0))
        
        # Track best
        if em > best_em:
            best_em = em
            best_em_method = method
        if f1 > best_f1:
            best_f1 = f1
            best_f1_method = method
        if recall > best_recall:
            best_recall = recall
            best_recall_method = method
        
        date_str = timestamp.strftime("%Y-%m-%d") if timestamp != datetime.min else "Unknown"
        
        print(f"{method:<40} | {em*100:>7.2f}% | {f1*100:>7.2f}% | {recall*100:>9.2f}% | {date_str:>12}")
    
    print("-" * 100)
    print()
    
    # Summary
    print("📊 BEST RESULTS:")
    print(f"   Best EM:       {best_em*100:.2f}% ({best_em_method})")
    print(f"   Best F1:       {best_f1*100:.2f}% ({best_f1_method})")
    print(f"   Best Recall:   {best_recall*100:.2f}% ({best_recall_method})")
    print()


def save_comparison_csv(results: dict, output_path: Path):
    """Save comparison as CSV file."""
    sorted_results = sorted(results.items(), key=lambda x: x[1]["timestamp"], reverse=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("Method,EM,F1,Retrieval_Recall,Date,Directory\n")
        
        for dir_name, data in sorted_results:
            method = data["method"]
            metrics = data["metrics"]
            timestamp = data["timestamp"]
            
            em = metrics.get("em", 0)
            f1 = metrics.get("f1", 0)
            recall = metrics.get("retrieval_recall_top5", metrics.get("retrieval_recall", 0))
            date_str = timestamp.strftime("%Y-%m-%d") if timestamp != datetime.min else "Unknown"
            
            f.write(f"{method},{em:.4f},{f1:.4f},{recall:.4f},{date_str},{dir_name}\n")
    
    print(f"📄 CSV saved to: {output_path}")


def analyze_retrieval_impact(results: dict):
    """Analyze how retrieval recall impacts overall performance."""
    print("\n📈 RETRIEVAL IMPACT ANALYSIS:")
    print("-" * 60)
    
    data_points = []
    for dir_name, data in results.items():
        metrics = data["metrics"]
        recall = metrics.get("retrieval_recall_top5", metrics.get("retrieval_recall", 0))
        f1 = metrics.get("f1", 0)
        em = metrics.get("em", 0)
        data_points.append((recall, f1, em, data["method"]))
    
    if not data_points:
        print("   No data available for analysis.")
        return
    
    # Sort by recall
    data_points.sort(key=lambda x: x[0], reverse=True)
    
    for recall, f1, em, method in data_points:
        print(f"   {method:<35} | Recall: {recall*100:5.1f}% → F1: {f1*100:5.1f}%, EM: {em*100:5.1f}%")
    
    # Calculate correlation insight
    if len(data_points) > 1:
        avg_recall = sum(x[0] for x in data_points) / len(data_points)
        print(f"\n   Average Retrieval Recall: {avg_recall*100:.1f}%")
        
        if avg_recall < 0.3:
            print("\n   ⚠️  LOW RETRIEVAL RECALL DETECTED!")
            print("   The retrieval system is the bottleneck. Consider:")
            print("   1. Using dense retrieval (E5, BGE) instead of BM25")
            print("   2. Hybrid retrieval (BM25 + dense)")
            print("   3. Query expansion/reformulation")
            print("   4. Re-indexing with better chunking strategy")


def main():
    print(f"Loading results from: {RESULTS_DIR}")
    
    if not RESULTS_DIR.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return
    
    results = load_metric_scores(RESULTS_DIR)
    
    if not results:
        print("❌ No experiment results found.")
        return
    
    print(f"Found {len(results)} experiment(s)\n")
    
    print_comparison_table(results)
    analyze_retrieval_impact(results)
    
    # Save CSV
    csv_path = RESULTS_DIR / "comparison_summary.csv"
    save_comparison_csv(results, csv_path)
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

