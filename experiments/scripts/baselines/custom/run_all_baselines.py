#!/usr/bin/env python3
"""
Run all baseline RAG methods on custom dataset.
Collects results from all methods and generates a summary.
"""

import os
import sys
import time
from datetime import datetime

# Import individual baseline runners
from run_naive_rag import run_naive_rag
from run_flare import run_flare
from run_iterretgen import run_iterretgen
# from run_self_rag import run_self_rag  # Requires specific model
# from run_skr import run_skr  # Requires judger setup
# from run_adaptive_rag import run_adaptive_rag  # Requires judger setup
# from run_ircot import run_ircot  # Complex setup

from base_config import RESULTS_DIR


def main():
    """Run all baseline methods."""
    print("=" * 70)
    print("BASELINE RAG METHODS EVALUATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    split = "test"  # Use test set for evaluation
    results_summary = {}
    
    # Define methods to run (ordered by complexity)
    methods = [
        ("Naive RAG", run_naive_rag),
        ("FLARE", run_flare),
        ("Iter-RetGen", run_iterretgen),
        # ("Self-RAG", run_self_rag),  # Skip - requires specific model
        # ("SKR", run_skr),  # Skip - requires judger setup
        # ("Adaptive-RAG", run_adaptive_rag),  # Skip - requires judger setup
        # ("IRCoT", run_ircot),  # Skip - complex setup
    ]
    
    for method_name, run_func in methods:
        print("\n" + "-" * 70 + "\n")
        start_time = time.time()
        
        try:
            results = run_func(split=split)
            elapsed = time.time() - start_time
            
            if results is not None:
                results_summary[method_name] = f"✅ Completed ({elapsed:.1f}s)"
            else:
                results_summary[method_name] = f"⚠️  Skipped"
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results_summary[method_name] = f"❌ Failed ({elapsed:.1f}s): {str(e)[:30]}"
    
    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS SUMMARY")
    print("=" * 70)
    for method, status in results_summary.items():
        print(f"  {method:20s}: {status}")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return summary for further processing
    return results_summary


if __name__ == "__main__":
    main()

