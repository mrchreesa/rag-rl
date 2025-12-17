#!/usr/bin/env python3
"""
Merge Final Dataset

This script merges auto-approved QA pairs with manually-approved pairs
to create the final high-quality dataset.

Usage:
    python merge_final.py

Input:
    - datasets/train_dataset_filtered.jsonl (auto-approved, score >= 8.5)
    - datasets/train_dataset_manually_approved.jsonl (manually approved borderline pairs)

Output:
    - datasets/train_dataset_final.jsonl (merged final dataset)
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime

# Input files
AUTO_APPROVED_FILE = "./datasets/train_dataset_filtered.jsonl"
MANUALLY_APPROVED_FILE = "./datasets/train_dataset_manually_approved.jsonl"

# Output file
FINAL_OUTPUT_FILE = "./datasets/train_dataset_final.jsonl"
STATS_FILE = "./datasets/final_dataset_stats.json"


def load_pairs(filepath: str) -> List[Dict[str, Any]]:
    """Load QA pairs from JSONL file."""
    pairs = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs


def save_pairs(filepath: str, pairs: List[Dict[str, Any]]):
    """Save QA pairs to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def merge_datasets():
    """Merge auto-approved and manually-approved pairs."""
    
    print("=" * 80)
    print("MERGING FINAL DATASET")
    print("=" * 80)
    
    # Load auto-approved pairs
    print(f"\n📗 Loading auto-approved pairs from {AUTO_APPROVED_FILE}...")
    auto_approved = load_pairs(AUTO_APPROVED_FILE)
    print(f"   Loaded: {len(auto_approved)} pairs")
    
    # Load manually-approved pairs
    print(f"\n📙 Loading manually-approved pairs from {MANUALLY_APPROVED_FILE}...")
    manually_approved = load_pairs(MANUALLY_APPROVED_FILE)
    print(f"   Loaded: {len(manually_approved)} pairs")
    
    # Mark source of each pair
    for pair in auto_approved:
        pair['approval_source'] = 'auto'
    for pair in manually_approved:
        pair['approval_source'] = 'manual'
    
    # Merge
    final_pairs = auto_approved + manually_approved
    
    # Sort by overall score (descending)
    final_pairs.sort(key=lambda x: x['quality_assessment'].get('overall_score', 0), reverse=True)
    
    # Calculate statistics
    scores = [p['quality_assessment'].get('overall_score', 0) for p in final_pairs]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    
    auto_count = len([p for p in final_pairs if p.get('approval_source') == 'auto'])
    manual_count = len([p for p in final_pairs if p.get('approval_source') == 'manual'])
    
    # Save final dataset
    print(f"\n💾 Saving final dataset to {FINAL_OUTPUT_FILE}...")
    save_pairs(FINAL_OUTPUT_FILE, final_pairs)
    
    # Save statistics
    stats = {
        'created_at': datetime.now().isoformat(),
        'total_pairs': len(final_pairs),
        'auto_approved': auto_count,
        'manually_approved': manual_count,
        'avg_score': round(avg_score, 2),
        'min_score': round(min_score, 2),
        'max_score': round(max_score, 2),
        'source_files': {
            'auto_approved': AUTO_APPROVED_FILE,
            'manually_approved': MANUALLY_APPROVED_FILE
        }
    }
    
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ FINAL DATASET CREATED!")
    print("=" * 80)
    print(f"\n📊 STATISTICS:")
    print(f"   Total pairs: {len(final_pairs)}")
    print(f"   - Auto-approved (score >= 8.5): {auto_count}")
    print(f"   - Manually approved: {manual_count}")
    print(f"\n   Score Statistics:")
    print(f"   - Average score: {avg_score:.2f}/10")
    print(f"   - Min score: {min_score:.2f}/10")
    print(f"   - Max score: {max_score:.2f}/10")
    print(f"\n📁 OUTPUT FILES:")
    print(f"   - Dataset: {FINAL_OUTPUT_FILE}")
    print(f"   - Stats: {STATS_FILE}")
    
    # Quality breakdown
    print(f"\n📈 QUALITY BREAKDOWN:")
    excellent = len([s for s in scores if s >= 9.0])
    good = len([s for s in scores if 8.0 <= s < 9.0])
    fair = len([s for s in scores if 7.5 <= s < 8.0])
    print(f"   Excellent (9.0+): {excellent} ({excellent/len(final_pairs)*100:.1f}%)")
    print(f"   Good (8.0-9.0): {good} ({good/len(final_pairs)*100:.1f}%)")
    print(f"   Fair (7.5-8.0): {fair} ({fair/len(final_pairs)*100:.1f}%)")
    
    print("\n🎉 Dataset ready for RAG-RL training!")


if __name__ == "__main__":
    merge_datasets()

