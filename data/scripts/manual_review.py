#!/usr/bin/env python3
"""
Manual Review Tool for Borderline QA Pairs

This script helps you review borderline QA pairs (scores 7.5-8.5) 
and mark them as approved or rejected.

Usage:
    python manual_review.py

Output:
    - datasets/train_dataset_manually_approved.jsonl (pairs you approved)
    - datasets/train_dataset_manually_rejected.jsonl (pairs you rejected)
"""

import os
import json
from typing import List, Dict, Any

BORDERLINE_FILE = "./datasets/train_dataset_borderline.jsonl"
APPROVED_FILE = "./datasets/train_dataset_manually_approved.jsonl"
REJECTED_FILE = "./datasets/train_dataset_manually_rejected.jsonl"
PROGRESS_FILE = "./datasets/.manual_review_progress.json"


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


def load_progress() -> Dict:
    """Load review progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'reviewed_indices': [], 'approved': [], 'rejected': []}


def save_progress(progress: Dict):
    """Save review progress."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def display_pair(pair: Dict[str, Any], index: int, total: int):
    """Display a QA pair for review."""
    score = pair['quality_assessment'].get('overall_score', 0)
    justification = pair['quality_assessment'].get('justification', 'N/A')
    
    print("\n" + "=" * 80)
    print(f"[{index + 1}/{total}] Score: {score:.2f}/10")
    print("=" * 80)
    
    print(f"\n📝 QUESTION:")
    print(f"   {pair['question']}")
    
    print(f"\n✅ ANSWER:")
    print(f"   {pair['answer']}")
    
    print(f"\n📚 CONTEXT (first 300 chars):")
    print(f"   {pair['context'][:300]}...")
    
    print(f"\n🏷️  TYPE: {pair.get('type', 'N/A')}")
    print(f"📄 SOURCE: {pair.get('source', 'N/A')}")
    
    print(f"\n🤖 CLAUDE'S JUSTIFICATION:")
    print(f"   {justification}")
    
    if pair.get('detected_issues'):
        print(f"\n⚠️  DETECTED ISSUES: {', '.join(pair['detected_issues'])}")
    
    print("\n" + "-" * 80)


def review_pairs():
    """Interactive review of borderline pairs."""
    
    print("=" * 80)
    print("MANUAL REVIEW TOOL FOR BORDERLINE QA PAIRS")
    print("=" * 80)
    
    # Load borderline pairs
    if not os.path.exists(BORDERLINE_FILE):
        print(f"\n❌ Borderline file not found: {BORDERLINE_FILE}")
        print("   Run assess_quality.py first to generate borderline pairs.")
        return
    
    pairs = load_pairs(BORDERLINE_FILE)
    if not pairs:
        print("\n✅ No borderline pairs to review!")
        return
    
    print(f"\nLoaded {len(pairs)} borderline pairs for review.")
    
    # Load progress
    progress = load_progress()
    reviewed_indices = set(progress.get('reviewed_indices', []))
    approved = progress.get('approved', [])
    rejected = progress.get('rejected', [])
    
    print(f"Already reviewed: {len(reviewed_indices)}/{len(pairs)}")
    print(f"Approved so far: {len(approved)}")
    print(f"Rejected so far: {len(rejected)}")
    
    print("\n📋 INSTRUCTIONS:")
    print("   [a] - Approve (include in final dataset)")
    print("   [r] - Reject (exclude from final dataset)")
    print("   [s] - Skip (review later)")
    print("   [q] - Quit and save progress")
    print("   [h] - Show help")
    
    # Review loop
    for i, pair in enumerate(pairs):
        if i in reviewed_indices:
            continue
        
        display_pair(pair, i, len(pairs))
        
        while True:
            choice = input("\n>>> Your decision [a/r/s/q/h]: ").strip().lower()
            
            if choice == 'a':
                approved.append(pair)
                reviewed_indices.add(i)
                print("   ✅ APPROVED")
                break
            elif choice == 'r':
                rejected.append(pair)
                reviewed_indices.add(i)
                print("   ❌ REJECTED")
                break
            elif choice == 's':
                print("   ⏭️  SKIPPED")
                break
            elif choice == 'q':
                # Save progress and exit
                progress = {
                    'reviewed_indices': list(reviewed_indices),
                    'approved': approved,
                    'rejected': rejected
                }
                save_progress(progress)
                save_pairs(APPROVED_FILE, approved)
                save_pairs(REJECTED_FILE, rejected)
                
                print(f"\n💾 Progress saved!")
                print(f"   Approved: {len(approved)}")
                print(f"   Rejected: {len(rejected)}")
                print(f"   Remaining: {len(pairs) - len(reviewed_indices)}")
                return
            elif choice == 'h':
                print("\n📋 HELP:")
                print("   [a] - Approve: Include this QA pair in the final dataset")
                print("   [r] - Reject: Exclude this QA pair from the final dataset")
                print("   [s] - Skip: Come back to this pair later")
                print("   [q] - Quit: Save progress and exit")
                print("\n   Quality Criteria:")
                print("   - Is the question testing meaningful understanding?")
                print("   - Is the answer accurate and complete?")
                print("   - Is the answer supported by the context?")
                print("   - Is there educational value?")
            else:
                print("   Invalid choice. Enter a, r, s, q, or h.")
    
    # All pairs reviewed
    progress = {
        'reviewed_indices': list(reviewed_indices),
        'approved': approved,
        'rejected': rejected
    }
    save_progress(progress)
    save_pairs(APPROVED_FILE, approved)
    save_pairs(REJECTED_FILE, rejected)
    
    print("\n" + "=" * 80)
    print("🎉 REVIEW COMPLETE!")
    print("=" * 80)
    print(f"   Total reviewed: {len(reviewed_indices)}/{len(pairs)}")
    print(f"   Approved: {len(approved)} → {APPROVED_FILE}")
    print(f"   Rejected: {len(rejected)} → {REJECTED_FILE}")
    print(f"\n   Next step: Run merge_final.py to create the final dataset")


if __name__ == "__main__":
    review_pairs()

