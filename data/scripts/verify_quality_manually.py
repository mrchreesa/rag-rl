import json
import random
from typing import List, Dict

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

def sample_for_review(filtered_file: str, rejected_file: str, n_filtered=50, n_rejected=20):
    """Sample pairs for manual review."""
    
    # Load datasets
    filtered = load_jsonl(filtered_file)
    rejected = load_jsonl(rejected_file)
    
    # Sample randomly
    filtered_sample = random.sample(filtered, min(n_filtered, len(filtered)))
    rejected_sample = random.sample(rejected, min(n_rejected, len(rejected)))
    
    # Sort by score for easier review
    filtered_sample.sort(key=lambda x: x['quality_assessment']['overall_score'], reverse=True)
    rejected_sample.sort(key=lambda x: x['quality_assessment']['overall_score'], reverse=True)
    
    # Save samples for review
    with open('verification_filtered_sample.jsonl', 'w', encoding='utf-8') as f:
        for pair in filtered_sample:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    with open('verification_rejected_sample.jsonl', 'w', encoding='utf-8') as f:
        for pair in rejected_sample:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✅ Created verification samples:")
    print(f"   - Filtered sample: {len(filtered_sample)} pairs → verification_filtered_sample.jsonl")
    print(f"   - Rejected sample: {len(rejected_sample)} pairs → verification_rejected_sample.jsonl")
    
    return filtered_sample, rejected_sample

def print_review_format(pairs: List[Dict], title: str):
    """Print pairs in review-friendly format."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    for i, qa in enumerate(pairs, 1):
        score = qa['quality_assessment']['overall_score']
        print(f"[{i}/{len(pairs)}] Score: {score:.2f}/10")
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer']}")
        print(f"Type: {qa['type']}")
        print(f"Source: {qa['source']}")
        print(f"Justification: {qa['quality_assessment']['justification']}")
        print(f"\nContext: {qa['context'][:200]}...")
        print(f"\n{'─'*80}\n")

if __name__ == "__main__":
    filtered_file = "./datasets/train_dataset_filtered.jsonl"
    rejected_file = "./datasets/train_dataset_filtered_rejected.jsonl"
    
    filtered_sample, rejected_sample = sample_for_review(filtered_file, rejected_file)
    
    # Print for review
    print_review_format(filtered_sample, "FILTERED PAIRS (Should be high quality)")
    print_review_format(rejected_sample, "REJECTED PAIRS (Check for false negatives)")