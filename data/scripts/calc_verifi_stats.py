import json

def calculate_stats():
    """Calculate verification statistics."""
    
    # Load your review notes (you'll need to create this)
    # Format: {"pair_id": {"status": "good/bad", "issues": [...]}}
    
    # Example structure:
    review_results = {
        # "1": {"status": "good", "issues": []},
        # "2": {"status": "bad", "issues": ["math_error"]},
    }
    
    total_reviewed = len(review_results)
    
    if total_reviewed == 0:
        print("⚠️  No review results found.")
        print("   Please add your review results to the 'review_results' dictionary.")
        print("   Format: {\"pair_id\": {\"status\": \"good/bad\", \"issues\": [...]}}")
        return
    
    good_pairs = sum(1 for r in review_results.values() if r['status'] == 'good')
    bad_pairs = total_reviewed - good_pairs
    
    print(f"Verification Results:")
    print(f"  Total reviewed: {total_reviewed}")
    print(f"  Good pairs: {good_pairs} ({good_pairs/total_reviewed*100:.1f}%)")
    print(f"  Bad pairs: {bad_pairs} ({bad_pairs/total_reviewed*100:.1f}%)")
    
    if bad_pairs / total_reviewed > 0.1:  # More than 10% bad
        print("\n⚠️  Warning: High error rate detected. Consider:")
        print("   - Adjusting quality criteria")
        print("   - Using a better model (Claude Sonnet)")
        print("   - Manual filtering of problematic pairs")

if __name__ == "__main__":
    calculate_stats()