#!/usr/bin/env python3
"""
Test FlashRAG retrieval with both custom and Wikipedia corpora.

This script verifies that:
1. Custom corpus BM25 index works correctly
2. Wikipedia DPR corpus BM25 index works correctly
3. Custom datasets are loadable in FlashRAG format
"""

import sys
import os

# Add FlashRAG to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'rag', 'FlashRAG'))

from flashrag.retriever import BM25Retriever
import json

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration for retrievers
CUSTOM_CONFIG = {
    "retrieval_method": "bm25",
    "index_path": os.path.join(BASE_DIR, "indexes", "custom_bm25"),
    "corpus_path": os.path.join(BASE_DIR, "corpus", "custom", "corpus.jsonl"),
    "bm25_backend": "bm25s",
    "retrieval_topk": 3,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "use_reranker": False,
    "silent_retrieval": False
}

WIKI_CONFIG = {
    "retrieval_method": "bm25",
    "index_path": os.path.join(BASE_DIR, "indexes", "wiki_bm25", "bm25"),
    "corpus_path": os.path.join(BASE_DIR, "corpus", "wiki", "wiki_dpr.jsonl"),
    "bm25_backend": "bm25s",
    "retrieval_topk": 3,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "use_reranker": False,
    "silent_retrieval": False
}


def test_retriever(config: dict, name: str, queries: list):
    """Test a retriever with sample queries."""
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print(f"{'='*70}")
    print(f"Index: {config['index_path']}")
    print(f"Corpus: {config['corpus_path']}")
    
    # Check paths exist
    if not os.path.exists(config['index_path']):
        print(f"❌ Index path does not exist!")
        return False
    if not os.path.exists(config['corpus_path']):
        print(f"❌ Corpus path does not exist!")
        return False
    
    try:
        print("\nLoading retriever...")
        retriever = BM25Retriever(config)
        print("✅ Retriever loaded successfully")
        
        for query in queries:
            print(f"\n📝 Query: \"{query}\"")
            print("-" * 50)
            
            results, scores = retriever.search(query, num=3, return_score=True)
            
            for i, (result, score) in enumerate(zip(results, scores), 1):
                contents = result.get('contents', '')
                # Extract title if present
                if '\n' in contents:
                    title = contents.split('\n')[0]
                    text = contents.split('\n', 1)[1][:200]
                else:
                    title = "N/A"
                    text = contents[:200]
                
                print(f"\n  Result {i} (score: {score:.4f}):")
                print(f"    Title: {title}")
                print(f"    Content: {text}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset(dataset_path: str, name: str):
    """Test loading a FlashRAG dataset."""
    print(f"\n{'='*70}")
    print(f"Testing Dataset: {name}")
    print(f"{'='*70}")
    print(f"Path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file does not exist!")
        return False
    
    try:
        items = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        print(f"✅ Loaded {len(items)} items")
        
        # Validate format
        required_fields = ['id', 'question', 'golden_answers']
        sample = items[0] if items else {}
        
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
        
        # Show sample
        if items:
            print(f"\n📋 Sample item:")
            print(f"   ID: {items[0].get('id')}")
            print(f"   Question: {items[0].get('question', '')[:100]}...")
            answers = items[0].get('golden_answers', [])
            if answers:
                print(f"   Answer: {answers[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("FLASHRAG DATA VERIFICATION TEST")
    print("=" * 70)
    
    results = {}
    
    # Test custom retrieval
    custom_queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning algorithms"
    ]
    results['custom_retrieval'] = test_retriever(CUSTOM_CONFIG, "Custom Corpus BM25", custom_queries)
    
    # Test wiki retrieval
    wiki_queries = [
        "What is artificial intelligence?",
        "History of the United States",
        "Theory of relativity Einstein"
    ]
    results['wiki_retrieval'] = test_retriever(WIKI_CONFIG, "Wikipedia DPR BM25", wiki_queries)
    
    # Test datasets
    train_dataset = os.path.join(BASE_DIR, "datasets", "custom_dataset", "train.jsonl")
    test_dataset = os.path.join(BASE_DIR, "datasets", "custom_dataset", "test.jsonl")
    
    results['train_dataset'] = verify_dataset(train_dataset, "Custom Train Dataset")
    results['test_dataset'] = verify_dataset(test_dataset, "Custom Test Dataset")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 All tests passed! FlashRAG data is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

