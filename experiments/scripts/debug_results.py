#!/usr/bin/env python3
"""
Debug RAG results to understand why metrics are low.
Analyzes retrieval quality, answer quality, and identifies issues.
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter

# Add FlashRAG to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/rag/FlashRAG'))

from flashrag.evaluator.utils import normalize_answer


def load_results(results_dir):
    """Load results from experiment output."""
    intermediate_file = os.path.join(results_dir, "intermediate_data.json")
    metric_file = os.path.join(results_dir, "metric_score.txt")
    
    results = []
    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    metrics = {}
    if os.path.exists(metric_file):
        with open(metric_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except:
                        metrics[key.strip()] = value.strip()
    
    return results, metrics


def analyze_retrieval_quality(results, top_k=5):
    """Analyze retrieval quality - check if retrieved docs contain answers."""
    print("=" * 70)
    print("RETRIEVAL QUALITY ANALYSIS")
    print("=" * 70)
    
    retrieval_hits = 0
    total_samples = 0
    sample_issues = []
    
    for item in results[:10]:  # Analyze first 10 samples
        question = item.get('question', '')
        golden_answers = item.get('golden_answers', [])
        
        # Check different possible locations for retrieval_result
        retrieval_result = item.get('retrieval_result', [])
        if not retrieval_result:
            retrieval_result = item.get('output', {}).get('retrieval_result', [])
        
        pred = item.get('pred', '')
        if not pred:
            pred = item.get('output', {}).get('pred', '')
        
        if not golden_answers:
            continue
        
        if not retrieval_result:
            print(f"   ⚠️  No retrieval results for question: {question[:80]}...")
            continue
        
        total_samples += 1
        
        # Check if any retrieved doc contains the answer
        hit = False
        retrieved_texts = []
        for doc in retrieval_result[:top_k]:
            content = doc.get('contents', '')
            retrieved_texts.append(content[:200] + "..." if len(content) > 200 else content)
            
            # Check if answer is in retrieved doc
            for answer in golden_answers:
                normalized_answer = normalize_answer(answer)
                normalized_content = normalize_answer(content)
                if normalized_answer in normalized_content:
                    hit = True
                    break
        
        if hit:
            retrieval_hits += 1
        else:
            # Store problematic samples
            sample_issues.append({
                'question': question,
                'golden_answer': golden_answers[0],
                'retrieved_preview': retrieved_texts[0] if retrieved_texts else "No documents retrieved",
                'predicted': pred[:100] if pred else "No prediction"
            })
    
    print(f"\n📊 Retrieval Statistics (first 10 samples):")
    print(f"   Total samples analyzed: {total_samples}")
    print(f"   Retrieval hits (answer in top-{top_k} docs): {retrieval_hits}")
    if total_samples > 0:
        print(f"   Retrieval miss rate: {(total_samples - retrieval_hits) / total_samples * 100:.1f}%")
    else:
        print(f"   ⚠️  No samples with retrieval results found")
    
    if sample_issues:
        print(f"\n❌ Sample Retrieval Failures:")
        for i, issue in enumerate(sample_issues[:3], 1):
            print(f"\n   Example {i}:")
            print(f"   Question: {issue['question'][:150]}...")
            print(f"   Expected Answer: {issue['golden_answer'][:100]}...")
            print(f"   Retrieved Doc Preview: {issue['retrieved_preview'][:150]}...")
            print(f"   Predicted Answer: {issue['predicted']}")
    
    return retrieval_hits, total_samples


def analyze_answer_quality(results):
    """Analyze answer quality - compare predictions vs golden answers."""
    print("\n" + "=" * 70)
    print("ANSWER QUALITY ANALYSIS")
    print("=" * 70)
    
    exact_matches = 0
    partial_matches = 0
    empty_predictions = 0
    sample_comparisons = []
    
    for item in results[:10]:  # Analyze first 10 samples
        question = item.get('question', '')
        golden_answers = item.get('golden_answers', [])
        pred = item.get('pred', '')
        if not pred:
            pred = item.get('output', {}).get('pred', '')
        
        if not golden_answers:
            continue
        
        # Check exact match
        normalized_pred = normalize_answer(pred)
        normalized_golden = normalize_answer(golden_answers[0])
        
        if normalized_pred == normalized_golden:
            exact_matches += 1
        elif not pred or pred.strip() == "":
            empty_predictions += 1
        elif any(word in normalized_pred for word in normalized_golden.split()[:3]):
            partial_matches += 1
        
        # Store for detailed analysis
        sample_comparisons.append({
            'question': question[:100],
            'golden': golden_answers[0][:100],
            'predicted': pred[:100] if pred else "(empty)",
            'exact_match': normalized_pred == normalized_golden
        })
    
    print(f"\n📊 Answer Quality Statistics (first 10 samples):")
    print(f"   Exact matches: {exact_matches}")
    print(f"   Partial matches: {partial_matches}")
    print(f"   Empty predictions: {empty_predictions}")
    
    print(f"\n📋 Sample Answer Comparisons:")
    for i, comp in enumerate(sample_comparisons[:5], 1):
        match_status = "✅ EXACT" if comp['exact_match'] else "❌ NO MATCH"
        print(f"\n   Example {i} ({match_status}):")
        print(f"   Question: {comp['question']}...")
        print(f"   Golden: {comp['golden']}...")
        print(f"   Predicted: {comp['predicted']}...")


def check_corpus_coverage(results, corpus_path):
    """Check if corpus contains answers to test questions."""
    print("\n" + "=" * 70)
    print("CORPUS COVERAGE ANALYSIS")
    print("=" * 70)
    
    # Load a sample of corpus
    print(f"\nLoading corpus from: {corpus_path}")
    corpus_samples = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100 documents
                    break
                corpus_samples.append(json.loads(line))
        print(f"   Loaded {len(corpus_samples)} corpus samples")
    except Exception as e:
        print(f"   ❌ Error loading corpus: {e}")
        return
    
    # Check if answers appear in corpus
    print(f"\nChecking if test answers appear in corpus...")
    answer_in_corpus = 0
    total_checked = 0
    
    for item in results[:5]:  # Check first 5 questions
        golden_answers = item.get('golden_answers', [])
        if not golden_answers:
            continue
        
        total_checked += 1
        answer = golden_answers[0]
        normalized_answer = normalize_answer(answer)
        
        # Search in corpus samples
        found = False
        for corpus_item in corpus_samples:
            content = corpus_item.get('contents', '')
            normalized_content = normalize_answer(content)
            if normalized_answer in normalized_content:
                found = True
                break
        
        if found:
            answer_in_corpus += 1
            print(f"   ✅ Answer found in corpus sample")
        else:
            print(f"   ❌ Answer NOT found: {answer[:80]}...")
    
    print(f"\n   Answers found in corpus: {answer_in_corpus}/{total_checked}")
    print(f"   Note: This only checks a sample of 100 corpus documents")


def analyze_retrieval_keywords(results):
    """Analyze what keywords are being retrieved vs what's in questions."""
    print("\n" + "=" * 70)
    print("KEYWORD ANALYSIS")
    print("=" * 70)
    
    question_keywords = []
    retrieved_keywords = []
    
    for item in results[:10]:
        question = item.get('question', '')
        retrieval_result = item.get('retrieval_result', [])
        if not retrieval_result:
            retrieval_result = item.get('output', {}).get('retrieval_result', [])
        
        # Extract keywords from question (simple approach)
        q_words = [w.lower() for w in question.split() if len(w) > 3]
        question_keywords.extend(q_words)
        
        # Extract keywords from retrieved docs
        for doc in retrieval_result[:2]:
            content = doc.get('contents', '')
            r_words = [w.lower() for w in content.split() if len(w) > 3]
            retrieved_keywords.extend(r_words[:50])  # Limit per doc
    
    # Find common keywords
    q_counter = Counter(question_keywords)
    r_counter = Counter(retrieved_keywords)
    
    print(f"\n📊 Top Question Keywords:")
    for word, count in q_counter.most_common(10):
        print(f"   {word}: {count}")
    
    print(f"\n📊 Top Retrieved Document Keywords:")
    for word, count in r_counter.most_common(10):
        print(f"   {word}: {count}")
    
    # Check overlap
    common = set(q_counter.keys()) & set(r_counter.keys())
    print(f"\n   Common keywords: {len(common)}")
    if common:
        print(f"   Examples: {list(common)[:10]}")


def main():
    """Main debugging function."""
    print("=" * 70)
    print("RAG RESULTS DEBUGGING")
    print("=" * 70)
    
    # Find latest results directory
    results_base = os.path.join(os.path.dirname(__file__), '../results')
    
    # Look for naive_rag results
    result_dirs = []
    for item in os.listdir(results_base):
        if 'naive_rag' in item:
            result_dirs.append(os.path.join(results_base, item))
    
    if not result_dirs:
        print(f"❌ No results found in {results_base}")
        return
    
    # Use most recent
    latest_dir = max(result_dirs, key=os.path.getmtime)
    print(f"\n📂 Analyzing results from: {os.path.basename(latest_dir)}")
    
    # Load results
    results, metrics = load_results(latest_dir)
    
    if not results:
        print("❌ No results data found")
        return
    
    print(f"   Loaded {len(results)} result samples")
    print(f"\n📊 Overall Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Run analyses
    analyze_retrieval_quality(results, top_k=5)
    analyze_answer_quality(results)
    
    # Check corpus coverage
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    corpus_path = os.path.join(base_dir, "data", "corpus", "custom", "corpus.jsonl")
    if os.path.exists(corpus_path):
        check_corpus_coverage(results, corpus_path)
    else:
        print(f"\n⚠️  Corpus file not found: {corpus_path}")
    
    analyze_retrieval_keywords(results)
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("DEBUGGING SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    if metrics.get('retrieval_recall_top5', 1.0) == 0.0:
        print("\n🔴 CRITICAL ISSUE: Retrieval is completely failing")
        print("   Recommendations:")
        print("   1. Check if corpus contains answers to test questions")
        print("   2. Try Wikipedia corpus instead of custom corpus")
        print("   3. Increase retrieval_topk (currently 5)")
        print("   4. Check if BM25 index is working correctly")
    
    if metrics.get('em', 1.0) == 0.0:
        print("\n⚠️  No exact matches found")
        print("   This is common for academic questions with precise answers")
        print("   Focus on improving F1 score instead")
    
    if metrics.get('f1', 1.0) < 0.1:
        print("\n⚠️  Very low F1 score (< 10%)")
        print("   Recommendations:")
        print("   1. Fix retrieval first (retrieval recall = 0.0)")
        print("   2. Check prompt quality")
        print("   3. Verify model is generating reasonable answers")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

