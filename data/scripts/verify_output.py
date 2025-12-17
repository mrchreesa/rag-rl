#!/usr/bin/env python3
"""
Verify corpus and dataset files are correctly formatted for FlashRAG.
"""

import json
import os
from pathlib import Path
from collections import Counter


def verify_jsonl_format(file_path: str, required_fields: list = None, max_check: int = 1000):
    """
    Verify JSONL file format and structure.
    
    Args:
        file_path: Path to JSONL file
        required_fields: List of required field names
        max_check: Maximum number of lines to check (0 = all)
    
    Returns:
        Dictionary with verification results
    """
    if not os.path.exists(file_path):
        return {
            "valid": False,
            "error": f"File not found: {file_path}",
            "stats": {}
        }
    
    stats = {
        "total_lines": 0,
        "valid_json": 0,
        "invalid_json": 0,
        "missing_fields": [],
        "field_counts": Counter(),
        "sample_ids": [],
        "errors": []
    }
    
    required_fields = required_fields or []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_check > 0 and line_num > max_check:
                    break
                
                if not line.strip():
                    continue
                
                stats["total_lines"] += 1
                
                try:
                    item = json.loads(line)
                    stats["valid_json"] += 1
                    
                    # Check required fields
                    for field in required_fields:
                        if field not in item:
                            stats["missing_fields"].append(f"Line {line_num}: missing '{field}'")
                    
                    # Count fields
                    stats["field_counts"].update(item.keys())
                    
                    # Collect sample IDs
                    if len(stats["sample_ids"]) < 5:
                        stats["sample_ids"].append(item.get("id", f"line_{line_num}"))
                    
                except json.JSONDecodeError as e:
                    stats["invalid_json"] += 1
                    stats["errors"].append(f"Line {line_num}: JSON decode error - {e}")
    
    except Exception as e:
        return {
            "valid": False,
            "error": f"Error reading file: {e}",
            "stats": stats
        }
    
    # Determine validity
    is_valid = (
        stats["invalid_json"] == 0 and
        len(stats["missing_fields"]) == 0 and
        stats["valid_json"] > 0
    )
    
    return {
        "valid": is_valid,
        "stats": stats
    }


def verify_corpus_file(file_path: str):
    """Verify corpus file format (requires 'id' and 'contents')."""
    result = verify_jsonl_format(file_path, required_fields=["id", "contents"])
    
    if result["valid"]:
        # Additional corpus-specific checks
        stats = result["stats"]
        
        # Check contents format (should have title\ntext or \ntext)
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            contents = sample.get("contents", "")
            has_newline = "\n" in contents
            has_title = contents.startswith("\n") or contents.split("\n")[0].strip() != ""
        
        result["corpus_checks"] = {
            "has_newline_format": has_newline,
            "has_title": has_title,
            "sample_contents_length": len(contents)
        }
    
    return result


def verify_dataset_file(file_path: str):
    """Verify dataset file format (requires 'id', 'question', 'golden_answers')."""
    result = verify_jsonl_format(file_path, required_fields=["id", "question", "golden_answers"])
    
    if result["valid"]:
        # Additional dataset-specific checks
        stats = result["stats"]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
            golden_answers = sample.get("golden_answers", [])
        
        result["dataset_checks"] = {
            "golden_answers_is_list": isinstance(golden_answers, list),
            "sample_question_length": len(sample.get("question", "")),
            "has_metadata": "metadata" in sample
        }
    
    return result


def verify_index_directory(index_dir: str):
    """Verify index directory contains required files."""
    required_files = [
        "corpus.jsonl",
        "vocab.index.json",
        "params.index.json"
    ]
    
    results = {}
    for file_name in required_files:
        file_path = os.path.join(index_dir, file_name)
        exists = os.path.exists(file_path)
        results[file_name] = {
            "exists": exists,
            "size": os.path.getsize(file_path) if exists else 0
        }
    
    all_exist = all(r["exists"] for r in results.values())
    
    return {
        "valid": all_exist,
        "files": results
    }


def main():
    """Run comprehensive verification."""
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("FLASHRAG DATA VERIFICATION")
    print("=" * 70)
    print()
    
    # Verify corpus files
    print("📚 VERIFYING CORPUS FILES")
    print("-" * 70)
    
    corpus_files = [
        ("Train Corpus (Chunked)", "corpus/train_corpus_chunked.jsonl"),
        ("Test Corpus (Chunked)", "corpus/test_corpus_chunked.jsonl"),
        ("Train Corpus (Original)", "corpus/train_corpus.jsonl"),
        ("Test Corpus (Original)", "corpus/test_corpus.jsonl"),
    ]
    
    for name, rel_path in corpus_files:
        file_path = base_dir / rel_path
        print(f"\n{name}: {rel_path}")
        
        if not file_path.exists():
            print(f"  ❌ File not found")
            continue
        
        result = verify_corpus_file(str(file_path))
        stats = result["stats"]
        
        if result["valid"]:
            print(f"  ✅ Valid JSONL format")
            print(f"     - Total lines: {stats['total_lines']}")
            print(f"     - Valid JSON: {stats['valid_json']}")
            print(f"     - Fields: {', '.join(stats['field_counts'].keys())}")
            
            if "corpus_checks" in result:
                checks = result["corpus_checks"]
                print(f"     - Has title/newline format: {checks['has_newline_format']}")
                print(f"     - Sample contents length: {checks['sample_contents_length']} chars")
        else:
            print(f"  ❌ Invalid format")
            if "error" in result:
                print(f"     Error: {result['error']}")
            if stats.get("errors"):
                print(f"     Errors: {stats['errors'][:3]}")
    
    # Verify dataset files
    print("\n\n📊 VERIFYING DATASET FILES")
    print("-" * 70)
    
    dataset_files = [
        ("Train Dataset", "datasets/custom_dataset/train.jsonl"),
        ("Test Dataset", "datasets/custom_dataset/test.jsonl"),
    ]
    
    for name, rel_path in dataset_files:
        file_path = base_dir / rel_path
        print(f"\n{name}: {rel_path}")
        
        if not file_path.exists():
            print(f"  ❌ File not found")
            continue
        
        result = verify_dataset_file(str(file_path))
        stats = result["stats"]
        
        if result["valid"]:
            print(f"  ✅ Valid FlashRAG format")
            print(f"     - Total items: {stats['total_lines']}")
            print(f"     - Valid JSON: {stats['valid_json']}")
            
            if "dataset_checks" in result:
                checks = result["dataset_checks"]
                print(f"     - golden_answers is list: {checks['golden_answers_is_list']}")
                print(f"     - Has metadata: {checks['has_metadata']}")
        else:
            print(f"  ❌ Invalid format")
            if stats.get("errors"):
                print(f"     Errors: {stats['errors'][:3]}")
    
    # Verify index
    print("\n\n🔍 VERIFYING INDEX")
    print("-" * 70)
    
    index_dir = base_dir / "indexes" / "bm25"
    print(f"\nBM25 Index: indexes/bm25/")
    
    result = verify_index_directory(str(index_dir))
    
    if result["valid"]:
        print(f"  ✅ Index directory valid")
        for file_name, file_info in result["files"].items():
            size_mb = file_info["size"] / (1024 * 1024)
            print(f"     - {file_name}: {size_mb:.2f} MB")
    else:
        print(f"  ❌ Index directory incomplete")
        for file_name, file_info in result["files"].items():
            status = "✅" if file_info["exists"] else "❌"
            print(f"     {status} {file_name}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()