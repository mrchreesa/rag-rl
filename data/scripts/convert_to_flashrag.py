#!/usr/bin/env python3
"""
Convert custom QA dataset format to FlashRAG-compatible format.

Custom format:
{
    "question": str,
    "answer": str,
    "type": str,
    "source": str,
    "context": str,
    "detected_issues": [],
    "quality_assessment": {...}
}

FlashRAG format:
{
    "id": str,
    "question": str,
    "golden_answers": List[str],
    "metadata": dict
}
"""

import json
import os
import argparse
from typing import List, Dict, Any


def convert_to_flashrag_format(
    input_file: str,
    output_file: str,
    id_prefix: str = "item",
    include_context_in_metadata: bool = True
) -> Dict[str, Any]:
    """
    Convert a custom QA dataset to FlashRAG format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        id_prefix: Prefix for item IDs (e.g., "train", "test", "dev")
        include_context_in_metadata: Whether to include context in metadata
    
    Returns:
        Dictionary with conversion statistics
    """
    converted_items = []
    stats = {
        "total_items": 0,
        "converted_items": 0,
        "skipped_items": 0,
        "errors": []
    }
    
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            stats["total_items"] += 1
            
            try:
                item = json.loads(line)
                
                # Extract question and answer
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()
                
                if not question or not answer:
                    stats["skipped_items"] += 1
                    stats["errors"].append(f"Line {line_num}: Missing question or answer")
                    continue
                
                # Create FlashRAG format item
                flashrag_item = {
                    "id": f"{id_prefix}_{stats['converted_items']}",
                    "question": question,
                    "golden_answers": [answer],  # Convert single answer to list
                    "metadata": {}
                }
                
                # Add optional metadata
                if item.get("type"):
                    flashrag_item["metadata"]["type"] = item["type"]
                
                if item.get("source"):
                    flashrag_item["metadata"]["source"] = item["source"]
                
                if include_context_in_metadata and item.get("context"):
                    flashrag_item["metadata"]["context"] = item["context"]
                
                # Add quality score if available (useful for debugging/analysis)
                if item.get("quality_assessment", {}).get("overall_score"):
                    flashrag_item["metadata"]["quality_score"] = item["quality_assessment"]["overall_score"]
                
                converted_items.append(flashrag_item)
                stats["converted_items"] += 1
                
            except json.JSONDecodeError as e:
                stats["skipped_items"] += 1
                stats["errors"].append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                stats["skipped_items"] += 1
                stats["errors"].append(f"Line {line_num}: {e}")
    
    # Write output
    print(f"Writing to: {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return stats


def print_sample(file_path: str, num_samples: int = 3):
    """Print sample items from a file."""
    print(f"\nSample items from {file_path}:")
    print("-" * 60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            print(json.dumps(item, indent=2, ensure_ascii=False)[:500])
            print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert custom QA dataset to FlashRAG format")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--prefix", "-p", default="item", help="ID prefix (e.g., 'train', 'test')")
    parser.add_argument("--no-context", action="store_true", help="Exclude context from metadata")
    parser.add_argument("--show-samples", action="store_true", help="Show sample converted items")
    
    args = parser.parse_args()
    
    stats = convert_to_flashrag_format(
        input_file=args.input,
        output_file=args.output,
        id_prefix=args.prefix,
        include_context_in_metadata=not args.no_context
    )
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total items processed: {stats['total_items']}")
    print(f"Successfully converted: {stats['converted_items']}")
    print(f"Skipped (errors): {stats['skipped_items']}")
    
    if stats['errors']:
        print(f"\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")
    
    if args.show_samples:
        print_sample(args.output)


if __name__ == "__main__":
    # Default conversion for train and test datasets
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - run default conversion
        base_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(base_dir, "datasets")
        
        conversions = [
            {
                "input": os.path.join(datasets_dir, "train_dataset_custom.jsonl"),
                "output": os.path.join(datasets_dir, "custom_dataset/train.jsonl"),
                "prefix": "train"
            },
            {
                "input": os.path.join(datasets_dir, "test_dataset_custom.jsonl"),
                "output": os.path.join(datasets_dir, "custom_dataset/test.jsonl"),
                "prefix": "test"
            }
        ]
        
        print("=" * 60)
        print("CONVERTING CUSTOM DATASETS TO FLASHRAG FORMAT")
        print("=" * 60)
        
        for conv in conversions:
            if os.path.exists(conv["input"]):
                print(f"\nConverting {conv['prefix']} dataset...")
                stats = convert_to_flashrag_format(
                    input_file=conv["input"],
                    output_file=conv["output"],
                    id_prefix=conv["prefix"],
                    include_context_in_metadata=True
                )
                print(f"  ✓ Converted {stats['converted_items']} items")
            else:
                print(f"\n⚠ Skipping {conv['prefix']} - file not found: {conv['input']}")
        
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE!")
        print("=" * 60)
        print("\nOutput files created in: datasets/custom_dataset/")
        print("  - train.jsonl")
        print("  - test.jsonl")
        print("\nTo use with FlashRAG, set:")
        print("  data_dir: 'path/to/data/datasets/'")
        print("  dataset_name: 'custom_dataset'")
        print("  split: ['train', 'test']")
    else:
        main()

