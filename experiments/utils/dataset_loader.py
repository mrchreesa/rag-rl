#!/usr/bin/env python3
"""
Custom dataset loader for FlashRAG.
Ensures FlashRAG can find and load your custom_dataset correctly.
"""

import os
import sys

# Add FlashRAG to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/rag/FlashRAG'))

from flashrag.config import Config
from flashrag.utils import get_dataset


def load_custom_dataset(config_path, split="test", base_dir=None):
    """
    Load custom dataset using FlashRAG's Dataset class.
    
    Args:
        config_path: Path to config YAML file
        split: Dataset split to load ('train' or 'test')
        base_dir: Base directory (defaults to project root)
    
    Returns:
        Dataset object containing the loaded data
    """
    if base_dir is None:
        # Get project root (3 levels up from experiments/utils/)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Construct dataset path
    dataset_path = os.path.join(base_dir, "data", "datasets", "custom_dataset")
    
    # Create config dict with dataset_path
    config_dict = {
        "dataset_path": dataset_path,
        "split": [split],
        "test_sample_num": None,  # Load all samples
        "random_sample": False,
    }
    
    # Load config
    config = Config(config_path, config_dict)
    
    # Load dataset
    dataset = get_dataset(config)
    
    return dataset[split]


def verify_dataset_format(dataset):
    """
    Verify that the dataset has the correct format for FlashRAG.
    
    Args:
        dataset: FlashRAG Dataset object
    
    Returns:
        dict with verification results
    """
    results = {
        "total_samples": len(dataset.data),
        "has_required_fields": True,
        "missing_fields": [],
        "sample_info": {}
    }
    
    if len(dataset.data) == 0:
        results["has_required_fields"] = False
        results["error"] = "Dataset is empty"
        return results
    
    # Check first sample
    sample = dataset.data[0]
    required_fields = ["id", "question", "golden_answers"]
    
    for field in required_fields:
        if not hasattr(sample, field) or getattr(sample, field) is None:
            results["missing_fields"].append(field)
            results["has_required_fields"] = False
    
    # Get sample info
    results["sample_info"] = {
        "id": sample.id,
        "question": sample.question[:100] + "..." if sample.question and len(sample.question) > 100 else sample.question,
        "golden_answers_count": len(sample.golden_answers) if sample.golden_answers else 0,
        "has_metadata": hasattr(sample, "metadata") and sample.metadata is not None
    }
    
    return results


if __name__ == "__main__":
    """Test dataset loading."""
    print("=" * 70)
    print("TESTING CUSTOM DATASET LOADING")
    print("=" * 70)
    
    # Get paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    config_path = os.path.join(base_dir, "experiments", "configs", "custom_baseline.yaml")
    
    # Test loading test set
    print("\n📂 Loading test dataset...")
    try:
        test_data = load_custom_dataset(config_path, split="test", base_dir=base_dir)
        print(f"✅ Successfully loaded test dataset")
        print(f"   Total samples: {len(test_data.data)}")
        
        # Verify format
        print("\n🔍 Verifying dataset format...")
        verification = verify_dataset_format(test_data)
        
        if verification["has_required_fields"]:
            print("✅ Dataset format is correct")
            print(f"   Total samples: {verification['total_samples']}")
            print(f"   Sample ID: {verification['sample_info']['id']}")
            print(f"   Question preview: {verification['sample_info']['question']}")
            print(f"   Golden answers: {verification['sample_info']['golden_answers_count']}")
            print(f"   Has metadata: {verification['sample_info']['has_metadata']}")
        else:
            print("❌ Dataset format issues:")
            print(f"   Missing fields: {verification['missing_fields']}")
            
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test loading train set
    print("\n📂 Loading train dataset...")
    try:
        train_data = load_custom_dataset(config_path, split="train", base_dir=base_dir)
        print(f"✅ Successfully loaded train dataset")
        print(f"   Total samples: {len(train_data.data)}")
    except Exception as e:
        print(f"❌ Error loading train dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DATASET LOADING TEST COMPLETE")
    print("=" * 70)

