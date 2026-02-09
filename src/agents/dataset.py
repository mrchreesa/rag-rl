"""
Dataset Adapters for RL-RAG Training

Provides dataset loading and formatting for Agent Lightning training.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union


class RAGDataset:
    """
    Base dataset class for RAG training.
    
    Implements the Dataset protocol expected by Agent Lightning.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize the dataset.
        
        Args:
            data: List of task dictionaries
        """
        self._data = data
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over tasks."""
        return iter(self._data)
    
    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get task by index."""
        return self._data[idx]
    
    def sample(self, n: int, seed: Optional[int] = None) -> "RAGDataset":
        """
        Return a random sample of the dataset.
        
        Args:
            n: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            New RAGDataset with sampled data
        """
        if seed is not None:
            random.seed(seed)
        
        n = min(n, len(self._data))
        sampled = random.sample(self._data, n)
        return RAGDataset(sampled)
    
    def shuffle(self, seed: Optional[int] = None) -> "RAGDataset":
        """
        Return a shuffled version of the dataset.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            New RAGDataset with shuffled data
        """
        if seed is not None:
            random.seed(seed)
        
        shuffled = list(self._data)
        random.shuffle(shuffled)
        return RAGDataset(shuffled)


def load_hotpotqa(
    split: str = "dev",
    data_dir: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> RAGDataset:
    """
    Load HotpotQA dataset for RL training.
    
    Args:
        split: Dataset split ("train" or "dev")
        data_dir: Directory containing hotpotqa files
        sample_size: If provided, return a random sample of this size
        seed: Random seed for sampling
        
    Returns:
        RAGDataset instance
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data/benchmarks/hotpotqa"
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / f"{split}.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"HotpotQA {split} file not found at {file_path}")
    
    data = []
    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            task = {
                "id": item.get("id", f"hotpot_{len(data)}"),
                "question": item["question"],
                "golden_answers": item["golden_answers"] if isinstance(item["golden_answers"], list) 
                                  else [item["golden_answers"]],
                "metadata": item.get("metadata", {})
            }
            data.append(task)
    
    dataset = RAGDataset(data)
    
    if sample_size is not None:
        dataset = dataset.sample(sample_size, seed=seed)
    
    return dataset


def load_custom_dataset(
    split: str = "test",
    data_dir: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> RAGDataset:
    """
    Load custom academic dataset for RL training.
    
    Args:
        split: Dataset split ("train" or "test")
        data_dir: Directory containing dataset files
        sample_size: If provided, return a random sample of this size
        seed: Random seed for sampling
        
    Returns:
        RAGDataset instance
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data/datasets/custom_dataset"
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / f"{split}.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Custom dataset {split} file not found at {file_path}")
    
    data = []
    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            task = {
                "id": item.get("id", f"custom_{len(data)}"),
                "question": item["question"],
                "golden_answers": item["golden_answers"] if isinstance(item["golden_answers"], list)
                                  else [item["golden_answers"]],
                "metadata": item.get("metadata", {})
            }
            data.append(task)
    
    dataset = RAGDataset(data)
    
    if sample_size is not None:
        dataset = dataset.sample(sample_size, seed=seed)
    
    return dataset


def load_combined_dataset(
    hotpotqa_size: int = 500,
    custom_size: int = 50,
    seed: int = 42
) -> RAGDataset:
    """
    Load a combined dataset from HotpotQA and custom data.
    
    Useful for training with diverse question types.
    
    Args:
        hotpotqa_size: Number of HotpotQA samples
        custom_size: Number of custom dataset samples
        seed: Random seed for sampling
        
    Returns:
        RAGDataset with combined data
    """
    hotpot = load_hotpotqa("dev", sample_size=hotpotqa_size, seed=seed)
    custom = load_custom_dataset("train", sample_size=custom_size, seed=seed)
    
    # Combine and shuffle
    combined_data = list(hotpot) + list(custom)
    random.seed(seed)
    random.shuffle(combined_data)
    
    return RAGDataset(combined_data)


class StreamingDataset:
    """
    Streaming dataset that yields tasks one at a time.
    
    Useful for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        sample_size: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            file_path: Path to JSONL file
            sample_size: If provided, stream only this many samples
            seed: Random seed for sampling
        """
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.seed = seed
        
        # Pre-compute line offsets if sampling
        if sample_size is not None:
            self._compute_sample_indices()
        else:
            self._sample_indices = None
    
    def _compute_sample_indices(self):
        """Pre-compute which line indices to sample."""
        # Count total lines
        with open(self.file_path) as f:
            total_lines = sum(1 for _ in f)
        
        # Sample indices
        random.seed(self.seed)
        n = min(self.sample_size, total_lines)
        self._sample_indices = set(random.sample(range(total_lines), n))
    
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate over tasks, yielding one at a time."""
        with open(self.file_path) as f:
            for idx, line in enumerate(f):
                if self._sample_indices is not None:
                    if idx not in self._sample_indices:
                        continue
                
                item = json.loads(line)
                yield {
                    "id": item.get("id", f"stream_{idx}"),
                    "question": item["question"],
                    "golden_answers": item["golden_answers"] if isinstance(item["golden_answers"], list)
                                      else [item["golden_answers"]],
                    "metadata": item.get("metadata", {})
                }


# Utility function to check dataset statistics
def dataset_stats(dataset: RAGDataset) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: RAGDataset instance
        
    Returns:
        Dictionary with statistics
    """
    if len(dataset) == 0:
        return {"count": 0}
    
    question_lengths = [len(task["question"]) for task in dataset]
    answer_lengths = [
        len(task["golden_answers"][0]) if task["golden_answers"] else 0
        for task in dataset
    ]
    
    # Count question types from metadata
    types = {}
    for task in dataset:
        qtype = task.get("metadata", {}).get("type", "unknown")
        types[qtype] = types.get(qtype, 0) + 1
    
    return {
        "count": len(dataset),
        "avg_question_length": sum(question_lengths) / len(question_lengths),
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
        "question_types": types
    }

