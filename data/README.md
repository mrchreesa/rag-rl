# Data Directory Structure

This directory contains all data files for the FlashRAG project, organized for easy use.

## Directory Structure

```
data/
├── corpus/                     # Document corpora for retrieval
│   ├── custom/                 # Custom PDF-derived corpus
│   │   ├── corpus.jsonl        # Chunked corpus (106MB, ~34K chunks)
│   │   ├── test_corpus.jsonl   # Test split chunks
│   │   ├── train_raw.jsonl     # Raw documents (unchunked)
│   │   └── test_raw.jsonl      # Raw test documents
│   └── wiki/                   # Wikipedia corpus
│       ├── wiki_dpr.jsonl      # DPR Wikipedia (14GB, 21M passages)
│       └── psgs_w100.tsv.gz    # Original compressed file
│
├── datasets/                   # QA datasets in FlashRAG format
│   ├── custom_dataset/         # Main dataset for training/testing
│   │   ├── train.jsonl         # 492 QA pairs
│   │   └── test.jsonl          # 87 QA pairs
│   ├── rejected/               # Filtered out QA pairs
│   └── verification_samples/   # Quality verification samples
│
├── indexes/                    # BM25 retrieval indexes
│   ├── custom_bm25/            # Index for custom corpus
│   └── wiki_bm25/bm25/         # Index for Wikipedia (23GB)
│
├── docs/                       # Source PDF documents
│   ├── train/
│   └── test/
│
├── scripts/                    # Utility scripts
│   ├── preprocess_pdf_corpus.py
│   ├── chunk_corpus.py
│   ├── convert_to_flashrag.py
│   ├── download_wiki_corpus.py
│   ├── assess_quality.py
│   └── ...
│
└── test_flashrag.py            # Verification test script
```

## Quick Start

### Test the Setup

```bash
cd /Users/kreeza/Desktop/Programming/FinalYearProject-RAG-RL
conda activate flashrag
python data/test_flashrag.py
```

### Using Custom Corpus

```python
from flashrag.retriever import BM25Retriever

config = {
    "retrieval_method": "bm25",
    "index_path": "data/indexes/custom_bm25",
    "corpus_path": "data/corpus/custom/corpus.jsonl",
    "bm25_backend": "bm25s",
    "retrieval_topk": 5,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "use_reranker": False,
    "silent_retrieval": False
}

retriever = BM25Retriever(config)
results, scores = retriever.search("your query", num=5, return_score=True)
```

### Using Wikipedia Corpus

```python
config = {
    "retrieval_method": "bm25",
    "index_path": "data/indexes/wiki_bm25/bm25",
    "corpus_path": "data/corpus/wiki/wiki_dpr.jsonl",
    "bm25_backend": "bm25s",
    # ... same other config options
}
```

### Loading Datasets

```python
from flashrag.dataset import Dataset

# Load custom dataset
dataset = Dataset(
    config,
    dataset_name="custom_dataset",
    data_dir="data/datasets/",
    split="train"
)
```

## File Formats

### Corpus Format (JSONL)

```json
{ "id": "0", "contents": "Title\nDocument text content...", "doc_id": "original_id" }
```

### Dataset Format (JSONL)

```json
{
	"id": "train_0",
	"question": "What is...",
	"golden_answers": ["Answer text"],
	"metadata": { "type": "factual", "source": "...", "quality_score": 8.5 }
}
```

## Statistics

| Resource             | Size   | Count               |
| -------------------- | ------ | ------------------- |
| Custom Corpus        | 106 MB | ~34,000 chunks      |
| Wikipedia Corpus     | 14 GB  | 21,015,324 passages |
| Custom BM25 Index    | 110 MB | -                   |
| Wikipedia BM25 Index | 23 GB  | -                   |
| Train Dataset        | -      | 492 QA pairs        |
| Test Dataset         | -      | 87 QA pairs         |
