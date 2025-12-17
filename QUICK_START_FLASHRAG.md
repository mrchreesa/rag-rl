# Quick Start Guide for FlashRAG

## Step 1: Installation

**Important**: Since FlashRAG is part of a larger application, install dependencies at the **root level**:

```bash
# From the project root directory
pip install -r requirements.txt
pip install -e src/rag/FlashRAG
```

This ensures all components share the same environment and avoids version conflicts.

Alternatively, if you want to use FlashRAG standalone:

```bash
pip install flashrag-dev --pre
```

### Optional Dependencies

For better performance, install optional dependencies:

```bash
# Install all extra dependencies
pip install flashrag-dev[full]

# Or install individually:
pip install vllm>=0.4.1  # For faster LLM inference
pip install sentence-transformers  # For easier retriever usage
pip install pyserini  # For BM25 retrieval
```

### Install FAISS (Required for dense retrieval)

FAISS needs to be installed via conda:

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU version (if you have CUDA)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Step 2: Prepare Your Corpus

Create a corpus file in JSONL format (`corpus.jsonl`):

```jsonl
{"id": "0", "contents": "Your document text here..."}
{"id": "1", "contents": "Another document..."}
```

## Step 3: Build an Index

### For Dense Retrieval (E5, BGE, etc.)

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat
```

### For Sparse Retrieval (BM25)

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

## Step 4: Run a Simple RAG Pipeline

Create a Python script (`my_first_rag.py`):

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

# Configure FlashRAG
config_dict = {
    "data_dir": "dataset/",
    "index_path": "indexes/e5_Flat.index",  # Path to your built index
    "corpus_path": "corpus.jsonl",  # Path to your corpus
    "model2path": {
        "e5": "intfloat/e5-base-v2",
        "llama3-8B-instruct": "meta-llama/Meta-Llama-3-8B-Instruct"
    },
    "generator_model": "llama3-8B-instruct",
    "retrieval_method": "e5",
    "retrieval_topk": 5,  # Number of documents to retrieve
    "metrics": ["em", "f1"],
    "save_intermediate_data": True,
}

config = Config(config_dict=config_dict)

# Load dataset (or create your own)
all_split = get_dataset(config)
test_data = all_split["test"]

# Create prompt template
prompt_template = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. "
                  "Only give me the answer and do not output any other words.\n"
                  "The following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)

# Initialize and run pipeline
pipeline = SequentialPipeline(config, prompt_template=prompt_template)
output_dataset = pipeline.run(test_data, do_eval=True)

# View results
print("Generated answers:")
print(output_dataset.pred)
```

## Step 5: Try the UI (Optional)

FlashRAG includes a user-friendly UI:

```bash
cd src/rag/FlashRAG/webui
python interface.py
```

## Quick Test with Example

FlashRAG includes example files. Try running:

```bash
cd src/rag/FlashRAG/examples/quick_start
python simple_pipeline.py --model_path /path/to/your/model --retriever_path /path/to/retriever
```

## Next Steps

1. **Explore Components**: Check out the available retrievers, generators, and refiners
2. **Try Different Methods**: FlashRAG supports 23+ RAG methods (see README)
3. **Customize Pipelines**: Build your own pipeline by inheriting `BasicPipeline`
4. **Use Pre-built Datasets**: Download from [HuggingFace](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)

## Common Issues

-   **FAISS installation**: Use conda, not pip
-   **Model paths**: Make sure you have access to the models (HuggingFace token if needed)
-   **GPU memory**: Adjust `gpu_memory_utilization` in config if you run out of memory

## Resources

-   Full documentation: Check `src/rag/FlashRAG/docs/`
-   Configuration guide: `src/rag/FlashRAG/docs/original_docs/configuration.md`
-   Basic usage: `src/rag/FlashRAG/docs/original_docs/basic_usage.md`
