# Installation Guide for FinalYearProject-RAG-RL

Since this is a multi-component application (RAG + RL), all dependencies should be installed at the **root level** to ensure compatibility across components.

## Installation Steps

### 1. Install Python Dependencies from Root

From the project root directory:

```bash
# Install all Python dependencies
pip install -r requirements.txt

# Install FlashRAG as an editable package (so changes are reflected immediately)
pip install -e src/rag/FlashRAG

# Install optional FlashRAG dependencies if needed
pip install sentence-transformers>=3.0.1  # For easier retriever usage
pip install pyserini  # For BM25 retrieval
pip install vllm  # For faster LLM inference (requires CUDA)
```

### 2. Install FAISS (Required for Dense Retrieval)

FAISS cannot be installed via pip reliably, so use conda:

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU version (if you have CUDA)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

### 3. Verify Installation

```bash
# Test FlashRAG import
python -c "from flashrag.config import Config; print('FlashRAG installed successfully!')"

# Check if all dependencies are available
python -c "import torch, transformers, datasets; print('Core dependencies OK!')"
```

## Why Install at Root?

1. **Dependency Management**: Centralized dependency management prevents version conflicts between components
2. **Environment Consistency**: All components (RAG, RL) share the same Python environment
3. **Easier Maintenance**: Single `requirements.txt` makes it easier to track and update dependencies
4. **Import Paths**: Installing FlashRAG as editable (`-e`) allows imports like `from flashrag import ...` from anywhere in your project

## Project Structure

```
FinalYearProject-RAG-RL/
├── requirements.txt          # Root-level dependencies (includes FlashRAG deps)
├── src/
│   ├── rag/
│   │   └── FlashRAG/         # FlashRAG submodule (installed as editable package)
│   │       ├── requirements.txt  # FlashRAG's original requirements (for reference)
│   │       └── setup.py
│   └── rl/
│       └── agent-lightning/  # RL component
└── data/
```

## Using FlashRAG in Your Code

Once installed, you can import FlashRAG from anywhere in your project:

```python
# From any Python file in your project
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_dataset
```

## Troubleshooting

### Version Conflicts

If you encounter version conflicts:

1. Check which component requires which version
2. Update `requirements.txt` with compatible versions
3. Consider using separate virtual environments for different components (advanced)

### FAISS Installation Issues

-   Make sure you're using conda, not pip
-   For macOS ARM (M1/M2), you may need: `conda install -c conda-forge faiss-cpu`
-   Check FAISS compatibility: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

### Import Errors

If you get import errors:

```bash
# Reinstall FlashRAG as editable
pip install -e src/rag/FlashRAG --force-reinstall
```

## Optional: Virtual Environment

For better isolation, use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Then install dependencies
pip install -r requirements.txt
pip install -e src/rag/FlashRAG
```
