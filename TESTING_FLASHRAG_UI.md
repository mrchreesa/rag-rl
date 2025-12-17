# Testing FlashRAG UI - Step by Step Guide

## Prerequisites

Before testing, make sure you have:

1. ✅ Installed all dependencies (see `INSTALLATION.md`)
2. ✅ Fixed the Gradio compatibility issues (already done)
3. ✅ Example corpus and index files exist (they do!)

## Step 1: Launch the UI

From the project root:

```bash
cd src/rag/FlashRAG/webui
python interface.py
```

The UI will start and you should see:

-   A local URL (usually `http://127.0.0.1:7860`)
-   A public URL (if you want to share it)

Open the URL in your browser.

## Step 2: Configure Basic Settings

The UI has several tabs. Start with the **Basic Settings**:

### Required Fields:

1. **Language**: Select `en` (English) or `zh` (Chinese)

2. **Method**: Choose a RAG method from the dropdown:

    - Start with `Naive RAG` (simplest)
    - Or try `Standard RAG`

3. **GPU ID**:

    - If you have GPU: Enter GPU IDs like `0` or `0,1`
    - If CPU only: Leave empty or enter `cpu`

4. **Framework**:

    - `hf` - HuggingFace (works on CPU, slower)
    - `vllm` - vLLM (GPU only, faster)
    - `fschat` - FastChat (GPU recommended)
    - `openai` - OpenAI API

5. **Generator Settings**:

    - **Generator Name**: `llama3.1-8b-instruct` (or your model name)
    - **Generator Model Path**:
        - HuggingFace model: `meta-llama/Llama-3.1-8B-Instruct`
        - Local path: `/path/to/your/model`
        - **Note**: For testing without downloading large models, you can use smaller models like:
            - `microsoft/DialoGPT-small` (very small, for quick testing)
            - `gpt2` (small, for testing)

6. **Retrieval Settings**:

    - **Retrieval Method**: `e5`
    - **Retrieval Model Path**: `intfloat/e5-base-v2` (will download from HuggingFace)

7. **Corpus & Index Paths**:
    - **Corpus Path**: `examples/quick_start/indexes/general_knowledge.jsonl`
        - This is relative to the FlashRAG root directory
    - **Index Path**: `examples/quick_start/indexes/e5_Flat.index`
        - This is the pre-built index for the corpus

## Step 3: Test with Pre-built Example (Easiest)

The UI includes example files. Use these paths (relative to `src/rag/FlashRAG/`):

```
Corpus Path: examples/quick_start/indexes/general_knowledge.jsonl
Index Path: examples/quick_start/indexes/e5_Flat.index
```

### Minimal Test Configuration:

```
Method: Naive RAG
Framework: hf
Generator Model Path: gpt2  (small model for quick testing)
Retrieval Method: e5
Retrieval Model Path: intfloat/e5-base-v2
Corpus Path: examples/quick_start/indexes/general_knowledge.jsonl
Index Path: examples/quick_start/indexes/e5_Flat.index
```

## Step 4: Preview Configuration

1. Click on the **Preview** tab
2. Click **Preview Config** button
3. Review the generated YAML configuration
4. If everything looks good, click **Save Config** to save it for later

## Step 5: Test Chat Interface

1. Go to the **Chat** tab
2. Enter a question like:
    - "What is Artificial Intelligence?"
    - "What is Machine Learning?"
    - "Explain Deep Learning"
3. Click submit or press Enter
4. The system will:
    - Retrieve relevant documents
    - Generate an answer based on the retrieved context

## Step 6: Test Evaluation (Optional)

1. Go to the **Evaluate** tab
2. Configure dataset settings:
    - **Dataset Name**: Leave empty for now (or use a test dataset)
    - **Data Dir**: Path to your dataset directory
3. Click **Preview** to see the evaluation configuration
4. Click **Run** to start evaluation (this may take time)

## Step 7: Build Your Own Index (Optional)

If you want to use your own corpus:

1. Go to the **Index Builder** tab
2. Configure:
    - **Retrieval Method**: `e5` or `bm25`
    - **Model Path**: `intfloat/e5-base-v2` (for e5)
    - **Corpus Path**: Path to your JSONL corpus file
    - **Save Dir**: Where to save the index
3. Click **Build Index**

## Troubleshooting

### Issue: "Model not found"

-   **Solution**: Make sure you have internet connection for HuggingFace models
-   Or use local model paths if you've downloaded models

### Issue: "Index file not found"

-   **Solution**: Check that the index path is correct and relative to FlashRAG root
-   Or build a new index using the Index Builder tab

### Issue: "CUDA out of memory"

-   **Solution**:
    -   Use smaller models (e.g., `gpt2` instead of large LLMs)
    -   Reduce `generator_batch_size` in settings
    -   Use CPU framework (`hf` with CPU)

### Issue: "FAISS index error"

-   **Solution**: Make sure FAISS is installed correctly:
    ```bash
    conda install -c pytorch faiss-cpu=1.8.0
    ```

### Issue: UI doesn't load

-   **Solution**:
    -   Check that Gradio is installed: `pip install gradio>=5.0.0`
    -   Check the terminal for error messages

## Quick Test Checklist

-   [ ] UI launches without errors
-   [ ] Can see all tabs (Basic, Retrieve, Rerank, Generate, Method, Preview, Chat, Evaluate, Index Builder)
-   [ ] Can fill in basic settings
-   [ ] Preview config generates valid YAML
-   [ ] Chat interface responds (even if with a small model)
-   [ ] No import errors in terminal

## Example Test Questions

Try these questions with the example corpus:

1. "What is Artificial Intelligence?"
2. "What are the main categories of AI?"
3. "Explain the difference between Machine Learning and Deep Learning"
4. "What is Natural Language Processing?"

## Next Steps

Once basic testing works:

1. **Use Better Models**: Replace `gpt2` with larger models like:

    - `meta-llama/Llama-3.1-8B-Instruct`
    - `mistralai/Mistral-7B-Instruct-v0.2`

2. **Try Different Methods**:

    - Test various RAG methods from the dropdown
    - Compare their performance

3. **Build Your Own Corpus**:

    - Prepare your own JSONL corpus
    - Build index using Index Builder
    - Test with your domain-specific data

4. **Run Full Evaluations**:
    - Download FlashRAG datasets from HuggingFace
    - Run comprehensive evaluations

## Notes

-   Paths in the UI are **relative to the FlashRAG root directory** (`src/rag/FlashRAG/`)
-   Models will be downloaded from HuggingFace on first use
-   Large models require significant disk space and memory
-   GPU is recommended for better performance but not required for testing
