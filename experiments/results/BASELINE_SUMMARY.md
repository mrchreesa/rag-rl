# Baseline Experiments Summary

## Overview

This document summarizes the baseline RAG experiments conducted on the custom academic/research Q&A dataset. These results establish the performance targets for AgentLightning RL optimization.

## Dataset

- **Test Set**: 87 samples
- **Corpus**: 41,717 documents (combined custom corpus)
- **Domain**: Academic/research Q&A

## Retrieval Methods Tested

| Method | Description |
|--------|-------------|
| **BM25** | Sparse keyword-based retrieval |
| **Dense E5** | Semantic search using `intfloat/e5-base-v2` embeddings + FAISS |

## Generation Models Tested

| Model | Description |
|-------|-------------|
| **Ollama (llama3.1:8b-instruct-q4_K_M)** | Local quantized LLaMA 3.1 |
| **GPT-4o-mini** | OpenAI's efficient model |

## Results Summary

### All Experiments

| Method | EM | F1 | Recall@k | Retrieval | Generator |
|--------|----:|---:|--------:|-----------|-----------|
| Dense E5 topk=10 | 3.45% | **31.10%** | **87.36%** | E5 | GPT-4o-mini |
| Dense E5 topk=5 | 3.45% | 30.82% | 83.91% | E5 | GPT-4o-mini |
| Dense E5 topk=3 | **4.60%** | 30.08% | 78.16% | E5 | GPT-4o-mini |
| BM25 topk=5 (OpenAI) | 2.30% | 27.49% | 75.86% | BM25 | GPT-4o-mini |
| BM25 topk=5 (Ollama) | 2.30% | 14.40% | 75.86% | BM25 | llama3.1 |
| BM25 Iter-RetGen | 2.30% | 12.96% | 72.41% | BM25 | llama3.1 |
| BM25 Naive RAG | 2.30% | 9.62% | 75.86% | BM25 | llama3.1 |
| BM25 topk=10 (Ollama) | 0.00% | 3.43% | 80.46% | BM25 | llama3.1 |
| BM25 Adaptive-RAG | 0.00% | 12.71% | 8.05% | BM25 | llama3.1 |

### Best Configuration

**Dense E5 + GPT-4o-mini (topk=10)**

```yaml
retrieval:
  method: dense
  model: intfloat/e5-base-v2
  index: FAISS Flat
  topk: 10

generation:
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 256
```

**Metrics:**
- **Exact Match**: 3.45%
- **F1 Score**: 31.10%
- **Retrieval Recall@10**: 87.36%

## Key Findings

### 1. Dense Retrieval Outperforms BM25

| Metric | BM25 (best) | Dense E5 (best) | Improvement |
|--------|------------:|----------------:|------------:|
| Recall@5 | 75.86% | 83.91% | **+8.05%** |
| Recall@10 | 80.46% | 87.36% | **+6.90%** |

Dense retrieval provides significantly better recall due to semantic understanding of academic terminology.

### 2. Generator Quality Matters More Than Retrieval Method

With the same BM25 retrieval (75.86% recall):
- Ollama llama3.1: 14.40% F1
- GPT-4o-mini: **27.49% F1** (+13% improvement)

The generator's ability to synthesize information from retrieved documents is the primary bottleneck.

### 3. TopK Ablation Results

| topk | Recall | F1 | EM |
|-----:|-------:|---:|---:|
| 3 | 78.16% | 30.08% | **4.60%** |
| 5 | 83.91% | 30.82% | 3.45% |
| 10 | **87.36%** | **31.10%** | 3.45% |

- Higher topk improves recall and F1
- Lower topk (3) achieves highest EM (possibly due to less noise)
- Optimal topk depends on task requirements

### 4. Advanced RAG Techniques Limited by Local LLM

- **Iter-RetGen**: No improvement over Naive RAG with Ollama
- **Adaptive-RAG**: Degraded performance (8.05% recall)
- These methods may perform better with more capable models

## Optimization Targets for AgentLightning

Based on these baselines, AgentLightning should aim to:

1. **Primary Target**: Exceed Dense E5 + GPT-4o-mini performance
   - Beat **31.10% F1** (current best)
   - Maintain **>80% retrieval recall**

2. **Optimization Opportunities**:
   - Learn optimal topk selection per query
   - Implement adaptive retrieval strategies
   - Query reformulation for difficult questions
   - Multi-hop reasoning for complex academic questions

3. **Efficiency Target**: Match GPT-4o-mini quality with local models
   - Current gap: 16.7% F1 between local and OpenAI models
   - RL could optimize prompting strategies for local LLMs

## Files

- **Comparison CSV**: `experiments/results/baselines/comparison_summary.csv`
- **Individual Results**: `experiments/results/baselines/*/`
- **Dense Retrieval Script**: `experiments/scripts/baselines/run_dense_custom.py`
- **E5 Index**: `data/indexes/custom_e5/e5_Flat.index`

## Next Steps

1. Configure AgentLightning with Dense E5 retrieval
2. Define RL reward based on F1 improvement over baseline
3. Train retrieval/generation policies to optimize answer quality
4. Compare learned policies against fixed baselines

