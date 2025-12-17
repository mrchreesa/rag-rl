# QA Pair Quality Assessment Guide

## Quick Start

### Option 1: Using OpenAI GPT-4o (Recommended - Best Quality)

1. **Install dependencies**:

```bash
pip install langchain-openai openai
```

2. **Set API key**:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Configure script**:

```python
MODEL_TYPE = "openai"
MODEL_NAME = "gpt-4o"  # or "gpt-4-turbo"
QUALITY_THRESHOLD = 7  # Keep pairs with score ≥ 7/10
```

4. **Run**:

```bash
python assess_quality.py
```

### Option 2: Using Anthropic Claude 3.5 Sonnet

1. **Install dependencies**:

```bash
pip install langchain-anthropic anthropic
```

2. **Set API key**:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

3. **Configure script**:

```python
MODEL_TYPE = "anthropic"
MODEL_NAME = "claude-3-5-sonnet-20241022"
QUALITY_THRESHOLD = 7
```

### Option 3: Using Local Ollama (Llama 3.1 70B)

1. **Pull model**:

```bash
ollama pull llama3.1:70b
```

2. **Configure script**:

```python
MODEL_TYPE = "ollama"
MODEL_NAME = "llama3.1:70b"
QUALITY_THRESHOLD = 7
```

## Quality Assessment Prompt

The script uses a comprehensive prompt that evaluates QA pairs on 4 criteria:

1. **Question Quality** (1-10): Tests meaningful understanding, not trivial
2. **Answer Accuracy** (1-10): Factually correct, no errors
3. **Relevance** (1-10): Question relates to context, answer addresses question
4. **Educational Value** (1-10): Promotes learning, useful for training

**Overall Score**: Average of 4 criteria

**Decision**: KEEP if score ≥ threshold, REJECT otherwise

## What Gets Filtered Out

The assessment automatically rejects:

-   ❌ Trivial questions (titles, authors, dates)
-   ❌ Questions from acknowledgments/references
-   ❌ Mathematical/formula errors
-   ❌ Incomplete or inaccurate answers
-   ❌ Questions not answerable from context

## Output Files

-   **`train_dataset_filtered.jsonl`**: High-quality QA pairs (score ≥ threshold)
-   **`train_dataset_filtered_rejected.jsonl`**: Low-quality pairs (for analysis)

Each QA pair includes quality assessment metadata:

```json
{
	"question": "...",
	"answer": "...",
	"type": "factual",
	"source": "...",
	"context": "...",
	"quality_assessment": {
		"question_quality": 8,
		"answer_accuracy": 9,
		"relevance": 8,
		"educational_value": 7,
		"overall_score": 8.0,
		"decision": "KEEP",
		"justification": "..."
	}
}
```

## Expected Results

Based on initial quality assessment (70% good, 30% issues):

-   **Expected retention**: ~60-70% of original pairs
-   **Quality improvement**: Average score increases from ~6.5 to ~8.0+
-   **Dataset size**: ~600-700 high-quality pairs (from 1000)

## Cost Estimates (API-based models)

### GPT-4o:

-   ~$0.01-0.02 per QA pair assessment
-   1000 pairs ≈ $10-20

### Claude 3.5 Sonnet:

-   ~$0.015-0.03 per QA pair assessment
-   1000 pairs ≈ $15-30

### GPT-4 Turbo (cheaper alternative):

-   ~$0.005-0.01 per QA pair assessment
-   1000 pairs ≈ $5-10

## Tips

1. **Start with a sample**: Test on 50-100 pairs first
2. **Adjust threshold**: Lower (6.5) for more pairs, higher (7.5) for stricter filtering
3. **Review rejected pairs**: Check `_rejected.jsonl` to refine criteria
4. **Batch processing**: Script processes all pairs automatically
5. **Progress tracking**: Updates every 10 pairs processed

## Troubleshooting

**Import errors**: Install missing packages:

```bash
pip install langchain-openai langchain-anthropic
```

**API errors**: Check API keys are set correctly:

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

**Ollama errors**: Ensure model is pulled:

```bash
ollama list
ollama pull llama3.1:70b
```
