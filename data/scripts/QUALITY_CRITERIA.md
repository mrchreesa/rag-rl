# QA Pair Quality Assessment Criteria

## Overview

This document defines the criteria for assessing the quality of generated question-answer pairs for RAG-RL training.

## Scoring System (1-10 scale)

### 1. Question Quality (1-10)

**Excellent (9-10)**:

-   Tests deep understanding of key concepts
-   Requires synthesis or analysis
-   Clear, well-formed, unambiguous
-   Appropriate for academic/research context

**Good (7-8)**:

-   Tests meaningful understanding
-   Requires some reasoning
-   Generally clear and well-formed
-   Relevant to the content

**Fair (5-6)**:

-   Tests basic recall
-   Somewhat clear
-   May be too simple or vague

**Poor (1-4)**:

-   Trivial (e.g., "What is the title?")
-   From acknowledgments/references
-   Unclear or ambiguous
-   Not educational

### 2. Answer Accuracy (1-10)

**Excellent (9-10)**:

-   Factually correct
-   Complete and comprehensive
-   No mathematical/formula errors
-   Directly supported by context

**Good (7-8)**:

-   Mostly accurate
-   Generally complete
-   Minor issues acceptable
-   Supported by context

**Fair (5-6)**:

-   Some inaccuracies
-   Incomplete information
-   May have errors

**Poor (1-4)**:

-   Factually incorrect
-   Significant errors
-   Not supported by context
-   Mathematical/formula errors

### 3. Relevance (1-10)

**Excellent (9-10)**:

-   Question directly relates to context
-   Answer perfectly addresses question
-   Context provides all necessary information

**Good (7-8)**:

-   Question relates to context
-   Answer addresses question
-   Context provides most information

**Fair (5-6)**:

-   Somewhat related
-   Partial answer
-   Limited context

**Poor (1-4)**:

-   Not related to context
-   Answer doesn't address question
-   Insufficient context

### 4. Educational Value (1-10)

**Excellent (9-10)**:

-   Promotes deep understanding
-   Tests synthesis/analysis skills
-   Useful for learning complex concepts
-   High training value

**Good (7-8)**:

-   Promotes understanding
-   Tests comprehension
-   Useful for learning
-   Good training value

**Fair (5-6)**:

-   Basic learning value
-   Tests recall
-   Limited training value

**Poor (1-4)**:

-   Minimal learning value
-   Trivial recall
-   Low training value

## Overall Quality Score

**Calculation**: Average of 4 criteria scores

**Decision Thresholds**:

-   **KEEP**: Score ≥ 7.0 (High quality)
-   **REJECT**: Score < 7.0 (Low quality)

## Common Issues to Reject

1. **Trivial Questions**:

    - "What is the title of the paper?"
    - "Who are the authors?"
    - "What is the publication date?"

2. **Acknowledgment/Reference Questions**:

    - Questions about funding sources
    - Questions about citations
    - Questions about author affiliations

3. **Mathematical Errors**:

    - Incorrect formulas
    - Wrong numerical values
    - Formula notation errors

4. **Incomplete Answers**:

    - Missing key information
    - Partial explanations
    - Unfinished thoughts

5. **Poor Context Match**:
    - Question not answerable from context
    - Answer from different section
    - Insufficient information

## Model Recommendations

### Top Tier (Best Quality Assessment):

1.  **Claude 3.5 Sonnet** (Anthropic) - **Recommended for quality assessment**

    -   Excellent reasoning and thorough evaluation
    -   Best at identifying subtle quality issues
    -   Available via Anthropic API
    -   Model name: `claude-3-5-sonnet-20241022`

2.  **Claude 3 Opus** (Anthropic) - Highest quality alternative

    -   Most capable Claude model
    -   Best for complex assessments
    -   Model name: `claude-3-opus-20240229`

3.  **GPT-4o** (OpenAI) - Excellent alternative
    -   Strong reasoning capabilities
    -   Fast and reliable

### Using Claude Code via Terminal (Anthropic API)

**Setup Instructions**:

1.  **Install dependencies**:

    ```bash
    pip install langchain-anthropic anthropic
    ```

2.  **Set API key** (get from https://console.anthropic.com/):

    ```bash
    export ANTHROPIC_API_KEY="your-api-key-here"
    ```

    Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):

    ```bash
    echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
    source ~/.zshrc
    ```

3.  **Configure `assess_quality.py`**:

    ```python
    MODEL_TYPE = "anthropic"
    MODEL_NAME = "claude-3-5-sonnet-20241022"  # Recommended
    # Alternative: "claude-3-opus-20240229" for highest quality
    QUALITY_THRESHOLD = 7
    ```

4.  **Run assessment from terminal**:
    ```bash
    cd /Users/kreeza/Desktop/Programming/FinalYearProject-RAG-RL/data
    python assess_quality.py
    ```

**Important**: Claude Code does **NOT** need access to PDF documents. The QA pairs already contain all necessary information:

-   ✅ **Question**: The generated question
-   ✅ **Answer**: The generated answer
-   ✅ **Context**: First 500 characters from the source document (embedded in JSONL)
-   ✅ **Source**: PDF filename for reference

All information needed for quality assessment is already embedded in the `train_dataset.jsonl` file. Claude evaluates the QA pairs based on this embedded context, not by reading the original PDFs.

**Verify Setup**:

```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# Test Claude connection (optional)
python -c "from langchain_anthropic import ChatAnthropic; llm = ChatAnthropic(model='claude-3-5-sonnet-20241022'); print(llm.invoke('Say hello').content)"
```

**Cost Estimate for Claude 3.5 Sonnet**:

-   ~$0.015-0.03 per QA pair assessment
-   1000 pairs ≈ $15-30
-   200 test pairs ≈ $3-6

## Output

-   **Filtered dataset**: `train_dataset_filtered.jsonl` (high quality pairs)
-   **Rejected pairs**: `train_dataset_filtered_rejected.jsonl` (for analysis)
-   **Statistics**: Quality scores and rejection rates
