import os
import json
import sys
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Configuration
INPUT_FILE = "./datasets/test_dataset.jsonl"  # Original dataset for full assessment
OUTPUT_FILE = "./datasets/test_dataset_filtered.jsonl"  # High-quality pairs
BORDERLINE_FILE = "./datasets/test_dataset_borderline.jsonl"  # Borderline pairs
PROMOTION_THRESHOLD = 8.0  # Promote borderline pairs with score >= 8.0
HIGH_THRESHOLD = 8.5  # Auto-approve pairs scoring >= 8.5
LOW_THRESHOLD = 7.5   # Reject pairs scoring < 7.5, borderline between 7.5-8.5
MODEL_NAME = "gpt-4o-mini"  # OpenAI GPT-4o-mini model

# Quality Assessment Criteria
QUALITY_CRITERIA = """
Evaluate each QA pair on the following criteria (score 1-10 for each, then provide overall score):

1. **Question Quality (1-10)**:
   - Tests meaningful understanding of key concepts
   - Not trivial (e.g., not from acknowledgments/references)
   - Clear and well-formed
   - Appropriate difficulty level

2. **Answer Accuracy (1-10)**:
   - Factually correct based on context
   - Complete and informative
   - No mathematical errors
   - Directly supported by the provided context

3. **Relevance (1-10)**:
   - Question directly relates to the context
   - Answer addresses the question properly
   - Context provides sufficient information

4. **Educational Value (1-10)**:
   - Tests understanding, not just recall
   - Promotes deeper comprehension
   - Useful for learning/training

**Overall Quality Score**: Average of the 4 criteria (1-10 scale)
**Decision**: KEEP if score ≥ 7, REJECT if score < 7
"""

# ============================================================================
# PHASE 1: Rule-Based Pre-Filtering
# ============================================================================

def detect_issues(qa_pair: Dict[str, Any]) -> List[str]:
    """
    Detect common quality issues using rule-based checks.
    Returns list of detected issues (empty list = no issues).
    """
    issues = []
    question = qa_pair.get('question', '').lower()
    answer = qa_pair.get('answer', '')
    context = qa_pair.get('context', '')
    
    # 1. Trivial questions (from acknowledgments, references, metadata)
    trivial_keywords = [
        'title', 'author', 'date', 'funding', 'acknowledgment', 'acknowledgement',
        'affiliation', 'email', 'address', 'reference', 'citation', 'doi',
        'abstract', 'keywords', 'corresponding author', 'disclaimer'
    ]
    if any(kw in question for kw in trivial_keywords):
        issues.append('trivial_question')
    
    # 2. Incomplete answers (too short or truncated)
    if len(answer) < 30:
        issues.append('answer_too_short')
    if answer.rstrip().endswith('...') or answer.rstrip().endswith('..'):
        issues.append('answer_truncated')
    
    # 3. Math notation issues (common in LaTeX extraction)
    latex_indicators = ['\\frac', '\\sum', '\\int', '\\sqrt', '\\partial', '\\nabla']
    if any(indicator in answer for indicator in latex_indicators):
        # Check if it looks like broken LaTeX
        if '\\' in answer and ('{' not in answer or '}' not in answer):
            issues.append('possible_math_error')
    
    # 4. Questions about figures/tables without proper context
    figure_keywords = ['figure', 'table', 'fig.', 'tab.', 'plot', 'graph', 'diagram']
    if any(kw in question for kw in figure_keywords):
        # Check if context mentions the figure/table
        if not any(kw in context.lower() for kw in figure_keywords):
            issues.append('figure_without_context')
    
    # 5. Very short questions (likely incomplete)
    if len(question) < 20:
        issues.append('question_too_short')
    
    # 6. Question contains special characters indicating parsing issues
    if any(char in question for char in ['□', '■', '●', '▪', '\x00']):
        issues.append('parsing_error')
    
    return issues


def get_llm(model_name: str):
    """Initialize OpenAI LLM."""
    return ChatOpenAI(model=model_name, temperature=0.0)


def assess_qa_quality(llm, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality of a single QA pair using LLM judge."""
    
    prompt = f"""You are an expert evaluator of educational question-answer pairs for academic research papers.

{QUALITY_CRITERIA}

**QA Pair to Evaluate:**
Question: {qa_pair['question']}
Answer: {qa_pair['answer']}
Type: {qa_pair['type']}
Context (first 500 chars): {qa_pair['context'][:500]}

**Your Task:**
1. Score each criterion (1-10)
2. Calculate overall quality score (average)
3. Make a KEEP/REJECT decision
4. Provide brief justification

Output ONLY valid JSON format:
{{
  "question_quality": <1-10>,
  "answer_accuracy": <1-10>,
  "relevance": <1-10>,
  "educational_value": <1-10>,
  "overall_score": <average, 1-10>,
  "decision": "KEEP" or "REJECT",
  "justification": "<brief explanation>"
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON from response
        if '{' in content and '}' in content:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            assessment = json.loads(json_str)
            return assessment
    except Exception as e:
        print(f"    [Error] Assessment failed: {e}")
        return {
            "overall_score": 0,
            "decision": "REJECT",
            "justification": f"Assessment error: {e}"
        }
    
    return {
        "overall_score": 0,
        "decision": "REJECT",
        "justification": "Failed to parse assessment"
    }


def promote_borderline_pairs():
    """
    Promote borderline pairs with score >= PROMOTION_THRESHOLD to filtered dataset.
    This increases the training set count by moving high-scoring borderline pairs.
    """
    print("=" * 80)
    print("PROMOTING BORDERLINE PAIRS TO FILTERED DATASET")
    print("=" * 80)
    
    # Load borderline pairs
    if not os.path.exists(BORDERLINE_FILE):
        print(f"\n❌ Borderline file not found: {BORDERLINE_FILE}")
        return
    
    print(f"\n📂 Loading borderline pairs from {BORDERLINE_FILE}...")
    borderline_pairs = []
    with open(BORDERLINE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                borderline_pairs.append(json.loads(line))
    
    print(f"   Loaded {len(borderline_pairs)} borderline pairs")
    
    # Filter pairs with score >= PROMOTION_THRESHOLD
    promoted_pairs = []
    remaining_borderline_pairs = []
    
    for pair in borderline_pairs:
        score = pair.get('quality_assessment', {}).get('overall_score', 0)
        issues = pair.get('detected_issues', [])
        
        # Promote if score >= threshold and no detected issues
        if score >= PROMOTION_THRESHOLD and len(issues) == 0:
            promoted_pairs.append(pair)
        else:
            remaining_borderline_pairs.append(pair)
    
    if not promoted_pairs:
        print(f"\n✅ No borderline pairs met the promotion threshold of {PROMOTION_THRESHOLD:.1f}.")
        return
    
    # Load existing filtered pairs
    print(f"\n📂 Loading existing filtered pairs from {OUTPUT_FILE}...")
    filtered_pairs = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    filtered_pairs.append(json.loads(line))
        print(f"   Loaded {len(filtered_pairs)} existing filtered pairs")
    else:
        print(f"   No existing filtered file found, creating new one")
    
    # Create a set of existing pairs to avoid duplicates (based on question)
    existing_questions = {pair.get('question', '').lower().strip() for pair in filtered_pairs}
    
    # Add promoted pairs (avoiding duplicates)
    new_promoted = []
    duplicates = 0
    for pair in promoted_pairs:
        question_key = pair.get('question', '').lower().strip()
        if question_key not in existing_questions:
            filtered_pairs.append(pair)
            existing_questions.add(question_key)
            new_promoted.append(pair)
        else:
            duplicates += 1
    
    # Sort filtered pairs by score (highest first)
    filtered_pairs.sort(key=lambda x: x.get('quality_assessment', {}).get('overall_score', 0), reverse=True)
    remaining_borderline_pairs.sort(key=lambda x: x.get('quality_assessment', {}).get('overall_score', 0), reverse=True)
    
    # Save updated files
    print(f"\n💾 Saving updated files...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save updated filtered dataset
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pair in filtered_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Save updated borderline dataset (without promoted pairs)
    with open(BORDERLINE_FILE, 'w', encoding='utf-8') as f:
        for pair in remaining_borderline_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Calculate statistics
    avg_score_promoted = sum(p.get('quality_assessment', {}).get('overall_score', 0) for p in new_promoted) / len(new_promoted) if new_promoted else 0
    avg_score_filtered = sum(p.get('quality_assessment', {}).get('overall_score', 0) for p in filtered_pairs) / len(filtered_pairs) if filtered_pairs else 0
    
    print(f"\n✅ Promotion Complete!")
    print(f"=" * 80)
    print(f"   📊 Statistics:")
    print(f"      - Promoted: {len(new_promoted)} pairs (score >= {PROMOTION_THRESHOLD:.1f})")
    print(f"      - Duplicates skipped: {duplicates}")
    print(f"      - Average score of promoted: {avg_score_promoted:.2f}/10")
    print(f"      - Remaining in borderline: {len(remaining_borderline_pairs)}")
    print(f"      - Total in filtered dataset: {len(filtered_pairs)}")
    print(f"      - Average score of filtered dataset: {avg_score_filtered:.2f}/10")
    print(f"\n   📁 Files updated:")
    print(f"      - {OUTPUT_FILE} (filtered dataset)")
    print(f"      - {BORDERLINE_FILE} (borderline dataset)")


def filter_dataset():
    """Filter QA pairs based on rule-based checks and quality assessment."""
    
    print(f"Loading QA pairs from {INPUT_FILE}...")
    qa_pairs = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # ========================================================================
    # Phase 1: Rule-Based Pre-Filtering
    # ========================================================================
    print(f"\n📋 Phase 1: Rule-Based Pre-Filtering...")
    pre_filtered_pairs = []
    auto_rejected = []
    
    for qa_pair in qa_pairs:
        issues = detect_issues(qa_pair)
        qa_pair['detected_issues'] = issues
        
        if len(issues) > 0:
            qa_pair['quality_assessment'] = {
                'overall_score': 0,
                'decision': 'REJECT',
                'justification': f"Auto-rejected due to: {', '.join(issues)}"
            }
            auto_rejected.append(qa_pair)
        else:
            pre_filtered_pairs.append(qa_pair)
    
    print(f"   Pre-filter results:")
    print(f"   - Passed: {len(pre_filtered_pairs)}")
    print(f"   - Auto-rejected: {len(auto_rejected)}")
    
    # ========================================================================
    # Phase 2: GPT-4o-mini Assessment + Threshold-Based Selection
    # ========================================================================
    print(f"\n🤖 Phase 2: GPT-4o-mini Assessment...")
    print(f"Initializing OpenAI GPT-4o-mini model: {MODEL_NAME}...")
    llm = get_llm(MODEL_NAME)
    
    assessed_pairs = []
    for i, qa_pair in enumerate(tqdm(pre_filtered_pairs, desc="Assessing QA pairs")):
        assessment = assess_qa_quality(llm, qa_pair)
        qa_pair['quality_assessment'] = assessment
        assessed_pairs.append(qa_pair)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pre_filtered_pairs)}...")
    
    # ========================================================================
    # Phase 3: Categorize by Score Threshold
    # ========================================================================
    print(f"\n📊 Phase 3: Categorizing by Threshold...")
    print(f"   HIGH_THRESHOLD: {HIGH_THRESHOLD} (auto-approve)")
    print(f"   LOW_THRESHOLD: {LOW_THRESHOLD} (reject below)")
    
    high_quality_pairs = []  # Score >= HIGH_THRESHOLD (auto-approve)
    borderline_pairs = []    # Score between LOW_THRESHOLD and HIGH_THRESHOLD
    rejected_pairs = []      # Score < LOW_THRESHOLD
    
    for qa_pair in assessed_pairs:
        score = qa_pair['quality_assessment'].get('overall_score', 0)
        issues = qa_pair.get('detected_issues', [])
        
        # Must have no detected issues AND meet score threshold
        if len(issues) == 0 and score >= HIGH_THRESHOLD:
            high_quality_pairs.append(qa_pair)
        elif len(issues) == 0 and score >= LOW_THRESHOLD:
            borderline_pairs.append(qa_pair)
        else:
            rejected_pairs.append(qa_pair)
    
    # Add auto-rejected pairs to rejected list
    rejected_pairs.extend(auto_rejected)
    
    # Sort by score
    high_quality_pairs.sort(key=lambda x: x['quality_assessment']['overall_score'], reverse=True)
    borderline_pairs.sort(key=lambda x: x['quality_assessment']['overall_score'], reverse=True)
    rejected_pairs.sort(key=lambda x: x['quality_assessment']['overall_score'], reverse=True)
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\n💾 Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save high-quality pairs (auto-approved)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for qa_pair in high_quality_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    # Save borderline pairs for manual review
    with open(BORDERLINE_FILE, 'w', encoding='utf-8') as f:
        for qa_pair in borderline_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    # Save rejected pairs for analysis
    rejected_file = OUTPUT_FILE.replace('.jsonl', '_rejected.jsonl')
    with open(rejected_file, 'w', encoding='utf-8') as f:
        for qa_pair in rejected_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    # ========================================================================
    # Statistics
    # ========================================================================
    avg_score_high = sum(q['quality_assessment']['overall_score'] for q in high_quality_pairs) / len(high_quality_pairs) if high_quality_pairs else 0
    avg_score_borderline = sum(q['quality_assessment']['overall_score'] for q in borderline_pairs) / len(borderline_pairs) if borderline_pairs else 0
    avg_score_rejected = sum(q['quality_assessment']['overall_score'] for q in rejected_pairs if q['quality_assessment']['overall_score'] > 0) / max(1, len([q for q in rejected_pairs if q['quality_assessment']['overall_score'] > 0]))
    
    print(f"\n✅ Quality Assessment Complete!")
    print(f"=" * 60)
    print(f"   Original: {len(qa_pairs)} QA pairs")
    print(f"   Auto-rejected (rule-based): {len(auto_rejected)}")
    print(f"=" * 60)
    print(f"\n   📗 HIGH QUALITY (auto-approved, score >= {HIGH_THRESHOLD}):")
    print(f"      Count: {len(high_quality_pairs)}")
    print(f"      Average score: {avg_score_high:.2f}/10")
    print(f"      File: {OUTPUT_FILE}")
    print(f"\n   📙 BORDERLINE (needs manual review, {LOW_THRESHOLD} <= score < {HIGH_THRESHOLD}):")
    print(f"      Count: {len(borderline_pairs)}")
    print(f"      Average score: {avg_score_borderline:.2f}/10")
    print(f"      File: {BORDERLINE_FILE}")
    print(f"\n   📕 REJECTED (score < {LOW_THRESHOLD} or has issues):")
    print(f"      Count: {len(rejected_pairs)}")
    print(f"      Average score: {avg_score_rejected:.2f}/10")
    print(f"      File: {rejected_file}")
    print(f"\n   Next step: Manually review borderline pairs in {BORDERLINE_FILE}")


if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "promote":
        # Run promotion function
        promote_borderline_pairs()
    else:
        # Run full assessment
        filter_dataset()
