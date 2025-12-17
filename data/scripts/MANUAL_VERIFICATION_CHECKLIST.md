# Manual Quality Verification Checklist

## Overview

This checklist is used to manually verify the quality of QA pairs after automated assessment. Review 50 filtered pairs and 20 rejected pairs to validate the automated filtering system.

---

## Step 1: Review Filtered Pairs (50 pairs)

For each pair, check the following criteria:

### Question Quality

-   [ ] Tests understanding (not trivial)
-   [ ] Clear and well-formed
-   [ ] Not from acknowledgments/references

### Answer Accuracy

-   [ ] Factually correct
-   [ ] Complete (not truncated)
-   [ ] No mathematical errors
-   [ ] Supported by context

### Relevance

-   [ ] Question relates to context
-   [ ] Answer addresses the question
-   [ ] Context provides sufficient info

### Educational Value

-   [ ] Promotes learning
-   [ ] Useful for training

### Documentation

-   [ ] Mark pairs with problems
-   [ ] Note specific issues (e.g., "math error", "incomplete answer")
-   [ ] Track false positives (should have been rejected)

---

## Step 2: Review Rejected Pairs (20 pairs)

Check for false negatives:

-   [ ] Are any actually high quality?
-   [ ] Was the rejection justified?
-   [ ] Any patterns in rejections?

### Documentation

-   [ ] Pairs that should have been kept
-   [ ] Common rejection reasons
-   [ ] Whether the filter is too strict

---

## Review Template

### Pair #[NUMBER]

**Score**: **\_/10  
**Source**: ******\_********  
**Type**: factual / reasoning

**Question**:

---

---

**Answer**:

---

---

**Context**:

---

---

**Assessment**:

-   [ ] Good quality - Keep
-   [ ] Has issues - Note below
-   [ ] Should be rejected

**Issues found**:

-   [ ] Math error
-   [ ] Incomplete answer
-   [ ] Trivial question
-   [ ] Poor context match
-   [ ] Other: ******\_\_\_\_******

**Notes**:

---

---

---

## Summary Sheet

### Filtered Pairs Review (50 pairs)

**Good pairs**: **_/50 (_**%)  
**Issues found**:

-   Math errors: \_\_\_
-   Incomplete answers: \_\_\_
-   Trivial questions: \_\_\_
-   Poor context match: \_\_\_
-   Other: \_\_\_

**False positives** (should have been rejected): \_\_\_

### Rejected Pairs Review (20 pairs)

**False negatives** (should have been kept): \_\_\_/20  
**Common rejection reasons**:

-   Low question quality: \_\_\_
-   Answer accuracy issues: \_\_\_
-   Poor relevance: \_\_\_
-   Low educational value: \_\_\_
-   Other: \_\_\_

**Filter assessment**:

-   [ ] Filter is appropriate (good balance)
-   [ ] Filter is too strict (rejecting good pairs)
-   [ ] Filter is too lenient (keeping bad pairs)

---

## Recommendations

Based on review findings:

-   [ ] Proceed with filtered dataset as-is
-   [ ] Remove problematic pairs manually
-   [ ] Adjust quality criteria/threshold
-   [ ] Re-run assessment with different model
-   [ ] Other: ******\_\_\_\_******

**Notes**:

---

---

---

---

## Quick Reference: Quality Criteria

### Excellent (9-10)

-   Deep understanding, synthesis/analysis required
-   Factually correct, comprehensive
-   Perfect relevance and context match
-   High educational value

### Good (7-8)

-   Meaningful understanding, some reasoning
-   Mostly accurate, generally complete
-   Good relevance
-   Promotes learning

### Fair (5-6)

-   Basic recall
-   Some inaccuracies
-   Partial relevance
-   Limited training value

### Poor (1-4)

-   Trivial questions
-   Factually incorrect
-   Not related to context
-   Minimal learning value

---

**Review Date**: ******\_\_\_******  
**Reviewer**: ******\_\_\_******  
**Time Spent**: \_\_\_ minutes
