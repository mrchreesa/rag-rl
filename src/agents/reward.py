"""
Reward Functions for RL-RAG Training

Implements reward calculation based on:
- Answer quality (F1 score, Exact Match)
- Retrieval efficiency (penalize unnecessary retrievals)
- Format compliance bonus
"""

import re
import string
from collections import Counter
from typing import List, Optional, Tuple


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison.
    
    Removes articles, punctuation, extra whitespace, and lowercases.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute token-level F1 score between prediction and ground truths.
    
    Returns the maximum F1 across all ground truth answers.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    max_f1 = 0.0
    normalized_prediction = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        
        # Handle yes/no/noanswer special cases
        if normalized_prediction in ["yes", "no", "noanswer"] and \
           normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and \
           normalized_prediction != normalized_ground_truth:
            continue
        
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            continue
        
        precision = num_same / len(prediction_tokens) if prediction_tokens else 0
        recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute exact match score.
    
    Returns 1.0 if normalized prediction matches any ground truth, 0.0 otherwise.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 or 0.0
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    normalized_prediction = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        if normalized_prediction == normalize_answer(ground_truth):
            return 1.0
    
    return 0.0


class RAGRewardCalculator:
    """
    Reward calculator for RL-RAG agent training.

    Implements the reward function:
    - Base reward from answer quality (F1 or EM)
    - Retrieval cost penalty (supports both fixed and per-document cost)
    - Efficiency bonus for correct answers without retrieval
    - CRITICAL: Wrong non-retrieval penalty to prevent "lazy agent" problem
    """

    def __init__(
        self,
        retrieval_cost: float = 0.1,
        correct_no_retrieval_bonus: float = 0.1,
        wrong_no_retrieval_penalty: float = 0.3,  # Penalize bad direct answers
        use_f1: bool = True,
        f1_threshold_for_correct: float = 0.5,
        f1_threshold_for_bad: float = 0.3,  # Threshold for "bad" answers
        format_bonus: float = 0.05,
        # Dynamic TopK parameters
        use_dynamic_cost: bool = False,
        base_retrieval_cost: float = 0.05,  # Fixed cost for any retrieval
        per_doc_cost: float = 0.01  # Additional cost per document retrieved
    ):
        """
        Initialize the reward calculator.

        Args:
            retrieval_cost: Penalty for each retrieval action (default 0.1)
                           Used when use_dynamic_cost=False
            correct_no_retrieval_bonus: Bonus for correct answer without retrieval (default 0.1)
            wrong_no_retrieval_penalty: Penalty for bad answer without retrieval (default 0.3)
                                        This MUST be > retrieval_cost to prevent lazy agent!
            use_f1: Use F1 score (True) or Exact Match (False) for quality
            f1_threshold_for_correct: F1 threshold to consider answer "correct" (default 0.5)
            f1_threshold_for_bad: F1 threshold below which answer is "bad" (default 0.3)
            format_bonus: Small bonus for proper answer format
            use_dynamic_cost: If True, use base_retrieval_cost + per_doc_cost * topk
            base_retrieval_cost: Fixed cost for any retrieval (only if use_dynamic_cost=True)
            per_doc_cost: Cost per document retrieved (only if use_dynamic_cost=True)
        """
        self.retrieval_cost = retrieval_cost
        self.correct_no_retrieval_bonus = correct_no_retrieval_bonus
        self.wrong_no_retrieval_penalty = wrong_no_retrieval_penalty
        self.use_f1 = use_f1
        self.f1_threshold_for_correct = f1_threshold_for_correct
        self.f1_threshold_for_bad = f1_threshold_for_bad
        self.format_bonus = format_bonus
        # Dynamic TopK cost parameters
        self.use_dynamic_cost = use_dynamic_cost
        self.base_retrieval_cost = base_retrieval_cost
        self.per_doc_cost = per_doc_cost

    def compute_retrieval_cost(self, topk_used: int) -> float:
        """
        Compute the retrieval cost based on number of documents.

        Args:
            topk_used: Number of documents retrieved (0 = no retrieval)

        Returns:
            Cost value
        """
        if topk_used == 0:
            return 0.0

        if self.use_dynamic_cost:
            # Dynamic cost: base + per-doc
            return self.base_retrieval_cost + self.per_doc_cost * topk_used
        else:
            # Fixed cost per retrieval operation
            return self.retrieval_cost

    def compute_reward(
        self,
        prediction: str,
        ground_truths: List[str],
        did_retrieve: bool,
        num_retrievals: int = 1,
        topk_used: int = 0
    ) -> Tuple[float, dict]:
        """
        Compute the reward for a single prediction.

        Reward formula:
        - If retrieve: quality_score - retrieval_cost(topk)
        - If no retrieve and correct: quality_score + efficiency_bonus
        - If no retrieve and WRONG: quality_score - wrong_no_retrieval_penalty

        Args:
            prediction: Model's predicted answer
            ground_truths: List of acceptable ground truth answers
            did_retrieve: Whether retrieval was performed
            num_retrievals: Number of retrieval calls made (legacy, for backward compat)
            topk_used: Number of documents retrieved (for dynamic topk)

        Returns:
            Tuple of (reward, metrics_dict)
        """
        # Calculate quality scores
        f1_score = compute_f1(prediction, ground_truths)
        em_score = compute_exact_match(prediction, ground_truths)

        # Determine quality score to use
        quality_score = f1_score if self.use_f1 else em_score

        # Determine if answer is "correct" or "bad"
        is_correct = f1_score >= self.f1_threshold_for_correct or em_score == 1.0
        is_bad = f1_score < self.f1_threshold_for_bad and em_score == 0.0

        # Calculate reward components
        reward = 0.0
        wrong_no_retrieval_applied = 0.0
        retrieval_penalty = 0.0

        # Base quality reward
        reward += quality_score

        # Retrieval cost/bonus
        if did_retrieve:
            # Calculate retrieval cost based on topk
            if topk_used > 0:
                retrieval_penalty = self.compute_retrieval_cost(topk_used)
            else:
                # Fallback: use legacy num_retrievals
                retrieval_penalty = self.retrieval_cost * num_retrievals
            reward -= retrieval_penalty
        else:
            # No retrieval case - this is where lazy agent problem occurs
            if is_correct:
                # Good! Correctly answered without retrieval
                reward += self.correct_no_retrieval_bonus
            elif is_bad:
                # BAD! Failed to answer without retrieval - should have retrieved!
                reward -= self.wrong_no_retrieval_penalty
                wrong_no_retrieval_applied = self.wrong_no_retrieval_penalty

        # Format bonus (small reward for non-empty, reasonable answers)
        if prediction and len(prediction.strip()) > 0 and len(prediction) < 500:
            reward += self.format_bonus

        metrics = {
            "f1": f1_score,
            "em": em_score,
            "quality_score": quality_score,
            "is_correct": is_correct,
            "is_bad": is_bad,
            "did_retrieve": did_retrieve,
            "topk_used": topk_used,
            "num_retrievals": num_retrievals,
            "retrieval_penalty": retrieval_penalty,
            "efficiency_bonus": self.correct_no_retrieval_bonus if (not did_retrieve and is_correct) else 0,
            "wrong_no_retrieval_penalty": wrong_no_retrieval_applied,
            "raw_reward": reward
        }

        return reward, metrics


def compute_rag_reward(
    prediction: str,
    ground_truths: List[str],
    did_retrieve: bool,
    retrieval_cost: float = 0.1
) -> float:
    """
    Simple reward function for quick calculations.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable ground truth answers  
        did_retrieve: Whether retrieval was performed
        retrieval_cost: Cost penalty for retrieval
        
    Returns:
        Scalar reward value
    """
    f1 = compute_f1(prediction, ground_truths)
    
    if did_retrieve:
        return f1 - retrieval_cost
    else:
        # Bonus for correct without retrieval
        bonus = 0.1 if f1 >= 0.5 else 0.0
        return f1 + bonus


# Reward function signatures compatible with Agent Lightning
def rag_reward_function(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    did_retrieve: Optional[bool] = None,
    **kwargs
) -> float:
    """
    Agent Lightning compatible reward function.
    
    Args:
        solution_str: The model's answer
        ground_truth: The expected answer(s) - can be string or list
        did_retrieve: Whether retrieval was performed
        
    Returns:
        Scalar reward value
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    
    # Default to assuming retrieval happened if not specified
    if did_retrieve is None:
        did_retrieve = True
    
    return compute_rag_reward(solution_str, ground_truth, did_retrieve)

