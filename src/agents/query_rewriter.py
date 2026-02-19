"""
RL-Based Query Rewriter for RAG

Implements query rewriting with reinforcement learning to improve
document retrieval relevance and answer quality.

Based on RaFe (EMNLP 2024) approach: train a query rewriter using
F1 score of final answer as the reward signal.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import random

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from .reward import compute_f1


class QueryRewriter:
    """
    Base query rewriter using LLM prompting.
    
    Can be extended with RL training for optimized rewriting.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_ollama: bool = False,
        num_rewrites: int = 3,
        temperature: float = 0.7
    ):
        """
        Initialize the query rewriter.
        
        Args:
            model: OpenAI model name
            use_ollama: Whether to use local Ollama
            num_rewrites: Number of rewrite candidates to generate during training
            temperature: Sampling temperature for diversity
        """
        self.model = model
        self.use_ollama = use_ollama
        self.num_rewrites = num_rewrites
        self.temperature = temperature
        
        # Initialize OpenAI client
        if use_ollama:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            self.model = "llama3.1:8b-instruct-q4_K_M"
        else:
            from openai import OpenAI
            self.client = OpenAI()
        
        # Rewrite prompt template
        self.rewrite_prompt = """You are a query optimization expert. Rewrite the following question to improve document retrieval for answering it.

Original Question: {question}

Guidelines for rewriting:
1. Expand abbreviations and acronyms
2. Add relevant domain-specific terms
3. Make implicit requirements explicit
4. Break compound questions into key components
5. Include synonyms for key concepts

Provide ONLY the rewritten query, nothing else."""

        # Statistics tracking
        self.stats = {
            "total_rewrites": 0,
            "avg_length_increase": 0.0,
            "training_samples": 0
        }
    
    def rewrite(self, question: str, deterministic: bool = True) -> str:
        """
        Rewrite a question to improve retrieval.
        
        Args:
            question: Original question
            deterministic: If True, use temperature=0 for consistent results
            
        Returns:
            Rewritten question
        """
        messages = [
            {"role": "system", "content": self.rewrite_prompt.format(question=question)},
            {"role": "user", "content": "Rewrite the query:"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=256,
                temperature=0.0 if deterministic else self.temperature
            )
            rewritten = response.choices[0].message.content.strip()
            
            # Track statistics
            self.stats["total_rewrites"] += 1
            len_increase = len(rewritten) - len(question)
            self.stats["avg_length_increase"] = (
                (self.stats["avg_length_increase"] * (self.stats["total_rewrites"] - 1) + len_increase)
                / self.stats["total_rewrites"]
            )
            
            return rewritten
            
        except Exception as e:
            print(f"Query rewrite failed: {e}")
            return question  # Fallback to original
    
    def generate_candidates(self, question: str) -> List[str]:
        """
        Generate multiple rewrite candidates for training.
        
        During RL training, we sample multiple rewrites and
        use the one that leads to the best answer.
        
        Args:
            question: Original question
            
        Returns:
            List of rewritten question candidates
        """
        candidates = [question]  # Always include original
        
        for _ in range(self.num_rewrites - 1):
            rewritten = self.rewrite(question, deterministic=False)
            if rewritten not in candidates:
                candidates.append(rewritten)
        
        return candidates


class RLQueryRewriter(QueryRewriter):
    """
    Query rewriter with reinforcement learning training.
    
    Learns to rewrite queries by maximizing F1 score of answers
    generated using retrieved documents.
    """
    
    def __init__(
        self,
        retriever=None,
        generator=None,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """
        Initialize RL query rewriter.
        
        Args:
            retriever: Document retriever instance
            generator: Answer generator instance
            learning_rate: Learning rate for policy updates
            **kwargs: Passed to QueryRewriter
        """
        super().__init__(**kwargs)
        
        self.retriever = retriever
        self.generator = generator
        self.learning_rate = learning_rate
        
        # Training history for analysis
        self.training_history: List[Dict[str, Any]] = []
        
        # Best rewrites discovered during training
        self.rewrite_cache: Dict[str, str] = {}
    
    def train_step(
        self,
        question: str,
        golden_answers: List[str],
        topk: int = 5
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Perform one RL training step.
        
        1. Generate candidate rewrites
        2. Retrieve documents for each candidate
        3. Generate answers for each
        4. Compute F1 scores
        5. Select best rewrite (for future use)
        
        Args:
            question: Original question
            golden_answers: Ground truth answers
            topk: Number of documents to retrieve
            
        Returns:
            Tuple of (best_reward, metrics_dict)
        """
        if self.retriever is None or self.generator is None:
            raise ValueError("Retriever and generator must be set for training")
        
        # Generate candidates
        candidates = self.generate_candidates(question)
        
        results = []
        for candidate in candidates:
            # Retrieve documents
            docs = self.retriever.retrieve([candidate], topk=topk)[0]
            
            # Generate answer
            answer = self.generator.generate_with_retrieval(question, docs)
            
            # Compute F1 reward
            f1_score = compute_f1(answer, golden_answers)
            
            results.append({
                "rewrite": candidate,
                "answer": answer,
                "f1": f1_score,
                "is_original": candidate == question
            })
        
        # Find best rewrite
        best_result = max(results, key=lambda x: x["f1"])
        
        # Cache the best rewrite if it's better than original
        original_result = next(r for r in results if r["is_original"])
        if best_result["f1"] > original_result["f1"]:
            self.rewrite_cache[question] = best_result["rewrite"]
        
        # Training metrics
        metrics = {
            "best_f1": best_result["f1"],
            "original_f1": original_result["f1"],
            "improvement": best_result["f1"] - original_result["f1"],
            "best_rewrite": best_result["rewrite"],
            "num_candidates": len(candidates),
            "best_was_original": best_result["is_original"]
        }
        
        self.training_history.append(metrics)
        self.stats["training_samples"] += 1
        
        return best_result["f1"], metrics
    
    def rewrite(self, question: str, deterministic: bool = True) -> str:
        """
        Rewrite using cached best rewrites or generate new.
        
        During inference, uses cached rewrites from training if available.
        """
        # Check cache first
        if question in self.rewrite_cache:
            return self.rewrite_cache[question]
        
        # Generate new rewrite
        return super().rewrite(question, deterministic)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        """
        if not self.training_history:
            return {"status": "no training data"}
        
        improvements = [h["improvement"] for h in self.training_history]
        best_was_original = [h["best_was_original"] for h in self.training_history]
        
        return {
            "total_samples": len(self.training_history),
            "avg_improvement": sum(improvements) / len(improvements),
            "max_improvement": max(improvements),
            "pct_rewrite_helped": 1 - (sum(best_was_original) / len(best_was_original)),
            "cache_size": len(self.rewrite_cache)
        }


class AdaptiveQueryRewriter(RLQueryRewriter):
    """
    Adaptive query rewriter that learns WHEN to rewrite.
    
    Some questions don't benefit from rewriting. This class
    learns to skip rewriting when the original is sufficient.
    """
    
    def __init__(self, rewrite_threshold: float = 0.1, **kwargs):
        """
        Initialize adaptive rewriter.
        
        Args:
            rewrite_threshold: Minimum expected improvement to trigger rewrite
            **kwargs: Passed to RLQueryRewriter
        """
        super().__init__(**kwargs)
        self.rewrite_threshold = rewrite_threshold
        
        # Track which question types benefit from rewriting
        self.question_type_stats: Dict[str, Dict[str, float]] = {}
    
    def should_rewrite(self, question: str) -> bool:
        """
        Decide whether to rewrite this question.
        
        Based on learned patterns of which questions benefit.
        """
        # Simple heuristics for now, can be replaced with learned classifier
        
        # Short questions often benefit from expansion
        if len(question.split()) < 5:
            return True
        
        # Questions with technical terms may need clarification
        technical_indicators = ["algorithm", "method", "technique", "process"]
        if any(term in question.lower() for term in technical_indicators):
            return True
        
        # Questions with pronouns may need context
        if any(word in question.lower().split() for word in ["it", "this", "that", "they"]):
            return True
        
        return False
    
    def rewrite(self, question: str, deterministic: bool = True) -> str:
        """
        Conditionally rewrite based on learned policy.
        """
        if not self.should_rewrite(question):
            return question
        
        return super().rewrite(question, deterministic)


class StrategyRewriter:
    """
    Strategy-based query rewriter for RL-trained query reformulation.

    Instead of a free-form LLM rewrite, selects from 5 discrete strategies:
      0: original     - Use question as-is (free)
      1: expand        - Add domain terms, synonyms, expand abbreviations
      2: decompose     - Extract core retrievable fact as focused query
      3: contextualize - Add implicit domain context
      4: simplify      - Strip to essential keywords

    The RL policy network learns which strategy works best per question.
    Only strategies 1-4 require an LLM call; strategy 0 is free.
    """

    STRATEGY_NAMES = ["original", "expand", "decompose", "contextualize", "simplify"]

    STRATEGY_PROMPTS = {
        "expand": (
            "You are a search query optimizer. Expand the following question by adding "
            "relevant domain-specific terms, synonyms, and expanding any abbreviations "
            "or acronyms. The goal is to improve document retrieval.\n\n"
            "Original question: {question}\n\n"
            "Provide ONLY the expanded query, nothing else."
        ),
        "decompose": (
            "You are a search query optimizer. Decompose the following question into its "
            "core retrievable fact. Extract the key entity and relationship that a search "
            "engine needs to find. Remove filler words and focus on what matters.\n\n"
            "Original question: {question}\n\n"
            "Provide ONLY the focused query, nothing else."
        ),
        "contextualize": (
            "You are a search query optimizer. Add implicit domain context to the following "
            "question. If the question is about a technical topic, add the field name. If it "
            "references something specific, add clarifying context that would help retrieval.\n\n"
            "Original question: {question}\n\n"
            "Provide ONLY the contextualized query, nothing else."
        ),
        "simplify": (
            "You are a search query optimizer. Simplify the following question to its "
            "essential keywords. Remove all question words, articles, and filler. Keep only "
            "the core terms that matter for finding relevant documents.\n\n"
            "Original question: {question}\n\n"
            "Provide ONLY the simplified keywords, nothing else."
        ),
    }

    def __init__(self, use_ollama: bool = False, model: str = "gpt-4o-mini"):
        """
        Initialize strategy rewriter.

        Args:
            use_ollama: Use local Ollama instead of OpenAI
            model: Model name for rewriting LLM calls
        """
        self.use_ollama = use_ollama
        self.model = model

        if use_ollama:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            self.model = "llama3.1:8b-instruct-q4_K_M"
        else:
            from openai import OpenAI
            self.client = OpenAI()

        self.stats = {strategy: 0 for strategy in self.STRATEGY_NAMES}

    def rewrite(self, question: str, strategy_idx: int) -> str:
        """
        Rewrite a question using the given strategy.

        Args:
            question: Original question
            strategy_idx: Strategy index (0-4)

        Returns:
            Rewritten question (or original if strategy_idx == 0)
        """
        strategy_name = self.STRATEGY_NAMES[strategy_idx]
        self.stats[strategy_name] += 1

        if strategy_idx == 0:
            return question

        prompt_template = self.STRATEGY_PROMPTS[strategy_name]
        prompt = prompt_template.format(question=question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else question
        except Exception as e:
            print(f"Strategy rewrite failed ({strategy_name}): {e}")
            return question


# Factory function
def create_query_rewriter(
    rl_enabled: bool = False,
    adaptive: bool = False,
    **kwargs
) -> QueryRewriter:
    """
    Factory function to create query rewriter.
    
    Args:
        rl_enabled: Whether to use RL training
        adaptive: Whether to use adaptive (conditional) rewriting
        **kwargs: Additional arguments
        
    Returns:
        QueryRewriter instance
    """
    if adaptive:
        return AdaptiveQueryRewriter(**kwargs)
    elif rl_enabled:
        return RLQueryRewriter(**kwargs)
    else:
        return QueryRewriter(**kwargs)
