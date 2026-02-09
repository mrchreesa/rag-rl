"""
RL-RAG Agent Implementation

Wraps FlashRAG components inside Agent Lightning's LitAgent framework.
The agent learns to decide when to retrieve vs. generate directly.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING

# Add Agent Lightning to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
AGENT_LIGHTNING_PATH = PROJECT_ROOT / 'src/rl/agent-lightning'
sys.path.insert(0, str(AGENT_LIGHTNING_PATH))

# Try to import Agent Lightning components, with fallback for missing dependencies
try:
    from agentlightning import LitAgent, NamedResources, Rollout, emit_reward
    AGENT_LIGHTNING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent Lightning not fully available: {e}")
    print("Running in standalone mode without Agent Lightning trainer support.")
    AGENT_LIGHTNING_AVAILABLE = False
    
    # Create placeholder classes for standalone operation
    from typing import Generic, TypeVar
    T = TypeVar('T')
    
    class LitAgent(Generic[T]):
        """Placeholder LitAgent for standalone operation."""
        def __init__(self, **kwargs):
            pass
    
    class NamedResources(dict):
        """Placeholder for NamedResources."""
        pass
    
    class Rollout:
        """Placeholder for Rollout."""
        pass
    
    def emit_reward(reward: float):
        """Placeholder emit_reward."""
        pass

from .flashrag_components import DenseRetrieverWrapper, GeneratorWrapper, RAGPipeline
from .reward import RAGRewardCalculator, compute_f1


class RLRAGAgent(LitAgent[Dict[str, Any]]):
    """
    RL Agent for Retrieval-Augmented Generation.
    
    This agent wraps FlashRAG's Dense E5 retriever and OpenAI generator,
    and learns to make binary retrieval decisions:
    - RETRIEVE: Use the retriever to get documents, then generate with context
    - GENERATE_DIRECTLY: Answer from the LLM's parametric knowledge
    
    The goal is to learn when retrieval is necessary (improving efficiency)
    while maintaining answer quality.
    """
    
    def __init__(
        self,
        trained_agents: Optional[str] = None,
        retriever: Optional[DenseRetrieverWrapper] = None,
        generator: Optional[GeneratorWrapper] = None,
        topk: int = 5,
        retrieval_cost: float = 0.1,
        use_ollama: bool = False
    ):
        """
        Initialize the RL-RAG agent.
        
        Args:
            trained_agents: Agent Lightning trained agents identifier
            retriever: Pre-initialized retriever (created if None)
            generator: Pre-initialized generator (created if None) 
            topk: Number of documents to retrieve
            retrieval_cost: Cost penalty for retrieval actions
            use_ollama: Use local Ollama instead of OpenAI API
        """
        super().__init__(trained_agents=trained_agents)
        
        self.topk = topk
        self.use_ollama = use_ollama
        self._retriever = retriever
        self._generator = generator
        
        # Initialize reward calculator
        self.reward_calculator = RAGRewardCalculator(
            retrieval_cost=retrieval_cost,
            correct_no_retrieval_bonus=0.1,
            use_f1=True,
            f1_threshold_for_correct=0.5
        )
        
        # Lazy initialization flags
        self._initialized = False
        self._pipeline: Optional[RAGPipeline] = None
        
        # Statistics tracking
        self.stats = {
            "total_questions": 0,
            "retrieval_decisions": 0,
            "direct_decisions": 0,
            "correct_with_retrieval": 0,
            "correct_without_retrieval": 0,
            "total_reward": 0.0
        }
    
    def _ensure_initialized(self):
        """Lazy initialization of heavy components."""
        if not self._initialized:
            print("Initializing RL-RAG Agent components...")
            
            if self._retriever is None:
                self._retriever = DenseRetrieverWrapper()
            
            if self._generator is None:
                self._generator = GeneratorWrapper(use_ollama=self.use_ollama)
            
            self._pipeline = RAGPipeline(
                retriever=self._retriever,
                generator=self._generator,
                topk=self.topk
            )
            
            self._initialized = True
            print("RL-RAG Agent initialized successfully!")
    
    def decide_retrieval(self, question: str, resources: NamedResources) -> bool:
        """
        Decide whether to retrieve for this question.
        
        This is the key decision point that the RL agent learns.
        In the initial version, we use a simple heuristic.
        During training, this decision comes from the policy network.
        
        Args:
            question: The input question
            resources: Agent Lightning resources (may contain LLM for decision)
            
        Returns:
            True if should retrieve, False otherwise
        """
        # During initial implementation, always retrieve
        # This will be replaced by the learned policy during training
        # 
        # The Agent Lightning framework handles this through the
        # LLM proxy and policy gradient updates
        return True
    
    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> float:
        """
        Execute a single rollout (question-answer episode).
        
        This is the main entry point called by Agent Lightning trainer.
        
        Args:
            task: Dictionary containing:
                - question: The question to answer
                - golden_answers: List of acceptable answers
                - id: Task identifier
            resources: Named resources from trainer
            rollout: Rollout metadata
            
        Returns:
            Reward value for this rollout
        """
        self._ensure_initialized()
        
        # Extract task data
        question = task.get("question", "")
        golden_answers = task.get("golden_answers", [])
        task_id = task.get("id", "unknown")
        
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        # Make retrieval decision
        should_retrieve = self.decide_retrieval(question, resources)
        
        # Execute pipeline
        try:
            answer, docs = self._pipeline.answer(question, should_retrieve=should_retrieve)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            answer = ""
            docs = None
        
        # Calculate reward
        reward, metrics = self.reward_calculator.compute_reward(
            prediction=answer,
            ground_truths=golden_answers,
            did_retrieve=should_retrieve,
            num_retrievals=1 if should_retrieve else 0
        )
        
        # Update statistics
        self.stats["total_questions"] += 1
        self.stats["total_reward"] += reward
        if should_retrieve:
            self.stats["retrieval_decisions"] += 1
            if metrics["is_correct"]:
                self.stats["correct_with_retrieval"] += 1
        else:
            self.stats["direct_decisions"] += 1
            if metrics["is_correct"]:
                self.stats["correct_without_retrieval"] += 1
        
        # Emit reward for Agent Lightning training
        emit_reward(reward)
        
        return reward
    
    async def rollout_async(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> float:
        """
        Async version of rollout for Agent Lightning async training.
        
        Currently wraps the sync version, but can be optimized for
        concurrent API calls.
        """
        return self.rollout(task, resources, rollout)
    
    def training_rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> float:
        """Training rollout - same as regular rollout."""
        return self.rollout(task, resources, rollout)
    
    def validation_rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout
    ) -> float:
        """
        Validation rollout - uses greedy decoding / deterministic policy.
        
        During validation, we want deterministic behavior to measure
        true performance without exploration noise.
        """
        return self.rollout(task, resources, rollout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = dict(self.stats)
        
        # Add derived metrics
        total = stats["total_questions"]
        if total > 0:
            stats["retrieval_rate"] = stats["retrieval_decisions"] / total
            stats["avg_reward"] = stats["total_reward"] / total
            
            if stats["retrieval_decisions"] > 0:
                stats["accuracy_with_retrieval"] = (
                    stats["correct_with_retrieval"] / stats["retrieval_decisions"]
                )
            else:
                stats["accuracy_with_retrieval"] = 0.0
            
            if stats["direct_decisions"] > 0:
                stats["accuracy_without_retrieval"] = (
                    stats["correct_without_retrieval"] / stats["direct_decisions"]
                )
            else:
                stats["accuracy_without_retrieval"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_questions": 0,
            "retrieval_decisions": 0,
            "direct_decisions": 0,
            "correct_with_retrieval": 0,
            "correct_without_retrieval": 0,
            "total_reward": 0.0
        }


class LearnedJudgerAgent(RLRAGAgent):
    """
    RL-RAG Agent with a learned retrieval judger.
    
    This extends the base RLRAGAgent with an LLM-based decision maker
    that learns when to retrieve through RL training.
    """
    
    def __init__(
        self,
        judger_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the learned judger agent.
        
        Args:
            judger_prompt: Custom prompt for the retrieval decision LLM
            **kwargs: Passed to RLRAGAgent
        """
        super().__init__(**kwargs)
        
        self.judger_prompt = judger_prompt or self._default_judger_prompt()
    
    def _default_judger_prompt(self) -> str:
        """Default prompt for retrieval decision."""
        return """You are a retrieval decision agent. Given a question, decide if you need to search external documents to answer it correctly.

Consider:
- If the question asks about specific facts, dates, names, or recent events -> RETRIEVE
- If the question is about common knowledge or simple reasoning -> GENERATE_DIRECTLY
- If you're uncertain about the answer -> RETRIEVE

Question: {question}

Respond with exactly one word: RETRIEVE or GENERATE_DIRECTLY"""
    
    def decide_retrieval(self, question: str, resources: NamedResources) -> bool:
        """
        Use LLM to decide whether to retrieve.
        
        This decision is what gets optimized through RL training.
        The Agent Lightning framework will update the LLM's behavior
        based on the rewards received.
        """
        # During training, Agent Lightning manages the LLM calls
        # and policy updates. For inference, we use direct prompting.
        
        # Check if we have an LLM resource from training
        llm = resources.get("decision_llm") if resources else None
        
        if llm is not None:
            # Use the provided LLM (from Agent Lightning training)
            # This is where the learned policy comes from
            prompt = self.judger_prompt.format(question=question)
            # The actual call would be handled by Agent Lightning's LLM proxy
            # For now, default to retrieve
            return True
        
        # Fallback: Use a simple heuristic for non-training scenarios
        # This will be replaced by the learned policy
        return self._heuristic_decision(question)
    
    def _heuristic_decision(self, question: str) -> bool:
        """
        Simple heuristic for retrieval decision (fallback).
        
        Used when no trained policy is available.
        """
        # Questions with specific indicators that suggest direct generation
        direct_indicators = [
            "what is the capital of",
            "who is the president of",
            "what year was",
            "simple math",
            "calculate",
            "2+2",
            "define "
        ]
        
        question_lower = question.lower()
        for indicator in direct_indicators:
            if indicator in question_lower:
                return False
        
        # Default to retrieval for safety
        return True


# Factory function for easy instantiation
def create_rl_rag_agent(
    use_learned_judger: bool = False,
    **kwargs
) -> RLRAGAgent:
    """
    Factory function to create an RL-RAG agent.
    
    Args:
        use_learned_judger: Whether to use learned retrieval decision
        **kwargs: Additional arguments for the agent
        
    Returns:
        Configured RLRAGAgent instance
    """
    if use_learned_judger:
        return LearnedJudgerAgent(**kwargs)
    return RLRAGAgent(**kwargs)

