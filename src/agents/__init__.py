"""
RL-RAG Agents Module

This module contains the integration between FlashRAG and Agent Lightning
for training RL-based retrieval decision agents.

Components:
- RLRAGAgent: Main LitAgent implementation wrapping FlashRAG
- LearnedJudgerAgent: Extended agent with learned retrieval decisions
- RAGRewardCalculator: Reward function for RL training
- DenseRetrieverWrapper: E5 + FAISS retriever wrapper
- GeneratorWrapper: OpenAI/Ollama generator wrapper
- RAGPipeline: Complete RAG pipeline with controllable retrieval
- Dataset utilities: HotpotQA and custom dataset loaders
"""

# Import components in order of dependency
# Dataset and reward have no external dependencies
from .dataset import (
    RAGDataset,
    load_hotpotqa,
    load_custom_dataset,
    load_combined_dataset,
    dataset_stats
)

from .reward import RAGRewardCalculator, compute_rag_reward, compute_f1, compute_exact_match

# FlashRAG components - requires FlashRAG but not Agent Lightning
try:
    from .flashrag_components import DenseRetrieverWrapper, GeneratorWrapper, RAGPipeline
    FLASHRAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FlashRAG components not available: {e}")
    FLASHRAG_AVAILABLE = False
    DenseRetrieverWrapper = None
    GeneratorWrapper = None  
    RAGPipeline = None

# Agent classes - requires Agent Lightning
try:
    from .rl_rag_agent import RLRAGAgent, LearnedJudgerAgent, create_rl_rag_agent
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RL-RAG Agent not available: {e}")
    AGENT_AVAILABLE = False
    RLRAGAgent = None
    LearnedJudgerAgent = None
    create_rl_rag_agent = None

# Query Rewriter - RL-based query optimization
try:
    from .query_rewriter import (
        QueryRewriter,
        RLQueryRewriter,
        AdaptiveQueryRewriter,
        StrategyRewriter,
        create_query_rewriter
    )
    QUERY_REWRITER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Query Rewriter not available: {e}")
    QUERY_REWRITER_AVAILABLE = False
    QueryRewriter = None
    RLQueryRewriter = None
    AdaptiveQueryRewriter = None
    StrategyRewriter = None
    create_query_rewriter = None

# Enhanced RL Pipeline - requires PyTorch
try:
    from .enhanced_pipeline import (
        EnhancedRAGPipeline,
        RetrievalPolicyNetwork,
        DynamicTopKPolicyNetwork,
        QueryRewritePolicyNetwork,
        RLTrainer,
        train_rl_rag
    )
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced Pipeline not available: {e}")
    ENHANCED_PIPELINE_AVAILABLE = False
    EnhancedRAGPipeline = None
    RetrievalPolicyNetwork = None
    DynamicTopKPolicyNetwork = None
    QueryRewritePolicyNetwork = None
    RLTrainer = None
    train_rl_rag = None


__all__ = [
    # Agents
    "RLRAGAgent",
    "LearnedJudgerAgent",
    "create_rl_rag_agent",
    # Reward
    "RAGRewardCalculator",
    "compute_rag_reward",
    "compute_f1",
    "compute_exact_match",
    # Components
    "DenseRetrieverWrapper",
    "GeneratorWrapper",
    "RAGPipeline",
    # Query Rewriter
    "QueryRewriter",
    "RLQueryRewriter",
    "AdaptiveQueryRewriter",
    "StrategyRewriter",
    "create_query_rewriter",
    # Enhanced Pipeline
    "EnhancedRAGPipeline",
    "RetrievalPolicyNetwork",
    "DynamicTopKPolicyNetwork",
    "QueryRewritePolicyNetwork",
    "RLTrainer",
    "train_rl_rag",
    # Dataset
    "RAGDataset",
    "load_hotpotqa",
    "load_custom_dataset",
    "load_combined_dataset",
    "dataset_stats",
    # Availability flags
    "FLASHRAG_AVAILABLE",
    "AGENT_AVAILABLE",
    "QUERY_REWRITER_AVAILABLE",
    "ENHANCED_PIPELINE_AVAILABLE",
]

