"""
Enhanced RL-RAG Pipeline

Extends the base RAG pipeline with:
1. Query rewriting for improved retrieval
2. Neural policy network for retrieval decisions
3. RL training loop with REINFORCE

This implements the recommended approaches from the research:
- Approach 1: Policy Gradient for Retrieval Decision
- Approach 6: Query Rewriting (RaFe method)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import random
import json
from datetime import datetime

from .flashrag_components import DenseRetrieverWrapper, GeneratorWrapper, RAGPipeline
from .reward import RAGRewardCalculator, compute_f1, compute_exact_match
from .query_rewriter import QueryRewriter, RLQueryRewriter, create_query_rewriter, StrategyRewriter
import re


def extract_question_features(question: str) -> List[float]:
    """
    Extract difficulty features from a question.

    SOLUTION 5: These features help the policy network understand
    what type of question it's dealing with.

    Features:
    1. Question length (normalized)
    2. Word count (normalized)
    3. Question type encoding (who/what/when/where/why/how)
    4. Contains specific indicator (years, names, etc.)
    5. Complexity score (number of clauses, entities)

    Args:
        question: Input question string

    Returns:
        List of 5 normalized feature values (0-1)
    """
    features = []

    # 1. Question length (normalized by typical max length ~200 chars)
    length_normalized = min(len(question) / 200.0, 1.0)
    features.append(length_normalized)

    # 2. Word count (normalized by typical max ~30 words)
    word_count = len(question.split())
    word_count_normalized = min(word_count / 30.0, 1.0)
    features.append(word_count_normalized)

    # 3. Question type encoding
    # These are questions that often need retrieval
    question_lower = question.lower()
    retrieval_likely_types = ['who', 'what', 'when', 'where', 'which', 'how many', 'how much']
    retrieval_unlikely_types = ['why', 'how does', 'how do', 'explain']

    needs_retrieval_score = 0.5  # Default uncertain
    for pattern in retrieval_likely_types:
        if question_lower.startswith(pattern) or f' {pattern} ' in question_lower:
            needs_retrieval_score = 0.8
            break
    for pattern in retrieval_unlikely_types:
        if question_lower.startswith(pattern):
            needs_retrieval_score = 0.3
            break
    features.append(needs_retrieval_score)

    # 4. Contains specific facts indicators (years, proper nouns, etc.)
    has_year = bool(re.search(r'\b(19|20)\d{2}\b', question))
    has_numbers = bool(re.search(r'\b\d+\b', question))
    # Rough heuristic for proper nouns (capitalized words not at start)
    words = question.split()
    proper_nouns = sum(1 for w in words[1:] if w and w[0].isupper())
    specificity_score = min((int(has_year) * 0.3 + int(has_numbers) * 0.2 +
                            proper_nouns * 0.1), 1.0)
    features.append(specificity_score)

    # 5. Complexity score (punctuation, conjunctions)
    complexity_indicators = [',', ';', ' and ', ' or ', ' but ', ' because ', ' if ']
    complexity_count = sum(question.count(ind) for ind in complexity_indicators)
    complexity_score = min(complexity_count / 5.0, 1.0)
    features.append(complexity_score)

    return features


class RetrievalPolicyNetwork(nn.Module):
    """
    Neural network for deciding whether to retrieve.

    Input: Question embedding (from E5 encoder) + optional difficulty features
    Output: Probability of retrieval

    UPDATED: Added temperature-based soft sampling and entropy computation
    to prevent the "lazy agent" problem.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        use_difficulty_features: bool = False,
        num_difficulty_features: int = 5
    ):
        """
        Initialize policy network.

        Args:
            input_dim: Dimension of input embeddings (768 for E5-base)
            hidden_dim: Hidden layer dimension
            use_difficulty_features: Whether to use additional question features
            num_difficulty_features: Number of difficulty features (if used)
        """
        super().__init__()

        self.use_difficulty_features = use_difficulty_features
        self.num_difficulty_features = num_difficulty_features

        # Adjust input dimension if using difficulty features
        actual_input_dim = input_dim
        if use_difficulty_features:
            actual_input_dim = input_dim + num_difficulty_features

        self.network = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning retrieval probability."""
        return self.network(x)

    def get_entropy(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the binary policy distribution.

        Args:
            prob: Probability of retrieval (0-1)

        Returns:
            Entropy value (max is ln(2) ≈ 0.693 for uniform distribution)
        """
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        prob_clamped = torch.clamp(prob, 1e-8, 1 - 1e-8)
        entropy = -(prob_clamped * torch.log(prob_clamped) +
                    (1 - prob_clamped) * torch.log(1 - prob_clamped))
        return entropy

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy with temperature-based soft sampling.

        Args:
            x: Input embedding
            deterministic: If True, use greedy action selection
            temperature: Temperature for soft sampling

        Returns:
            Tuple of (should_retrieve, log_prob, entropy)
        """
        prob = self.forward(x).squeeze()  # Ensure scalar output

        # Compute entropy for regularization
        entropy = self.get_entropy(prob)

        if deterministic:
            if temperature < 1.0:
                logit = torch.log(prob / (1 - prob + 1e-8) + 1e-8)
                soft_prob = torch.sigmoid(logit / temperature)
                action = soft_prob > 0.5
            else:
                action = prob > 0.5
        else:
            if temperature != 1.0:
                logit = torch.log(prob / (1 - prob + 1e-8) + 1e-8)
                prob_temp = torch.sigmoid(logit / temperature)
                action = torch.bernoulli(prob_temp)
            else:
                action = torch.bernoulli(prob)

        # Compute log probability for REINFORCE
        prob_clamped = torch.clamp(prob, 1e-8, 1 - 1e-8)
        action_bool = bool(action.item())
        if action_bool:
            log_prob = torch.log(prob_clamped)
        else:
            log_prob = torch.log(1 - prob_clamped)

        return action_bool, log_prob, entropy


class DynamicTopKPolicyNetwork(nn.Module):
    """
    Neural network for deciding optimal number of documents to retrieve.

    Instead of binary retrieve/no-retrieve, learns to select from discrete
    topk options: [0, 1, 3, 5, 7, 10] where 0 = no retrieval.

    This allows the policy to learn:
    - Simple factual questions -> low topk (1-3)
    - Complex reasoning questions -> high topk (7-10)
    - Questions outside corpus -> topk=0 (no retrieval)
    """

    # Default topk options: 0 = no retrieval
    DEFAULT_TOPK_OPTIONS = [0, 1, 3, 5, 7, 10]

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        topk_options: Optional[List[int]] = None,
        use_difficulty_features: bool = False,
        num_difficulty_features: int = 5
    ):
        """
        Initialize dynamic topk policy network.

        Args:
            input_dim: Dimension of input embeddings (768 for E5-base)
            hidden_dim: Hidden layer dimension
            topk_options: List of possible topk values (default: [0, 1, 3, 5, 7, 10])
            use_difficulty_features: Whether to use additional question features
            num_difficulty_features: Number of difficulty features (if used)
        """
        super().__init__()

        self.topk_options = topk_options or self.DEFAULT_TOPK_OPTIONS
        self.num_actions = len(self.topk_options)
        self.use_difficulty_features = use_difficulty_features
        self.num_difficulty_features = num_difficulty_features

        # Adjust input dimension if using difficulty features
        actual_input_dim = input_dim
        if use_difficulty_features:
            actual_input_dim = input_dim + num_difficulty_features

        self.network = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.num_actions)  # Output logits for each topk option
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability distribution over topk options."""
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before softmax."""
        return self.network(x)

    def get_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the categorical policy distribution.

        Higher entropy = more uncertainty/exploration
        Lower entropy = more deterministic policy

        Args:
            probs: Probability distribution over topk options

        Returns:
            Entropy value (max is ln(num_actions) for uniform distribution)
        """
        # Clamp to avoid log(0)
        probs_clamped = torch.clamp(probs, 1e-8, 1.0)

        # Handle unbatched case (1D tensor)
        if probs.dim() == 1:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum()
        else:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
        return entropy

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample topk value from policy with temperature-based sampling.

        Args:
            x: Input embedding
            deterministic: If True, use greedy action selection (argmax)
            temperature: Temperature for sampling (lower = more deterministic)
                        Default 1.0 = standard sampling
                        Only used when deterministic=False

        Returns:
            Tuple of (topk_value, log_prob, entropy)
        """
        probs = self.forward(x)

        # Apply temperature if not deterministic
        if not deterministic and temperature != 1.0:
            logits = self.get_logits(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

        # Compute entropy for regularization
        entropy = self.get_entropy(probs)

        # Handle unbatched case - multinomial requires 2D input
        is_unbatched = probs.dim() == 1
        if is_unbatched:
            probs_for_sampling = probs.unsqueeze(0)
        else:
            probs_for_sampling = probs

        if deterministic:
            # Greedy: select highest probability action
            if temperature != 1.0:
                logits = self.get_logits(x)
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                probs_for_sampling = probs.unsqueeze(0) if is_unbatched else probs
            action_idx = probs_for_sampling.argmax(dim=-1)
        else:
            # Stochastic: sample from distribution
            action_idx = torch.multinomial(probs_for_sampling, 1).squeeze(-1)

        # Extract action index and compute log prob
        if is_unbatched:
            action_idx_item = action_idx.squeeze().item()
            topk = self.topk_options[action_idx_item]
            log_prob = torch.log(probs[action_idx_item] + 1e-8)
        else:
            action_idx_item = action_idx[0].item()
            topk = self.topk_options[action_idx_item]
            log_prob = torch.log(probs[0, action_idx_item] + 1e-8)

        return topk, log_prob, entropy

    def get_action_distribution(self, x: torch.Tensor) -> Dict[int, float]:
        """
        Get the full probability distribution over topk options.

        Useful for logging and analysis.

        Args:
            x: Input embedding

        Returns:
            Dictionary mapping topk values to probabilities
        """
        probs = self.forward(x)
        if probs.dim() > 1:
            probs = probs[0]
        return {k: probs[i].item() for i, k in enumerate(self.topk_options)}


class QueryRewritePolicyNetwork(nn.Module):
    """
    Neural network for selecting the best query rewriting strategy per question.

    Input: Question embedding (from E5 encoder) + optional difficulty features
    Output: Categorical distribution over 5 strategies:
        0: original, 1: expand, 2: decompose, 3: contextualize, 4: simplify

    Smaller than DynamicTopKPolicyNetwork since the action space is smaller
    and the task is simpler (selecting a rewrite template vs. optimizing k).
    """

    STRATEGY_NAMES = StrategyRewriter.STRATEGY_NAMES
    NUM_STRATEGIES = len(STRATEGY_NAMES)

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        use_difficulty_features: bool = False,
        num_difficulty_features: int = 5
    ):
        super().__init__()

        self.use_difficulty_features = use_difficulty_features
        self.num_difficulty_features = num_difficulty_features

        actual_input_dim = input_dim
        if use_difficulty_features:
            actual_input_dim = input_dim + num_difficulty_features

        self.network = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.NUM_STRATEGIES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability distribution over strategies."""
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before softmax."""
        return self.network(x)

    def get_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the categorical policy distribution."""
        probs_clamped = torch.clamp(probs, 1e-8, 1.0)
        if probs.dim() == 1:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum()
        else:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
        return entropy

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample rewrite strategy from policy.

        Returns:
            Tuple of (strategy_idx, log_prob, entropy)
        """
        probs = self.forward(x)

        if not deterministic and temperature != 1.0:
            logits = self.get_logits(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

        entropy = self.get_entropy(probs)

        is_unbatched = probs.dim() == 1
        if is_unbatched:
            probs_for_sampling = probs.unsqueeze(0)
        else:
            probs_for_sampling = probs

        if deterministic:
            if temperature != 1.0:
                logits = self.get_logits(x)
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                probs_for_sampling = probs.unsqueeze(0) if is_unbatched else probs
            action_idx = probs_for_sampling.argmax(dim=-1)
        else:
            action_idx = torch.multinomial(probs_for_sampling, 1).squeeze(-1)

        if is_unbatched:
            action_idx_item = action_idx.squeeze().item()
            log_prob = torch.log(probs[action_idx_item] + 1e-8)
        else:
            action_idx_item = action_idx[0].item()
            log_prob = torch.log(probs[0, action_idx_item] + 1e-8)

        return action_idx_item, log_prob, entropy

    def get_action_distribution(self, x: torch.Tensor) -> Dict[str, float]:
        """Get probability distribution over strategies."""
        probs = self.forward(x)
        if probs.dim() > 1:
            probs = probs[0]
        return {name: probs[i].item() for i, name in enumerate(self.STRATEGY_NAMES)}


class EnhancedRAGPipeline(RAGPipeline):
    """
    RAG Pipeline with query rewriting and learned retrieval.

    Extends base RAGPipeline with:
    1. Optional query rewriting before retrieval
    2. Neural policy for retrieval decisions (binary or dynamic topk)
    3. Question difficulty features for better policy decisions
    4. RL-trained query rewriting strategy selection
    """

    def __init__(
        self,
        retriever: Optional[DenseRetrieverWrapper] = None,
        generator: Optional[GeneratorWrapper] = None,
        topk: int = 5,
        use_query_rewriter: bool = True,
        use_learned_retrieval: bool = True,
        use_ollama: bool = False,
        generator_model: str = "gpt-4o-mini",
        use_difficulty_features: bool = False,  # Enable question difficulty features
        use_dynamic_topk: bool = False,  # Enable dynamic topk policy
        topk_options: Optional[List[int]] = None,  # Custom topk options
        use_learned_rewrite: bool = False  # Enable RL-trained query rewriting
    ):
        """
        Initialize enhanced RAG pipeline.

        Args:
            retriever: Dense retriever wrapper (default: E5-base + FAISS)
            generator: Generator wrapper (default: GPT-4o-mini)
            topk: Default number of documents to retrieve (used when policy not active)
            use_query_rewriter: Enable query rewriting
            use_learned_retrieval: Enable learned retrieval policy
            use_ollama: Use Ollama instead of OpenAI
            generator_model: Model name for generator (e.g. 'gpt-4o-mini', 'claude-3-5-haiku-20241022')
            use_difficulty_features: If True, policy uses additional question
                                     features (length, type, complexity)
            use_dynamic_topk: If True, use DynamicTopKPolicyNetwork instead of binary
            topk_options: Custom topk options for dynamic topk (default: [0,1,3,5,7,10])
            use_learned_rewrite: If True, use RL-trained query rewriting strategy selection
        """
        # Initialize components if not provided
        retriever = retriever or DenseRetrieverWrapper()
        generator = generator or GeneratorWrapper(model=generator_model, use_ollama=use_ollama)

        super().__init__(retriever, generator, topk)

        self.use_query_rewriter = use_query_rewriter
        self.use_learned_retrieval = use_learned_retrieval
        self.use_difficulty_features = use_difficulty_features
        self.use_dynamic_topk = use_dynamic_topk
        self.use_learned_rewrite = use_learned_rewrite
        self.topk_options = topk_options or DynamicTopKPolicyNetwork.DEFAULT_TOPK_OPTIONS

        # Initialize query rewriter (legacy prompt-based)
        if use_query_rewriter and not use_learned_rewrite:
            self.query_rewriter = create_query_rewriter(
                rl_enabled=True,
                retriever=self.retriever,
                generator=self.generator,
                use_ollama=use_ollama
            )
        else:
            self.query_rewriter = None

        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        # Initialize retrieval policy network
        if use_learned_retrieval:
            # Choose policy network type
            if use_dynamic_topk:
                self.policy_network = DynamicTopKPolicyNetwork(
                    topk_options=self.topk_options,
                    use_difficulty_features=use_difficulty_features,
                    num_difficulty_features=5
                )
                print(f"   Policy Network: DynamicTopK ({str(device).upper()})")
                print(f"   TopK Options: {self.topk_options}")
            else:
                self.policy_network = RetrievalPolicyNetwork(
                    use_difficulty_features=use_difficulty_features,
                    num_difficulty_features=5
                )
                print(f"   Policy Network: Binary ({str(device).upper()})")

            self.policy_network.to(device)

            self.policy_optimizer = torch.optim.Adam(
                self.policy_network.parameters(),
                lr=1e-3
            )
        else:
            self.policy_network = None
            self.policy_optimizer = None

        # Initialize learned rewrite policy
        if use_learned_rewrite:
            self.rewrite_policy_network = QueryRewritePolicyNetwork(
                use_difficulty_features=use_difficulty_features,
                num_difficulty_features=5
            )
            self.rewrite_policy_network.to(device)

            self.rewrite_optimizer = torch.optim.Adam(
                self.rewrite_policy_network.parameters(),
                lr=1e-3
            )

            self.strategy_rewriter = StrategyRewriter(
                use_ollama=use_ollama,
                model=generator_model
            )
            print(f"   Rewrite Policy: Learned Strategy ({str(device).upper()})")
            print(f"   Strategies: {QueryRewritePolicyNetwork.STRATEGY_NAMES}")
        else:
            self.rewrite_policy_network = None
            self.rewrite_optimizer = None
            self.strategy_rewriter = None

        # Training buffer for policy gradient (topk policy)
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        self.episode_entropies: List[torch.Tensor] = []

        # Training buffer for rewrite policy
        self.rewrite_log_probs: List[torch.Tensor] = []
        self.rewrite_rewards: List[float] = []
        self.rewrite_entropies: List[torch.Tensor] = []
    
    def get_question_embedding(self, question: str) -> torch.Tensor:
        """
        Get question embedding from E5 encoder, optionally with difficulty features.

        SOLUTION 5: If use_difficulty_features is enabled, concatenates
        the E5 embedding with hand-crafted question features.
        """
        embedding = self.retriever.model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        tensor = torch.tensor(embedding, dtype=torch.float32)

        # SOLUTION 5: Add difficulty features if enabled
        if self.use_difficulty_features:
            features = extract_question_features(question)
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            tensor = torch.cat([tensor, feature_tensor])

        # Move to same device as policy network if available
        if hasattr(self, 'device') and self.device is not None:
            tensor = tensor.to(self.device)
        return tensor
    
    def decide_retrieval(
        self,
        question: str,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decide retrieval strategy using learned policy.

        For binary policy: returns (should_retrieve: bool, log_prob, entropy)
        For dynamic topk: returns (topk: int, log_prob, entropy)

        Args:
            question: Input question
            deterministic: Use greedy policy (for evaluation)
            temperature: Temperature for soft sampling (lower = more deterministic)

        Returns:
            Tuple of (decision, log_prob, entropy)
            - For binary: decision is bool (should_retrieve)
            - For dynamic topk: decision is int (topk value, 0 = no retrieval)
        """
        if not self.use_learned_retrieval or self.policy_network is None:
            # Default behavior
            if self.use_dynamic_topk:
                return self.topk, None, None  # Use default topk
            else:
                return True, None, None  # Always retrieve

        embedding = self.get_question_embedding(question)
        decision, log_prob, entropy = self.policy_network.get_action(
            embedding, deterministic=deterministic, temperature=temperature
        )

        return decision, log_prob, entropy

    def get_topk_distribution(self, question: str) -> Optional[Dict[int, float]]:
        """
        Get the probability distribution over topk options (dynamic topk only).

        Args:
            question: Input question

        Returns:
            Dictionary mapping topk values to probabilities, or None if not dynamic topk
        """
        if not self.use_dynamic_topk or not isinstance(self.policy_network, DynamicTopKPolicyNetwork):
            return None

        embedding = self.get_question_embedding(question)
        return self.policy_network.get_action_distribution(embedding)
    
    def decide_rewrite(
        self,
        question: str,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decide query rewriting strategy using learned policy.

        Args:
            question: Input question
            deterministic: Use greedy policy (for evaluation)
            temperature: Temperature for sampling

        Returns:
            Tuple of (strategy_idx, log_prob, entropy)
        """
        if not self.use_learned_rewrite or self.rewrite_policy_network is None:
            return 0, None, None  # original strategy

        embedding = self.get_question_embedding(question)
        strategy_idx, log_prob, entropy = self.rewrite_policy_network.get_action(
            embedding, deterministic=deterministic, temperature=temperature
        )
        return strategy_idx, log_prob, entropy

    def answer(
        self,
        question: str,
        should_retrieve: Optional[bool] = None,
        topk_override: Optional[int] = None,
        use_rewrite: Optional[bool] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        rewrite_strategy_override: Optional[int] = None
    ) -> Tuple[str, Optional[List[Dict]], Dict[str, Any]]:
        """
        Answer a question with optional query rewriting and learned retrieval.

        Args:
            question: Input question
            should_retrieve: Override retrieval decision (None = use policy)
                            Ignored if use_dynamic_topk=True
            topk_override: Override topk value (for dynamic topk mode)
            use_rewrite: Override rewrite decision
            deterministic: Use deterministic policy
            temperature: Temperature for soft sampling during evaluation
            rewrite_strategy_override: Override rewrite strategy (for exploration)

        Returns:
            Tuple of (answer, docs, metadata)
        """
        metadata = {
            "original_question": question,
            "rewritten_question": None,
            "did_retrieve": False,
            "topk_used": 0,
            "retrieval_probability": None,
            "topk_distribution": None,
            "log_prob": None,
            "entropy": None,
            "rewrite_strategy": None,
            "rewrite_strategy_name": None,
            "rewrite_log_prob": None,
            "rewrite_entropy": None
        }

        # --- Step 1: Decide rewrite strategy (if learned rewrite enabled) ---
        rewrite_log_prob = None
        rewrite_entropy = None
        retrieval_query = question

        if self.use_learned_rewrite and self.strategy_rewriter is not None:
            if rewrite_strategy_override is not None:
                strategy_idx = rewrite_strategy_override
                rewrite_log_prob = None
                rewrite_entropy = None
            else:
                strategy_idx, rewrite_log_prob, rewrite_entropy = self.decide_rewrite(
                    question, deterministic, temperature
                )

            retrieval_query = self.strategy_rewriter.rewrite(question, strategy_idx)
            metadata["rewrite_strategy"] = strategy_idx
            metadata["rewrite_strategy_name"] = StrategyRewriter.STRATEGY_NAMES[strategy_idx]
            metadata["rewrite_log_prob"] = rewrite_log_prob
            metadata["rewrite_entropy"] = rewrite_entropy.item() if isinstance(rewrite_entropy, torch.Tensor) else rewrite_entropy
            if retrieval_query != question:
                metadata["rewritten_question"] = retrieval_query

        # --- Step 2: Decide retrieval strategy ---
        log_prob = None
        entropy = None
        topk_to_use = 0

        if self.use_dynamic_topk:
            # Dynamic TopK mode
            if topk_override is not None:
                topk_to_use = topk_override
            else:
                decision, log_prob, entropy = self.decide_retrieval(
                    question, deterministic, temperature
                )
                topk_to_use = decision

                if isinstance(self.policy_network, DynamicTopKPolicyNetwork):
                    metadata["topk_distribution"] = self.get_topk_distribution(question)

            should_retrieve = topk_to_use > 0
            metadata["topk_used"] = topk_to_use
            metadata["entropy"] = entropy.item() if entropy is not None else None
        else:
            # Binary mode (legacy)
            if should_retrieve is None:
                should_retrieve, log_prob, entropy = self.decide_retrieval(
                    question, deterministic, temperature
                )
                if self.policy_network is not None:
                    embedding = self.get_question_embedding(question)
                    prob = self.policy_network(embedding).item()
                    metadata["retrieval_probability"] = prob
                    metadata["entropy"] = entropy.item() if entropy is not None else None

            if should_retrieve:
                topk_to_use = self.topk
                metadata["topk_used"] = topk_to_use

        metadata["did_retrieve"] = should_retrieve
        metadata["log_prob"] = log_prob

        # --- Step 3: Retrieve and generate ---
        if should_retrieve and topk_to_use > 0:
            # Apply legacy query rewriter if learned rewrite is not active
            if not self.use_learned_rewrite:
                if (use_rewrite or (use_rewrite is None and self.use_query_rewriter)) and self.query_rewriter:
                    retrieval_query = self.query_rewriter.rewrite(question)
                    metadata["rewritten_question"] = retrieval_query

            # Retrieve with (possibly rewritten) query, generate with original question
            docs = self.retriever.retrieve([retrieval_query], topk=topk_to_use)[0]
            answer = self.generator.generate_with_retrieval(question, docs)
            return answer, docs, metadata
        else:
            # Generate directly without retrieval
            answer = self.generator.generate_direct(question)
            return answer, None, metadata
    
    def store_transition(
        self,
        log_prob: torch.Tensor,
        reward: float,
        entropy: Optional[torch.Tensor] = None
    ):
        """Store transition for policy gradient update."""
        if log_prob is not None:
            self.episode_log_probs.append(log_prob)
            self.episode_rewards.append(reward)
            if entropy is not None:
                self.episode_entropies.append(entropy)
    
    def update_policy(
        self,
        baseline: float = 0.0,
        entropy_coef: float = 0.01
    ) -> Tuple[float, float]:
        """
        Update policy using REINFORCE with entropy bonus.

        SOLUTION 3: Add entropy regularization to encourage exploration.
        This prevents the policy from becoming too deterministic too early,
        which is a key cause of the "lazy agent" problem.

        Args:
            baseline: Reward baseline for variance reduction
            entropy_coef: Coefficient for entropy bonus (default 0.01)
                         Higher = more exploration

        Returns:
            Tuple of (policy_loss, entropy_bonus)
        """
        if not self.episode_log_probs or self.policy_optimizer is None:
            return 0.0, 0.0

        # Compute advantages
        advantages = [r - baseline for r in self.episode_rewards]

        # Compute policy gradient loss
        policy_losses = []
        for log_prob, advantage in zip(self.episode_log_probs, advantages):
            policy_losses.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_losses).mean()

        # SOLUTION 3: Add entropy bonus to encourage exploration
        entropy_bonus = torch.tensor(0.0)
        if self.episode_entropies:
            entropy_bonus = torch.stack(self.episode_entropies).mean()
            # Subtract entropy (we want to MAXIMIZE entropy, so subtract from loss)
            # Higher entropy_coef = more exploration
            policy_loss = policy_loss - entropy_coef * entropy_bonus

        # Update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()

        # Clear buffer
        avg_loss = policy_loss.item()
        avg_entropy = entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_entropies = []

        return avg_loss, avg_entropy

    def store_rewrite_transition(
        self,
        log_prob: torch.Tensor,
        reward: float,
        entropy: Optional[torch.Tensor] = None
    ):
        """Store transition for rewrite policy gradient update."""
        if log_prob is not None:
            self.rewrite_log_probs.append(log_prob)
            self.rewrite_rewards.append(reward)
            if entropy is not None:
                self.rewrite_entropies.append(entropy)

    def update_rewrite_policy(
        self,
        baseline: float = 0.0,
        entropy_coef: float = 0.02
    ) -> Tuple[float, float]:
        """
        Update rewrite policy using REINFORCE with entropy bonus.

        Uses higher default entropy_coef (0.02) than topk policy (0.01)
        to encourage more exploration across rewrite strategies.

        Returns:
            Tuple of (policy_loss, entropy_bonus)
        """
        if not self.rewrite_log_probs or self.rewrite_optimizer is None:
            return 0.0, 0.0

        advantages = [r - baseline for r in self.rewrite_rewards]

        policy_losses = []
        for log_prob, advantage in zip(self.rewrite_log_probs, advantages):
            policy_losses.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_losses).mean()

        entropy_bonus = torch.tensor(0.0)
        if self.rewrite_entropies:
            entropy_bonus = torch.stack(self.rewrite_entropies).mean()
            policy_loss = policy_loss - entropy_coef * entropy_bonus

        self.rewrite_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rewrite_policy_network.parameters(), 1.0)
        self.rewrite_optimizer.step()

        avg_loss = policy_loss.item()
        avg_entropy = entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus
        self.rewrite_log_probs = []
        self.rewrite_rewards = []
        self.rewrite_entropies = []

        return avg_loss, avg_entropy

    def update_rewrite_policy_grpo(
        self,
        group_log_probs: List[List[torch.Tensor]],
        group_rewards: List[List[float]],
        group_entropies: List[List[torch.Tensor]],
        entropy_coef: float = 0.02,
        eps: float = 1e-8
    ) -> Tuple[float, float]:
        """
        Update rewrite policy using GRPO.

        Same group-relative advantage as topk GRPO, but for rewrite strategies.
        """
        if not group_log_probs or self.rewrite_optimizer is None:
            return 0.0, 0.0

        all_policy_losses = []
        all_entropies = []

        for log_probs, rewards, entropies in zip(group_log_probs, group_rewards, group_entropies):
            if len(rewards) < 2:
                continue

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std()
            advantages = (rewards_tensor - mean_r) / (std_r + eps)

            for log_prob, advantage in zip(log_probs, advantages):
                all_policy_losses.append(-log_prob * advantage)

            all_entropies.extend(entropies)

        if not all_policy_losses:
            return 0.0, 0.0

        policy_loss = torch.stack(all_policy_losses).mean()

        entropy_bonus = torch.tensor(0.0)
        if all_entropies:
            entropy_bonus = torch.stack(all_entropies).mean()
            policy_loss = policy_loss - entropy_coef * entropy_bonus

        self.rewrite_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rewrite_policy_network.parameters(), 1.0)
        self.rewrite_optimizer.step()

        avg_loss = policy_loss.item()
        avg_entropy = entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus

        return avg_loss, avg_entropy

    def update_policy_grpo(
        self,
        group_log_probs: List[List[torch.Tensor]],
        group_rewards: List[List[float]],
        group_entropies: List[List[torch.Tensor]],
        entropy_coef: float = 0.01,
        eps: float = 1e-8
    ) -> Tuple[float, float]:
        """
        Update policy using GRPO (Group Relative Policy Optimization).

        For each query, multiple actions are sampled. Advantages are computed
        relative to the group mean (within-group normalization), producing
        lower-variance gradients than REINFORCE without a value network.

        Reference: DeepSeek R1 (2024), RAG-RL (2025)

        Args:
            group_log_probs: List of groups, each a list of log_probs for sampled actions
            group_rewards: List of groups, each a list of rewards for sampled actions
            group_entropies: List of groups, each a list of entropies
            entropy_coef: Entropy bonus coefficient
            eps: Small constant for numerical stability

        Returns:
            Tuple of (policy_loss, avg_entropy)
        """
        if not group_log_probs or self.policy_optimizer is None:
            return 0.0, 0.0

        all_policy_losses = []
        all_entropies = []

        for log_probs, rewards, entropies in zip(group_log_probs, group_rewards, group_entropies):
            if len(rewards) < 2:
                continue

            # Group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std()
            advantages = (rewards_tensor - mean_r) / (std_r + eps)

            # Policy gradient: -log_pi(a_i) * A_i
            for log_prob, advantage in zip(log_probs, advantages):
                all_policy_losses.append(-log_prob * advantage)

            all_entropies.extend(entropies)

        if not all_policy_losses:
            return 0.0, 0.0

        policy_loss = torch.stack(all_policy_losses).mean()

        # Entropy bonus
        entropy_bonus = torch.tensor(0.0)
        if all_entropies:
            entropy_bonus = torch.stack(all_entropies).mean()
            policy_loss = policy_loss - entropy_coef * entropy_bonus

        # Update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()

        avg_loss = policy_loss.item()
        avg_entropy = entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus

        return avg_loss, avg_entropy


class RLTrainer:
    """
    Trainer for RL-enhanced RAG pipeline.

    Trains:
    1. Retrieval policy (when to retrieve / how many docs)
    2. Query rewriter (how to retrieve)

    Supports both binary retrieval decisions and dynamic topk selection.
    """

    def __init__(
        self,
        pipeline: EnhancedRAGPipeline,
        retrieval_cost: float = 0.1,
        wrong_no_retrieval_penalty: float = 0.3,
        entropy_coef: float = 0.01,
        rewrite_entropy_coef: float = 0.02,
        eval_temperature: float = 0.7,
        output_dir: Optional[str] = None,
        use_wandb: bool = False,
        # Dynamic TopK parameters
        use_dynamic_cost: bool = False,
        base_retrieval_cost: float = 0.05,
        per_doc_cost: float = 0.01
    ):
        """
        Initialize trainer.

        Args:
            pipeline: Enhanced RAG pipeline to train
            retrieval_cost: Cost penalty for retrieval (binary mode)
            wrong_no_retrieval_penalty: Penalty for bad answers without retrieval
            entropy_coef: Coefficient for entropy bonus (higher = more exploration)
            rewrite_entropy_coef: Entropy coef for rewrite policy (higher default to explore strategies)
            eval_temperature: Temperature for soft sampling during evaluation
            output_dir: Directory for saving results
            use_wandb: Enable Weights & Biases logging
            use_dynamic_cost: If True, use per-document cost scaling
            base_retrieval_cost: Fixed cost for any retrieval (dynamic mode)
            per_doc_cost: Cost per document retrieved (dynamic mode)
        """
        self.pipeline = pipeline
        self.retrieval_cost = retrieval_cost
        self.wrong_no_retrieval_penalty = wrong_no_retrieval_penalty
        self.entropy_coef = entropy_coef
        self.rewrite_entropy_coef = rewrite_entropy_coef
        self.eval_temperature = eval_temperature
        self.use_wandb = use_wandb
        self.use_dynamic_cost = use_dynamic_cost
        self.base_retrieval_cost = base_retrieval_cost
        self.per_doc_cost = per_doc_cost

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"experiments/results/rl_training_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure reward calculator based on mode
        if use_dynamic_cost or pipeline.use_dynamic_topk:
            self.reward_calculator = RAGRewardCalculator(
                retrieval_cost=retrieval_cost,
                correct_no_retrieval_bonus=0.1,
                wrong_no_retrieval_penalty=wrong_no_retrieval_penalty,
                use_f1=True,
                use_dynamic_cost=True,
                base_retrieval_cost=base_retrieval_cost,
                per_doc_cost=per_doc_cost
            )
        else:
            self.reward_calculator = RAGRewardCalculator(
                retrieval_cost=retrieval_cost,
                correct_no_retrieval_bonus=0.1,
                wrong_no_retrieval_penalty=wrong_no_retrieval_penalty,
                use_f1=True
            )

        # Training history
        self.history = {
            "train_rewards": [],
            "train_f1": [],
            "train_retrieval_rate": [],
            "train_entropy": [],
            "train_avg_topk": [],
            "train_topk_distribution": [],
            "val_rewards": [],
            "val_f1": [],
            "val_retrieval_rate": [],
            "val_avg_topk": [],
            "policy_losses": [],
            "lazy_agent_failures": [],
            # Rewrite policy history
            "rewrite_losses": [],
            "rewrite_entropy": [],
            "rewrite_strategy_distribution": [],
        }

        # W&B step counter for consistent x-axis
        self._wandb_step = 0
        self._current_epoch = 0
    
    def train_epoch(
        self,
        train_data: List[Dict[str, Any]],
        update_every: int = 10,
        epsilon: float = 0.2,  # Exploration rate
        min_retrieval_rate: float = 0.0,  # Curriculum learning minimum
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Supports both binary retrieval decisions and dynamic topk selection.
        Jointly trains rewrite policy when use_learned_rewrite is enabled.
        """
        epoch_rewards = []
        epoch_f1 = []
        epoch_retrieve = []
        epoch_entropy = []
        epoch_topk = []
        epoch_wrong_no_retr_penalties = []
        epoch_rewrite_strategies = []

        use_dynamic_topk = self.pipeline.use_dynamic_topk
        use_learned_rewrite = self.pipeline.use_learned_rewrite

        for i, sample in enumerate(train_data):
            question = sample["question"]
            golden_answers = sample["golden_answers"]

            # 1. Epsilon-greedy exploration + curriculum learning
            force_retrieve = False
            topk_override = None
            rewrite_strategy_override = None

            if random.random() < epsilon:
                force_retrieve = True
                if use_dynamic_topk:
                    topk_override = random.choice(self.pipeline.topk_options)
                    if topk_override == 0:
                        topk_override = random.choice([k for k in self.pipeline.topk_options if k > 0])
                # Also explore rewrite strategies randomly
                if use_learned_rewrite and random.random() < 0.5:
                    rewrite_strategy_override = random.randint(0, QueryRewritePolicyNetwork.NUM_STRATEGIES - 1)
            elif min_retrieval_rate > 0 and random.random() < min_retrieval_rate:
                force_retrieve = True
                if use_dynamic_topk:
                    topk_override = random.choice([5, 7, 10])

            # 2. Forward pass
            if use_dynamic_topk:
                answer, docs, metadata = self.pipeline.answer(
                    question,
                    topk_override=topk_override,
                    rewrite_strategy_override=rewrite_strategy_override,
                    deterministic=False
                )
            else:
                should_retrieve_override = True if force_retrieve else None
                answer, docs, metadata = self.pipeline.answer(
                    question,
                    should_retrieve=should_retrieve_override,
                    rewrite_strategy_override=rewrite_strategy_override,
                    deterministic=False
                )

            did_retrieve = metadata["did_retrieve"]
            topk_used = metadata.get("topk_used", 0)
            log_prob = metadata["log_prob"]
            entropy = metadata.get("entropy")
            rewrite_log_prob = metadata.get("rewrite_log_prob")
            rewrite_entropy = metadata.get("rewrite_entropy")
            rewrite_strategy = metadata.get("rewrite_strategy")

            epoch_topk.append(topk_used)

            if entropy is not None:
                epoch_entropy.append(entropy)

            if rewrite_strategy is not None:
                epoch_rewrite_strategies.append(rewrite_strategy)

            # 3. Train legacy Query Rewriter if retrieval was performed
            if did_retrieve and hasattr(self.pipeline, 'query_rewriter') and \
               self.pipeline.query_rewriter is not None and \
               hasattr(self.pipeline.query_rewriter, 'train_step'):
                self.pipeline.query_rewriter.train_step(question, golden_answers)

            # 4. Compute reward
            reward, metrics = self.reward_calculator.compute_reward(
                prediction=answer,
                ground_truths=golden_answers,
                did_retrieve=did_retrieve,
                topk_used=topk_used
            )

            if metrics.get("wrong_no_retrieval_penalty", 0) > 0:
                epoch_wrong_no_retr_penalties.append(1)

            # 5. Store transitions for BOTH policies
            if log_prob is not None:
                entropy_tensor = None
                if entropy is not None:
                    entropy_tensor = torch.tensor(entropy) if not isinstance(entropy, torch.Tensor) else entropy
                self.pipeline.store_transition(log_prob, reward, entropy_tensor)

            if rewrite_log_prob is not None:
                rewrite_entropy_tensor = None
                if rewrite_entropy is not None:
                    rewrite_entropy_tensor = torch.tensor(rewrite_entropy) if not isinstance(rewrite_entropy, torch.Tensor) else rewrite_entropy
                self.pipeline.store_rewrite_transition(rewrite_log_prob, reward, rewrite_entropy_tensor)

            epoch_rewards.append(reward)
            epoch_f1.append(metrics["f1"])
            epoch_retrieve.append(1 if did_retrieve else 0)

            # Update BOTH policies periodically
            if (i + 1) % update_every == 0:
                baseline = sum(epoch_rewards[-update_every:]) / update_every
                loss, avg_ent = self.pipeline.update_policy(
                    baseline, entropy_coef=self.entropy_coef
                )
                self.history["policy_losses"].append(loss)

                # Update rewrite policy if active
                rewrite_loss = 0.0
                rewrite_ent = 0.0
                if use_learned_rewrite:
                    rewrite_loss, rewrite_ent = self.pipeline.update_rewrite_policy(
                        baseline, entropy_coef=self.rewrite_entropy_coef
                    )
                    self.history["rewrite_losses"].append(rewrite_loss)

                if verbose:
                    avg_f1 = sum(epoch_f1[-update_every:]) / update_every
                    avg_retr = sum(epoch_retrieve[-update_every:]) / update_every
                    msg = f"  [{i+1}/{len(train_data)}] F1: {avg_f1:.3f}, Retr: {avg_retr:.1%}"
                    if use_dynamic_topk:
                        avg_topk = sum(epoch_topk[-update_every:]) / update_every
                        msg += f", AvgK: {avg_topk:.1f}"
                    msg += f", Loss: {loss:.4f}, Ent: {avg_ent:.4f}"
                    if use_learned_rewrite:
                        msg += f", RwLoss: {rewrite_loss:.4f}"
                    print(msg)

                if self.use_wandb:
                    import wandb
                    self._wandb_step += 1
                    log_dict = {
                        "train/f1": sum(epoch_f1[-update_every:]) / update_every,
                        "train/retrieval_rate": sum(epoch_retrieve[-update_every:]) / update_every,
                        "train/reward": sum(epoch_rewards[-update_every:]) / update_every,
                        "train/policy_loss": loss,
                        "train/entropy": avg_ent,
                        "explore/epsilon": epsilon,
                        "explore/min_retrieval_rate": min_retrieval_rate,
                        "explore/curriculum_phase": self._current_epoch // max(1, self._current_epoch) if hasattr(self, '_current_epoch') else 0,
                        "step": self._wandb_step,
                        "epoch": self._current_epoch
                    }
                    if use_dynamic_topk:
                        log_dict["train/avg_topk"] = sum(epoch_topk[-update_every:]) / update_every
                    if use_learned_rewrite:
                        log_dict["train/rewrite_loss"] = rewrite_loss
                        log_dict["train/rewrite_entropy"] = rewrite_ent
                    wandb.log(log_dict)


        # Final updates
        if len(self.pipeline.episode_log_probs) > 0:
            baseline = sum(epoch_rewards) / len(epoch_rewards)
            self.pipeline.update_policy(baseline, entropy_coef=self.entropy_coef)
        if use_learned_rewrite and len(self.pipeline.rewrite_log_probs) > 0:
            baseline = sum(epoch_rewards) / len(epoch_rewards)
            self.pipeline.update_rewrite_policy(baseline, entropy_coef=self.rewrite_entropy_coef)

        # Compute epoch metrics
        avg_entropy = sum(epoch_entropy) / len(epoch_entropy) if epoch_entropy else 0.0
        lazy_agent_failures = len(epoch_wrong_no_retr_penalties)
        avg_topk = sum(epoch_topk) / len(epoch_topk) if epoch_topk else 0.0

        metrics = {
            "avg_reward": sum(epoch_rewards) / len(epoch_rewards),
            "avg_f1": sum(epoch_f1) / len(epoch_f1),
            "retrieval_rate": sum(epoch_retrieve) / len(epoch_retrieve),
            "avg_entropy": avg_entropy,
            "avg_topk": avg_topk,
            "lazy_agent_failures": lazy_agent_failures
        }

        # Topk distribution
        if use_dynamic_topk and epoch_topk:
            from collections import Counter
            topk_counts = Counter(epoch_topk)
            topk_dist = {k: topk_counts.get(k, 0) / len(epoch_topk) for k in self.pipeline.topk_options}
            metrics["topk_distribution"] = topk_dist
            self.history["train_topk_distribution"].append(topk_dist)

        # Rewrite strategy distribution
        if use_learned_rewrite and epoch_rewrite_strategies:
            from collections import Counter
            strategy_counts = Counter(epoch_rewrite_strategies)
            strategy_dist = {
                StrategyRewriter.STRATEGY_NAMES[s]: strategy_counts.get(s, 0) / len(epoch_rewrite_strategies)
                for s in range(QueryRewritePolicyNetwork.NUM_STRATEGIES)
            }
            metrics["rewrite_strategy_distribution"] = strategy_dist
            self.history["rewrite_strategy_distribution"].append(strategy_dist)

        self.history["train_rewards"].append(metrics["avg_reward"])
        self.history["train_f1"].append(metrics["avg_f1"])
        self.history["train_retrieval_rate"].append(metrics["retrieval_rate"])
        self.history["train_entropy"].append(avg_entropy)
        self.history["train_avg_topk"].append(avg_topk)
        self.history["lazy_agent_failures"].append(lazy_agent_failures)

        # Log epoch-level summary to wandb
        if self.use_wandb:
            import wandb
            log_dict = {
                "epoch_summary/train_f1": metrics["avg_f1"],
                "epoch_summary/train_reward": metrics["avg_reward"],
                "epoch_summary/train_retrieval_rate": metrics["retrieval_rate"],
                "epoch_summary/train_entropy": avg_entropy,
                "epoch_summary/lazy_agent_failures": lazy_agent_failures,
                "epoch_summary/lazy_agent_rate": lazy_agent_failures / len(train_data) if train_data else 0,
                "epoch": self._current_epoch
            }
            if use_dynamic_topk:
                log_dict["epoch_summary/train_avg_topk"] = avg_topk
                if "topk_distribution" in metrics:
                    for k, v in metrics["topk_distribution"].items():
                        log_dict[f"topk_dist/k{k}"] = v
            if use_learned_rewrite and "rewrite_strategy_distribution" in metrics:
                for name, frac in metrics["rewrite_strategy_distribution"].items():
                    log_dict[f"rewrite_dist/{name}"] = frac
            wandb.log(log_dict)

        return metrics
    
    def train_epoch_grpo(
        self,
        train_data: List[Dict[str, Any]],
        group_size: int = 8,
        epsilon: float = 0.2,
        min_retrieval_rate: float = 0.0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch using GRPO (Group Relative Policy Optimization).

        For each query, samples group_size different actions (both k and rewrite
        strategy), executes all, computes group-relative advantages, and updates
        both policies jointly.
        """
        epoch_rewards = []
        epoch_f1 = []
        epoch_retrieve = []
        epoch_entropy = []
        epoch_topk = []
        epoch_wrong_no_retr_penalties = []
        epoch_rewrite_strategies = []

        use_dynamic_topk = self.pipeline.use_dynamic_topk
        use_learned_rewrite = self.pipeline.use_learned_rewrite

        # Accumulate groups for batch update (topk policy)
        group_log_probs = []
        group_rewards = []
        group_entropies = []

        # Accumulate groups for rewrite policy
        rewrite_group_log_probs = []
        rewrite_group_rewards = []
        rewrite_group_entropies = []

        for i, sample in enumerate(train_data):
            question = sample["question"]
            golden_answers = sample["golden_answers"]

            sample_log_probs = []
            sample_rewards = []
            sample_entropies = []
            sample_f1s = []
            sample_topks = []

            # Rewrite groups for this query
            sample_rewrite_log_probs = []
            sample_rewrite_rewards = []
            sample_rewrite_entropies = []

            for g in range(group_size):
                force_retrieve = False
                topk_override = None
                rewrite_strategy_override = None

                if random.random() < epsilon:
                    force_retrieve = True
                    if use_dynamic_topk:
                        topk_override = random.choice(
                            [k for k in self.pipeline.topk_options if k > 0]
                        )
                    if use_learned_rewrite and random.random() < 0.5:
                        rewrite_strategy_override = random.randint(0, QueryRewritePolicyNetwork.NUM_STRATEGIES - 1)
                elif min_retrieval_rate > 0 and random.random() < min_retrieval_rate:
                    force_retrieve = True
                    if use_dynamic_topk:
                        topk_override = random.choice([5, 7, 10])

                if use_dynamic_topk:
                    answer, docs, metadata = self.pipeline.answer(
                        question,
                        topk_override=topk_override,
                        rewrite_strategy_override=rewrite_strategy_override,
                        deterministic=False
                    )
                else:
                    should_retrieve_override = True if force_retrieve else None
                    answer, docs, metadata = self.pipeline.answer(
                        question,
                        should_retrieve=should_retrieve_override,
                        rewrite_strategy_override=rewrite_strategy_override,
                        deterministic=False
                    )

                did_retrieve = metadata["did_retrieve"]
                topk_used = metadata.get("topk_used", 0)
                log_prob = metadata["log_prob"]
                entropy = metadata.get("entropy")
                rewrite_log_prob = metadata.get("rewrite_log_prob")
                rewrite_entropy = metadata.get("rewrite_entropy")
                rewrite_strategy = metadata.get("rewrite_strategy")

                reward, metrics = self.reward_calculator.compute_reward(
                    prediction=answer,
                    ground_truths=golden_answers,
                    did_retrieve=did_retrieve,
                    topk_used=topk_used
                )

                # Topk policy transitions
                if log_prob is not None:
                    sample_log_probs.append(log_prob)
                    sample_rewards.append(reward)
                    entropy_tensor = None
                    if entropy is not None:
                        entropy_tensor = torch.tensor(entropy) if not isinstance(entropy, torch.Tensor) else entropy
                    else:
                        entropy_tensor = torch.tensor(0.0)
                    sample_entropies.append(entropy_tensor)

                # Rewrite policy transitions
                if rewrite_log_prob is not None:
                    sample_rewrite_log_probs.append(rewrite_log_prob)
                    sample_rewrite_rewards.append(reward)
                    rw_ent = None
                    if rewrite_entropy is not None:
                        rw_ent = torch.tensor(rewrite_entropy) if not isinstance(rewrite_entropy, torch.Tensor) else rewrite_entropy
                    else:
                        rw_ent = torch.tensor(0.0)
                    sample_rewrite_entropies.append(rw_ent)

                if rewrite_strategy is not None:
                    epoch_rewrite_strategies.append(rewrite_strategy)

                sample_f1s.append(metrics["f1"])
                sample_topks.append(topk_used)

                if metrics.get("wrong_no_retrieval_penalty", 0) > 0:
                    epoch_wrong_no_retr_penalties.append(1)

            # Accumulate topk group
            if len(sample_log_probs) >= 2:
                group_log_probs.append(sample_log_probs)
                group_rewards.append(sample_rewards)
                group_entropies.append(sample_entropies)

            # Accumulate rewrite group
            if len(sample_rewrite_log_probs) >= 2:
                rewrite_group_log_probs.append(sample_rewrite_log_probs)
                rewrite_group_rewards.append(sample_rewrite_rewards)
                rewrite_group_entropies.append(sample_rewrite_entropies)

            epoch_rewards.append(sum(sample_rewards) / len(sample_rewards) if sample_rewards else 0)
            epoch_f1.append(max(sample_f1s) if sample_f1s else 0)
            epoch_retrieve.append(1 if any(k > 0 for k in sample_topks) else 0)
            epoch_topk.append(sum(sample_topks) / len(sample_topks) if sample_topks else 0)
            if sample_entropies:
                epoch_entropy.append(sum(e.item() for e in sample_entropies) / len(sample_entropies))

            # Update both policies every N queries
            if (i + 1) % 5 == 0 and group_log_probs:
                loss, avg_ent = self.pipeline.update_policy_grpo(
                    group_log_probs, group_rewards, group_entropies,
                    entropy_coef=self.entropy_coef
                )
                self.history["policy_losses"].append(loss)
                group_log_probs = []
                group_rewards = []
                group_entropies = []

                rewrite_loss = 0.0
                if use_learned_rewrite and rewrite_group_log_probs:
                    rewrite_loss, _ = self.pipeline.update_rewrite_policy_grpo(
                        rewrite_group_log_probs, rewrite_group_rewards, rewrite_group_entropies,
                        entropy_coef=self.rewrite_entropy_coef
                    )
                    self.history["rewrite_losses"].append(rewrite_loss)
                    rewrite_group_log_probs = []
                    rewrite_group_rewards = []
                    rewrite_group_entropies = []

                if verbose:
                    recent = min(5, len(epoch_f1))
                    avg_f1 = sum(epoch_f1[-recent:]) / recent
                    avg_retr = sum(epoch_retrieve[-recent:]) / recent
                    msg = f"  [{i+1}/{len(train_data)}] F1: {avg_f1:.3f}, Retr: {avg_retr:.1%}"
                    if use_dynamic_topk:
                        avg_topk = sum(epoch_topk[-recent:]) / recent
                        msg += f", AvgK: {avg_topk:.1f}"
                    msg += f", Loss: {loss:.4f}, Ent: {avg_ent:.4f}"
                    if use_learned_rewrite:
                        msg += f", RwLoss: {rewrite_loss:.4f}"
                    print(msg)

                if self.use_wandb:
                    import wandb
                    self._wandb_step += 1
                    log_dict = {
                        "train/f1": avg_f1,
                        "train/retrieval_rate": avg_retr,
                        "train/reward": sum(epoch_rewards[-recent:]) / recent,
                        "train/policy_loss": loss,
                        "train/entropy": avg_ent,
                        "explore/epsilon": epsilon,
                        "step": self._wandb_step,
                        "epoch": self._current_epoch,
                        "algorithm": "grpo"
                    }
                    if use_dynamic_topk:
                        log_dict["train/avg_topk"] = sum(epoch_topk[-recent:]) / recent
                    if use_learned_rewrite:
                        log_dict["train/rewrite_loss"] = rewrite_loss
                    wandb.log(log_dict)

        # Final updates for remaining groups
        if group_log_probs:
            self.pipeline.update_policy_grpo(
                group_log_probs, group_rewards, group_entropies,
                entropy_coef=self.entropy_coef
            )
        if use_learned_rewrite and rewrite_group_log_probs:
            self.pipeline.update_rewrite_policy_grpo(
                rewrite_group_log_probs, rewrite_group_rewards, rewrite_group_entropies,
                entropy_coef=self.rewrite_entropy_coef
            )

        # Compute epoch metrics
        avg_entropy = sum(epoch_entropy) / len(epoch_entropy) if epoch_entropy else 0.0
        lazy_agent_failures = len(epoch_wrong_no_retr_penalties)
        avg_topk = sum(epoch_topk) / len(epoch_topk) if epoch_topk else 0.0

        metrics = {
            "avg_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
            "avg_f1": sum(epoch_f1) / len(epoch_f1) if epoch_f1 else 0,
            "retrieval_rate": sum(epoch_retrieve) / len(epoch_retrieve) if epoch_retrieve else 0,
            "avg_entropy": avg_entropy,
            "avg_topk": avg_topk,
            "lazy_agent_failures": lazy_agent_failures
        }

        # Topk distribution
        if use_dynamic_topk and epoch_topk:
            from collections import Counter
            topk_counts = Counter([int(round(t)) for t in epoch_topk])
            topk_dist = {k: topk_counts.get(k, 0) / len(epoch_topk) for k in self.pipeline.topk_options}
            metrics["topk_distribution"] = topk_dist
            self.history["train_topk_distribution"].append(topk_dist)

        # Rewrite strategy distribution
        if use_learned_rewrite and epoch_rewrite_strategies:
            from collections import Counter
            strategy_counts = Counter(epoch_rewrite_strategies)
            strategy_dist = {
                StrategyRewriter.STRATEGY_NAMES[s]: strategy_counts.get(s, 0) / len(epoch_rewrite_strategies)
                for s in range(QueryRewritePolicyNetwork.NUM_STRATEGIES)
            }
            metrics["rewrite_strategy_distribution"] = strategy_dist
            self.history["rewrite_strategy_distribution"].append(strategy_dist)

        self.history["train_rewards"].append(metrics["avg_reward"])
        self.history["train_f1"].append(metrics["avg_f1"])
        self.history["train_retrieval_rate"].append(metrics["retrieval_rate"])
        self.history["train_entropy"].append(avg_entropy)
        self.history["train_avg_topk"].append(avg_topk)
        self.history["lazy_agent_failures"].append(lazy_agent_failures)

        # Log epoch summary to wandb
        if self.use_wandb:
            import wandb
            log_dict = {
                "epoch_summary/train_f1": metrics["avg_f1"],
                "epoch_summary/train_reward": metrics["avg_reward"],
                "epoch_summary/train_retrieval_rate": metrics["retrieval_rate"],
                "epoch_summary/train_entropy": avg_entropy,
                "epoch_summary/lazy_agent_failures": lazy_agent_failures,
                "epoch": self._current_epoch,
                "algorithm": "grpo"
            }
            if use_dynamic_topk:
                log_dict["epoch_summary/train_avg_topk"] = avg_topk
            if use_learned_rewrite and "rewrite_strategy_distribution" in metrics:
                for name, frac in metrics["rewrite_strategy_distribution"].items():
                    log_dict[f"rewrite_dist/{name}"] = frac
            wandb.log(log_dict)

        return metrics

    def evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        verbose: bool = True,
        use_temperature: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on validation data.

        Supports both binary and dynamic topk modes.

        Args:
            eval_data: Evaluation samples
            verbose: Print progress
            use_temperature: If True, use temperature-based soft sampling

        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_f1 = []
        eval_em = []
        eval_retrieve = []
        eval_probs = []
        eval_topk = []
        eval_rewrite_strategies = []

        use_dynamic_topk = self.pipeline.use_dynamic_topk
        use_learned_rewrite = self.pipeline.use_learned_rewrite

        for i, sample in enumerate(eval_data):
            question = sample["question"]
            golden_answers = sample["golden_answers"]

            answer, docs, metadata = self.pipeline.answer(
                question,
                deterministic=True,
                temperature=self.eval_temperature if use_temperature else 1.0
            )

            did_retrieve = metadata["did_retrieve"]
            topk_used = metadata.get("topk_used", 0)
            prob = metadata.get("retrieval_probability") or 0.5
            rewrite_strategy = metadata.get("rewrite_strategy")
            eval_probs.append(prob)
            eval_topk.append(topk_used)
            if rewrite_strategy is not None:
                eval_rewrite_strategies.append(rewrite_strategy)

            reward, metrics = self.reward_calculator.compute_reward(
                prediction=answer,
                ground_truths=golden_answers,
                did_retrieve=did_retrieve,
                topk_used=topk_used
            )

            eval_rewards.append(reward)
            eval_f1.append(metrics["f1"])
            eval_em.append(metrics["em"])
            eval_retrieve.append(1 if did_retrieve else 0)

        avg_prob = sum(eval_probs) / len(eval_probs) if eval_probs else 0.5
        avg_topk = sum(eval_topk) / len(eval_topk) if eval_topk else 0.0

        metrics = {
            "avg_reward": sum(eval_rewards) / len(eval_rewards),
            "avg_f1": sum(eval_f1) / len(eval_f1),
            "avg_em": sum(eval_em) / len(eval_em),
            "retrieval_rate": sum(eval_retrieve) / len(eval_retrieve),
            "avg_retrieval_prob": avg_prob,
            "avg_topk": avg_topk
        }

        if use_dynamic_topk and eval_topk:
            from collections import Counter
            topk_counts = Counter(eval_topk)
            topk_dist = {k: topk_counts.get(k, 0) / len(eval_topk) for k in self.pipeline.topk_options}
            metrics["topk_distribution"] = topk_dist

        if use_learned_rewrite and eval_rewrite_strategies:
            from collections import Counter
            strategy_counts = Counter(eval_rewrite_strategies)
            strategy_dist = {
                StrategyRewriter.STRATEGY_NAMES[s]: strategy_counts.get(s, 0) / len(eval_rewrite_strategies)
                for s in range(QueryRewritePolicyNetwork.NUM_STRATEGIES)
            }
            metrics["rewrite_strategy_distribution"] = strategy_dist

        self.history["val_rewards"].append(metrics["avg_reward"])
        self.history["val_f1"].append(metrics["avg_f1"])
        self.history["val_retrieval_rate"].append(metrics["retrieval_rate"])
        self.history["val_avg_topk"].append(avg_topk)

        if verbose:
            msg = f"  Eval - F1: {metrics['avg_f1']:.3f}, EM: {metrics['avg_em']:.3f}, "
            msg += f"Reward: {metrics['avg_reward']:.3f}, Retr: {metrics['retrieval_rate']:.1%}"
            if use_dynamic_topk:
                msg += f", AvgK: {avg_topk:.1f}"
            else:
                msg += f", Avg Prob: {avg_prob:.3f}"
            if use_learned_rewrite and "rewrite_strategy_distribution" in metrics:
                dominant = max(metrics["rewrite_strategy_distribution"], key=metrics["rewrite_strategy_distribution"].get)
                msg += f", TopStrat: {dominant}"
            print(msg)

        if self.use_wandb:
            import wandb
            log_dict = {
                "val/f1": metrics["avg_f1"],
                "val/em": metrics["avg_em"],
                "val/reward": metrics["avg_reward"],
                "val/retrieval_rate": metrics["retrieval_rate"],
                "val/avg_retrieval_prob": avg_prob,
                "epoch_summary/val_f1": metrics["avg_f1"],
                "epoch_summary/val_retrieval_rate": metrics["retrieval_rate"],
                "epoch": self._current_epoch
            }
            if use_dynamic_topk:
                log_dict["val/avg_topk"] = avg_topk
                log_dict["epoch_summary/val_avg_topk"] = avg_topk
                if "topk_distribution" in metrics:
                    for k, v in metrics["topk_distribution"].items():
                        log_dict[f"val_topk_dist/k{k}"] = v
            if use_learned_rewrite and "rewrite_strategy_distribution" in metrics:
                for name, frac in metrics["rewrite_strategy_distribution"].items():
                    log_dict[f"val_rewrite_dist/{name}"] = frac
            wandb.log(log_dict)

        return metrics
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        epochs: int = 5,
        update_every: int = 10,
        start_epsilon: float = 0.5,  # Start with high exploration
        min_epsilon: float = 0.05,
        use_curriculum: bool = True,  # NEW: Enable curriculum learning
        curriculum_phases: int = 3,  # NEW: Number of curriculum phases
        algorithm: str = "reinforce",  # Algorithm: "reinforce" or "grpo"
        group_size: int = 8  # GRPO group size
    ) -> Dict[str, Any]:
        """
        Full training loop with epsilon decay and curriculum learning.

        Supports both REINFORCE and GRPO algorithms.

        SOLUTION 2: Curriculum Learning
        - Phase 1: Force high retrieval rate (start learning with good examples)
        - Phase 2: Gradually reduce forced retrieval
        - Phase 3: Let policy decide with learned baseline
        """
        use_dynamic_topk = self.pipeline.use_dynamic_topk

        print(f"\n🚀 Starting RL Training")
        print(f"   Algorithm: {algorithm.upper()}" + (f" (group_size={group_size})" if algorithm == "grpo" else ""))
        print(f"   Train samples: {len(train_data)}")
        print(f"   Val samples: {len(val_data)}")
        print(f"   Epochs: {epochs}")
        print(f"   Policy Mode: {'Dynamic TopK' if use_dynamic_topk else 'Binary'}")
        if use_dynamic_topk:
            print(f"   TopK Options: {self.pipeline.topk_options}")
            print(f"   Base Retrieval Cost: {self.base_retrieval_cost}")
            print(f"   Per-Doc Cost: {self.per_doc_cost}")
        else:
            print(f"   Retrieval cost: {self.retrieval_cost}")
        print(f"   Wrong no-retrieval penalty: {self.wrong_no_retrieval_penalty}")
        print(f"   Entropy coefficient: {self.entropy_coef}")
        if self.pipeline.use_learned_rewrite:
            print(f"   Rewrite entropy coef: {self.rewrite_entropy_coef}")
            print(f"   Learned Rewrite: Enabled ({QueryRewritePolicyNetwork.NUM_STRATEGIES} strategies)")
        print(f"   Eval temperature: {self.eval_temperature}")
        print(f"   Start Epsilon: {start_epsilon}")
        print(f"   Curriculum Learning: {'Enabled' if use_curriculum else 'Disabled'}")
        print()

        best_f1 = 0.0
        best_epoch = 0
        epsilon = start_epsilon

        # SOLUTION 2: Curriculum learning phases
        # Divide epochs into phases with decreasing min_retrieval_rate
        if use_curriculum:
            epochs_per_phase = max(1, epochs // curriculum_phases)
            # Start with high forced retrieval, decrease over phases
            phase_min_rates = [0.8, 0.4, 0.0][:curriculum_phases]
        else:
            phase_min_rates = [0.0] * epochs  # No curriculum

        for epoch in range(epochs):
            self._current_epoch = epoch + 1  # Track for W&B logging

            # Determine current curriculum phase
            if use_curriculum:
                phase = min(epoch // epochs_per_phase, curriculum_phases - 1)
                min_retrieval_rate = phase_min_rates[phase]
            else:
                phase = 0
                min_retrieval_rate = 0.0

            print(f"Epoch {epoch + 1}/{epochs} (ε: {epsilon:.2f}, min_retr: {min_retrieval_rate:.1%}, phase: {phase + 1})")

            # Shuffle training data
            random.shuffle(train_data)

            # Train with current epsilon AND curriculum min_retrieval_rate
            if algorithm == "grpo":
                train_metrics = self.train_epoch_grpo(
                    train_data,
                    group_size=group_size,
                    epsilon=epsilon,
                    min_retrieval_rate=min_retrieval_rate
                )
            else:
                train_metrics = self.train_epoch(
                    train_data,
                    update_every,
                    epsilon=epsilon,
                    min_retrieval_rate=min_retrieval_rate
                )

            # Print training metrics
            if use_dynamic_topk:
                print(f"  Train - F1: {train_metrics['avg_f1']:.3f}, "
                      f"Reward: {train_metrics['avg_reward']:.3f}, "
                      f"Retr: {train_metrics['retrieval_rate']:.1%}, "
                      f"AvgK: {train_metrics.get('avg_topk', 0):.1f}, "
                      f"Entropy: {train_metrics.get('avg_entropy', 0):.4f}")
            else:
                print(f"  Train - F1: {train_metrics['avg_f1']:.3f}, "
                      f"Reward: {train_metrics['avg_reward']:.3f}, "
                      f"Retr: {train_metrics['retrieval_rate']:.1%}, "
                      f"Entropy: {train_metrics.get('avg_entropy', 0):.4f}")

            # Check for lazy agent warning
            if train_metrics.get('lazy_agent_failures', 0) > len(train_data) * 0.3:
                print(f"  ⚠️  High lazy agent failures: {train_metrics['lazy_agent_failures']}")

            # Evaluate
            val_metrics = self.evaluate(val_data)

            # Save best model
            if val_metrics["avg_f1"] > best_f1:
                best_f1 = val_metrics["avg_f1"]
                best_epoch = epoch + 1
                self.save_checkpoint("best_model.pt")

            # Epsilon decay
            epsilon = max(min_epsilon, epsilon * 0.7)

            print()

        
        print(f"✅ Training complete!")
        print(f"   Best F1: {best_f1:.3f} (epoch {best_epoch})")

        # Print API usage and cost summary
        if hasattr(self.pipeline.generator, 'print_usage_summary'):
            self.pipeline.generator.print_usage_summary()

        # Save final results
        self.save_results()

        # Get usage stats for return value
        usage_stats = {}
        if hasattr(self.pipeline.generator, 'usage_stats'):
            usage_stats = self.pipeline.generator.usage_stats

        return {
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "history": self.history,
            "usage_stats": usage_stats
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint (includes rewrite policy if active)."""
        if self.pipeline.policy_network is not None:
            path = self.output_dir / filename
            checkpoint = {
                "policy_network": self.pipeline.policy_network.state_dict(),
                "optimizer": self.pipeline.policy_optimizer.state_dict(),
                "history": self.history,
                "config": {
                    "use_dynamic_topk": self.pipeline.use_dynamic_topk,
                    "topk_options": self.pipeline.topk_options,
                    "use_difficulty_features": self.pipeline.use_difficulty_features,
                    "use_learned_rewrite": self.pipeline.use_learned_rewrite
                }
            }
            # Save rewrite policy if active
            if self.pipeline.rewrite_policy_network is not None:
                checkpoint["rewrite_policy_network"] = self.pipeline.rewrite_policy_network.state_dict()
                checkpoint["rewrite_optimizer"] = self.pipeline.rewrite_optimizer.state_dict()
            torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint (backward compatible with pre-rewrite checkpoints)."""
        path = self.output_dir / filename
        if path.exists() and self.pipeline.policy_network is not None:
            checkpoint = torch.load(path)
            self.pipeline.policy_network.load_state_dict(checkpoint["policy_network"])
            self.pipeline.policy_optimizer.load_state_dict(checkpoint["optimizer"])
            self.history = checkpoint.get("history", self.history)
            # Load rewrite policy if present in checkpoint and pipeline supports it
            if "rewrite_policy_network" in checkpoint and self.pipeline.rewrite_policy_network is not None:
                self.pipeline.rewrite_policy_network.load_state_dict(checkpoint["rewrite_policy_network"])
                if "rewrite_optimizer" in checkpoint:
                    self.pipeline.rewrite_optimizer.load_state_dict(checkpoint["rewrite_optimizer"])

    def save_results(self):
        """Save training results to JSON."""
        # Get usage stats if available
        usage_stats = {}
        if hasattr(self.pipeline.generator, 'usage_stats'):
            usage_stats = self.pipeline.generator.usage_stats

        results = {
            "config": {
                "retrieval_cost": self.retrieval_cost,
                "wrong_no_retrieval_penalty": self.wrong_no_retrieval_penalty,
                "entropy_coef": self.entropy_coef,
                "eval_temperature": self.eval_temperature,
                "use_query_rewriter": self.pipeline.use_query_rewriter,
                "use_learned_retrieval": self.pipeline.use_learned_retrieval,
                # Dynamic TopK config
                "use_dynamic_topk": self.pipeline.use_dynamic_topk,
                "topk_options": self.pipeline.topk_options,
                "use_dynamic_cost": self.use_dynamic_cost,
                "base_retrieval_cost": self.base_retrieval_cost,
                "per_doc_cost": self.per_doc_cost,
                # Learned rewrite config
                "use_learned_rewrite": self.pipeline.use_learned_rewrite,
                "rewrite_entropy_coef": self.rewrite_entropy_coef
            },
            "history": self.history,
            "usage_stats": usage_stats
        }

        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)


# Convenience function for quick training
def train_rl_rag(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    epochs: int = 5,
    retrieval_cost: float = 0.1,
    use_query_rewriter: bool = True,
    use_ollama: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to train RL-enhanced RAG.
    
    Args:
        train_data: Training samples (list of {question, golden_answers})
        val_data: Validation samples
        epochs: Training epochs
        retrieval_cost: Retrieval penalty
        use_query_rewriter: Enable query rewriting
        use_ollama: Use Ollama instead of OpenAI
        output_dir: Output directory
        
    Returns:
        Training results
    """
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline(
        use_query_rewriter=use_query_rewriter,
        use_learned_retrieval=True,
        use_ollama=use_ollama
    )
    
    # Initialize trainer
    trainer = RLTrainer(
        pipeline=pipeline,
        retrieval_cost=retrieval_cost,
        output_dir=output_dir
    )
    
    # Train
    results = trainer.train(train_data, val_data, epochs)
    
    return results
