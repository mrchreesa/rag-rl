"""
FlashRAG Component Wrappers

Wraps FlashRAG retriever and generator components for use in RL training.
These provide a clean interface for the LitAgent to interact with.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Add FlashRAG to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
FLASHRAG_PATH = PROJECT_ROOT / 'src/rag/FlashRAG'
sys.path.insert(0, str(FLASHRAG_PATH))


class DenseRetrieverWrapper:
    """
    Wrapper for Dense E5 retriever using FAISS.
    
    This wrapper provides a clean interface for retrieval that can be
    controlled by the RL agent's decision to retrieve or not.
    """
    
    def __init__(
        self, 
        index_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        model_path: str = "intfloat/e5-base-v2"
    ):
        """
        Initialize the dense retriever.
        
        Args:
            index_path: Path to FAISS index. Defaults to custom_e5 index.
            corpus_path: Path to corpus JSONL file.
            model_path: HuggingFace model path for E5.
        """
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        from flashrag.retriever.utils import load_corpus, parse_query
        
        self.parse_query = parse_query
        
        # Default paths
        if index_path is None:
            index_path = str(PROJECT_ROOT / "data/indexes/custom_e5/e5_Flat.index")
        if corpus_path is None:
            corpus_path = str(PROJECT_ROOT / "data/corpus/custom/combined_corpus.jsonl")
        
        # Determine device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"Loading E5 model from {model_path} on {device.upper()}...")
        self.model = SentenceTransformer(
            model_path, 
            trust_remote_code=True, 
            model_kwargs={"torch_dtype": torch.float16},
            device=device
        )
        self.device = device
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Index contains {self.index.ntotal} vectors")
        
        print(f"Loading corpus from {corpus_path}...")
        self.corpus = load_corpus(corpus_path)
        print(f"Corpus contains {len(self.corpus)} documents")
        
        self._retrieval_count = 0
    
    def retrieve(self, queries: list, topk: int = 5) -> list:
        """
        Retrieve top-k documents for each query.
        
        Args:
            queries: List of query strings
            topk: Number of documents to retrieve
            
        Returns:
            List of document lists, one per query
        """
        # Parse queries with E5 prefix
        parsed_queries = self.parse_query('e5', queries, 'query: ', is_query=True)
        
        # Encode queries
        emb = self.model.encode(
            parsed_queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = emb.astype(np.float32, order="C")
        
        # Search
        scores, ids = self.index.search(emb, topk)
        
        # Get documents
        results = []
        for query_ids, query_scores in zip(ids, scores):
            docs = []
            for idx, score in zip(query_ids, query_scores):
                doc = dict(self.corpus[idx])
                doc['retrieval_score'] = float(score)
                docs.append(doc)
            results.append(docs)
        
        self._retrieval_count += len(queries)
        return results
    
    @property
    def retrieval_count(self) -> int:
        """Total number of retrieval calls made."""
        return self._retrieval_count
    
    def reset_count(self):
        """Reset the retrieval counter."""
        self._retrieval_count = 0


class GeneratorWrapper:
    """
    Wrapper for LLM generation using OpenAI API.

    Supports both with-retrieval and without-retrieval generation modes.
    Includes automatic retry with exponential backoff for rate limits.
    Includes cost tracking for API usage.
    """

    PRICING = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        "claude-3-5-haiku-20241022": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
        "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    }

    ANTHROPIC_MODELS = {"claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"}

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_ollama: bool = False,
        ollama_model: str = "llama3.1:8b-instruct-q4_K_M",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        max_context_tokens: int = 120000  # Leave room for question and system prompt
    ):
        """
        Initialize the generator.

        Args:
            model: Model name (OpenAI or Anthropic)
            use_ollama: Whether to use local Ollama instead
            ollama_model: Ollama model name if use_ollama is True
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Base delay between retries (exponential backoff)
            max_context_tokens: Maximum tokens for context (default 120k, leaving room for prompts)
        """
        self.use_ollama = use_ollama
        self.use_anthropic = model in self.ANTHROPIC_MODELS
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_context_tokens = max_context_tokens

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

        if use_ollama:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            self.model = ollama_model
        elif self.use_anthropic:
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model
        else:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model

    @property
    def total_cost(self) -> float:
        """Calculate total cost in USD based on token usage."""
        if self.use_ollama:
            return 0.0  # Ollama is free

        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o-mini"])
        input_cost = self._total_input_tokens * pricing["input"]
        output_cost = self._total_output_tokens * pricing["output"]
        return input_cost + output_cost

    @property
    def usage_stats(self) -> dict:
        """Get detailed usage statistics."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd": self.total_cost,
            "model": self.model,
            "is_local": self.use_ollama
        }

    def reset_usage(self):
        """Reset usage tracking counters."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def print_usage_summary(self):
        """Print a summary of API usage and costs."""
        stats = self.usage_stats
        print(f"\n{'='*50}")
        print("API USAGE SUMMARY")
        print(f"{'='*50}")
        print(f"Model: {stats['model']} {'(local)' if stats['is_local'] else '(API)'}")
        print(f"Total API calls: {stats['total_calls']:,}")
        print(f"Input tokens: {stats['total_input_tokens']:,}")
        print(f"Output tokens: {stats['total_output_tokens']:,}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        if not stats['is_local']:
            print(f"Estimated cost: ${stats['total_cost_usd']:.4f}")
        else:
            print("Cost: $0.00 (local model)")
        print(f"{'='*50}\n")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: ~4 chars per token).
        
        For more accurate counting, could use tiktoken, but this is faster.
        """
        return len(text) // 4
    
    def _truncate_documents(self, docs: list, question: str, max_tokens: int) -> list:
        """
        Truncate documents to fit within token limit.
        
        Strategy:
        1. Sort by retrieval score (highest first)
        2. Keep documents in order until we hit the limit
        3. For the last document that would exceed limit, truncate it
        
        Args:
            docs: List of document dicts with 'contents', 'title', 'retrieval_score'
            question: The question (to estimate prompt overhead)
            max_tokens: Maximum tokens available for documents
            
        Returns:
            Truncated list of documents
        """
        # Estimate tokens for question and system prompt overhead (~500 tokens)
        prompt_overhead = 500 + self._estimate_tokens(question)
        available_tokens = max_tokens - prompt_overhead
        
        if available_tokens <= 0:
            # If even the prompt overhead exceeds limit, return empty
            return []
        
        # Sort by retrieval score (highest first) if available
        sorted_docs = sorted(
            docs,
            key=lambda d: d.get('retrieval_score', 0.0),
            reverse=True
        )
        
        truncated_docs = []
        current_tokens = 0
        
        for doc in sorted_docs:
            title = doc.get("title", "Unknown")
            contents = doc.get("contents", "")
            
            # Estimate tokens for this document (with formatting)
            doc_text = f"Doc {len(truncated_docs)+1}(Title: {title}) {contents}\n\n"
            doc_tokens = self._estimate_tokens(doc_text)
            
            if current_tokens + doc_tokens <= available_tokens:
                # Can fit entire document
                truncated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # Need to truncate this document
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only include if we have meaningful space
                    # Truncate contents to fit
                    # Reserve ~50 tokens for title/formatting
                    content_tokens = remaining_tokens - 50
                    if content_tokens > 0:
                        # Rough char limit: tokens * 4
                        max_chars = content_tokens * 4
                        truncated_contents = contents[:max_chars]
                        
                        truncated_doc = dict(doc)
                        truncated_doc["contents"] = truncated_contents
                        truncated_docs.append(truncated_doc)
                break
        
        return truncated_docs
    
    def _call_with_retry(self, messages: list, max_tokens: int = 256, temperature: float = 0.0,
                         docs: list = None, question: str = None) -> str:
        """
        Make API call with retry logic for rate limits and context length errors.

        Args:
            messages: List of message dicts for the API
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            docs: Original documents (for truncation if context exceeded)
            question: Original question (for truncation if context exceeded)
        """
        import time

        for attempt in range(self.max_retries):
            try:
                if self.use_anthropic:
                    # Convert OpenAI-style messages to Anthropic format
                    system_msg = ""
                    anthropic_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_msg = msg["content"]
                        else:
                            anthropic_messages.append(msg)

                    kwargs = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if system_msg:
                        kwargs["system"] = system_msg

                    response = self.client.messages.create(**kwargs)

                    self._total_calls += 1
                    self._total_input_tokens += response.usage.input_tokens
                    self._total_output_tokens += response.usage.output_tokens

                    return response.content[0].text
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    # Track token usage for cost estimation
                    self._total_calls += 1
                    if hasattr(response, 'usage') and response.usage:
                        self._total_input_tokens += response.usage.prompt_tokens
                        self._total_output_tokens += response.usage.completion_tokens

                    return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limits
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"\n⚠️  Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                    continue
                
                # Handle context length exceeded
                if "context_length" in error_str.lower() or "context_length_exceeded" in error_str.lower():
                    if docs is not None and question is not None and attempt < 2:
                        # Try truncating documents (more aggressive on retry)
                        print(f"\n⚠️  Context length exceeded, truncating documents (attempt {attempt + 1})...")
                        # Use more aggressive truncation: reduce by 20% each retry
                        aggressive_limit = int(self.max_context_tokens * (0.8 ** attempt))
                        truncated_docs = self._truncate_documents(docs, question, aggressive_limit)
                        
                        if len(truncated_docs) == 0:
                            # Even after truncation, can't fit. Return a fallback answer.
                            print("⚠️  Cannot fit any documents, generating without context...")
                            return self.generate_direct(question)
                        
                        # Rebuild messages with truncated docs
                        context = ""
                        for i, doc in enumerate(truncated_docs):
                            title = doc.get("title", "Unknown")
                            contents = doc.get("contents", "")
                            context += f"Doc {i+1}(Title: {title}) {contents}\n\n"
                        
                        system_prompt = f"""Answer the question based on the given documents. Only give me the answer and do not output any other words.
The following are given documents.

{context}"""
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Question: {question}"}
                        ]
                        continue  # Retry with truncated docs
                    else:
                        # Already tried truncation multiple times or no docs available, fallback
                        print("⚠️  Context still too long after truncation, generating without context...")
                        if question:
                            return self.generate_direct(question)
                        raise
                
                # Other errors: raise immediately
                raise
        
        # All retries exhausted
        raise Exception(f"API call failed after {self.max_retries} retries")
    
    def generate_with_retrieval(self, question: str, docs: list) -> str:
        """
        Generate answer using retrieved documents as context.
        
        Args:
            question: The question to answer
            docs: List of retrieved documents
            
        Returns:
            Generated answer string
        """
        # Pre-truncate documents to avoid context length issues
        truncated_docs = self._truncate_documents(docs, question, self.max_context_tokens)
        
        if len(truncated_docs) == 0:
            # No documents can fit, fall back to direct generation
            print("⚠️  No documents fit in context, generating without retrieval...")
            return self.generate_direct(question)
        
        # Build context from truncated docs
        context = ""
        for i, doc in enumerate(truncated_docs):
            title = doc.get("title", "Unknown")
            contents = doc.get("contents", "")
            context += f"Doc {i+1}(Title: {title}) {contents}\n\n"
        
        system_prompt = f"""Answer the question based on the given documents. Only give me the answer and do not output any other words.
The following are given documents.

{context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        # Pass docs and question for potential further truncation if still too long
        return self._call_with_retry(messages, docs=docs, question=question)
    
    def generate_direct(self, question: str) -> str:
        """
        Generate answer directly without retrieval (from model's knowledge).
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer string
        """
        system_prompt = """You are a knowledgeable assistant. Answer the question directly using your knowledge. 
Only give me the answer and do not output any other words. Be concise."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        return self._call_with_retry(messages)


class RAGPipeline:
    """
    Complete RAG pipeline combining retriever and generator.
    
    This class is controlled by the RL agent's retrieval decisions.
    """
    
    def __init__(
        self,
        retriever: Optional[DenseRetrieverWrapper] = None,
        generator: Optional[GeneratorWrapper] = None,
        topk: int = 5,
        generator_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Dense retriever instance (created if None)
            generator: Generator instance (created if None)
            topk: Default number of documents to retrieve
            generator_model: Model name for generator
        """
        self.retriever = retriever or DenseRetrieverWrapper()
        self.generator = generator or GeneratorWrapper(model=generator_model)
        self.topk = topk
    
    def answer(self, question: str, should_retrieve: bool = True) -> tuple:
        """
        Answer a question with optional retrieval.
        
        Args:
            question: The question to answer
            should_retrieve: Whether to retrieve documents
            
        Returns:
            Tuple of (answer, retrieved_docs or None)
        """
        if should_retrieve:
            docs = self.retriever.retrieve([question], topk=self.topk)[0]
            answer = self.generator.generate_with_retrieval(question, docs)
            return answer, docs
        else:
            answer = self.generator.generate_direct(question)
            return answer, None

