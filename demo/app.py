"""
RL-Optimized RAG System — Dissertation Demo

Interactive Gradio demo for the dissertation panel presentation.
Supports text/voice input, RL vs baseline comparison, and voice output.

Usage:
    python demo/app.py                    # Local only
    python demo/app.py --share            # Public URL for panel
    python demo/app.py --checkpoint PATH  # Specific RL checkpoint
    python demo/app.py --no-voice         # Disable voice features
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path

# Fix OpenMP conflict (must be before torch import)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Project paths
DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
AGENTS_PATH = PROJECT_ROOT / 'src' / 'agents'
FLASHRAG_PATH = PROJECT_ROOT / 'src' / 'rag' / 'FlashRAG'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(FLASHRAG_PATH))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Load environment variables
from utils.env_loader import load_env_file
load_env_file(PROJECT_ROOT)

import gradio as gr
import torch

from agents.flashrag_components import DenseRetrieverWrapper, GeneratorWrapper, RAGPipeline
from agents.enhanced_pipeline import EnhancedRAGPipeline
from agents.reward import RAGRewardCalculator, compute_f1, compute_exact_match
from agents.dataset import load_custom_dataset, load_hotpotqa

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------

rl_pipeline: EnhancedRAGPipeline = None
baseline_pipeline: RAGPipeline = None
reward_calc: RAGRewardCalculator = None
cached_examples: list = []
test_dataset: list = []
VOICE_ENABLED = True
CHECKPOINT_PATH = None

# Latest checkpoint by default
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "experiments" / "results" / "rl_enhanced_20260208_223508" / "best_model.pt"
)


def find_latest_checkpoint() -> str:
    """Find the most recent RL checkpoint."""
    results_dir = PROJECT_ROOT / "experiments" / "results"
    checkpoints = sorted(results_dir.glob("rl_enhanced_*/best_model.pt"))
    if checkpoints:
        return str(checkpoints[-1])
    return DEFAULT_CHECKPOINT


def init_pipelines(checkpoint_path: str = None):
    """Initialize RL and baseline pipelines (called once at startup)."""
    global rl_pipeline, baseline_pipeline, reward_calc, cached_examples, test_dataset

    print("Initializing pipelines...")

    # Shared components
    retriever = DenseRetrieverWrapper()
    generator = GeneratorWrapper(model="gpt-4o-mini")

    # --- RL Pipeline ---
    rl_pipeline = EnhancedRAGPipeline(
        retriever=retriever,
        generator=generator,
        topk=10,
        use_query_rewriter=False,
        use_learned_retrieval=True,
        use_ollama=False,
    )

    # Load trained checkpoint
    cp = checkpoint_path or find_latest_checkpoint()
    if Path(cp).exists():
        print(f"Loading checkpoint: {cp}")
        checkpoint = torch.load(cp, map_location="cpu")
        config = checkpoint.get("config", {})

        # Rebuild policy if checkpoint used dynamic topk
        if config.get("use_dynamic_topk", False):
            from agents.enhanced_pipeline import DynamicTopKPolicyNetwork
            topk_options = config.get("topk_options", [0, 1, 3, 5, 7, 10])
            input_dim = 768
            if config.get("use_difficulty_features", False):
                input_dim += 5
            rl_pipeline.policy_network = DynamicTopKPolicyNetwork(
                input_dim=input_dim,
                topk_options=topk_options,
                use_difficulty_features=config.get("use_difficulty_features", False),
            )
            rl_pipeline.use_dynamic_topk = True
            rl_pipeline.topk_options = topk_options

        rl_pipeline.policy_network.load_state_dict(checkpoint["policy_network"])
        rl_pipeline.policy_network.eval()
        print("  Checkpoint loaded.")
    else:
        print(f"  WARNING: No checkpoint at {cp} — policy is untrained.")

    # --- Baseline Pipeline (always retrieves, topk=10 matches best baseline) ---
    baseline_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        topk=10,
    )

    # --- Reward Calculator (matches Dynamic TopK training params) ---
    reward_calc = RAGRewardCalculator(
        retrieval_cost=0.05,
        correct_no_retrieval_bonus=0.1,
        wrong_no_retrieval_penalty=0.3,
        use_f1=True,
        f1_threshold_for_correct=0.5,
        use_dynamic_cost=True,
        base_retrieval_cost=0.05,
        per_doc_cost=0.01,
    )

    # --- Load cached examples ---
    examples_path = DEMO_DIR / "examples.json"
    if examples_path.exists():
        with open(examples_path) as f:
            cached_examples = json.load(f)
        print(f"  Loaded {len(cached_examples)} cached examples.")

    # --- Load test dataset for ground truth lookup ---
    try:
        ds = load_custom_dataset(split="test")
        test_dataset = list(ds)
        print(f"  Loaded {len(test_dataset)} test samples.")
    except Exception as e:
        print(f"  Could not load test dataset: {e}")
        test_dataset = []

    print("Pipelines ready.\n")


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------

def find_ground_truth(question: str) -> list:
    """Look up ground truth answers for a question in the test set."""
    q_lower = question.strip().lower()
    for item in test_dataset:
        if item["question"].strip().lower() == q_lower:
            return item.get("golden_answers", [])
    for ex in cached_examples:
        if ex["question"].strip().lower() == q_lower:
            return ex.get("golden_answers", [])
    return []


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    if not audio_path:
        return ""
    try:
        import openai
        client = openai.OpenAI()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f
            )
        return transcript.text
    except Exception as e:
        return f"[Transcription error: {e}]"


def format_docs(docs: list) -> str:
    """Format retrieved documents for display."""
    if not docs:
        return "_No documents retrieved._"
    lines = []
    for i, doc in enumerate(docs, 1):
        score = doc.get("score", 0)
        content = doc.get("contents", doc.get("content", ""))
        # Truncate long documents
        if len(content) > 400:
            content = content[:400] + "..."
        score_str = f" (score: {score:.3f})" if score else ""
        lines.append(f"**Doc {i}**{score_str}\n{content}")
    return "\n\n---\n\n".join(lines)


def format_policy_output(metadata: dict) -> str:
    """Format the raw policy network output for display."""
    lines = []
    if "retrieval_probability" in metadata and metadata["retrieval_probability"] is not None:
        lines.append(f"**Retrieval Probability**: {metadata['retrieval_probability']:.1%}")
    if "topk_distribution" in metadata and metadata["topk_distribution"]:
        lines.append("**TopK Distribution**:")
        for k, prob in sorted(metadata["topk_distribution"].items()):
            bar = "#" * int(prob * 30)
            lines.append(f"  k={k}: {prob:.1%} {bar}")
    if "entropy" in metadata and metadata["entropy"] is not None:
        ent = metadata["entropy"]
        if isinstance(ent, torch.Tensor):
            ent = ent.item()
        lines.append(f"**Entropy**: {ent:.4f}")
    if "did_retrieve" in metadata:
        lines.append(f"**Decision**: {'RETRIEVE' if metadata['did_retrieve'] else 'GENERATE DIRECTLY'}")
    if "topk_used" in metadata:
        lines.append(f"**TopK Used**: {metadata['topk_used']}")
    return "\n".join(lines) if lines else "_No policy output (policy not loaded)._"


def run_demo(
    question: str,
    audio: str,
    dataset_choice: str,
):
    """
    Main demo function — runs both RL agent and baseline, returns all outputs.

    Returns a tuple of values for each Gradio output component.
    """
    # 1. Handle voice input
    if audio and (not question or not question.strip()):
        if VOICE_ENABLED:
            question = transcribe_audio(audio)
        else:
            question = "[Voice disabled]"

    if not question or not question.strip():
        empty = "Please enter a question or record audio."
        return (
            empty,   # question_display
            "",      # rl_answer
            "",      # baseline_answer
            "",      # rl_policy_md
            "",      # efficiency_md
            "",      # metrics_md
            "",      # docs_md
            "",      # ground_truth_md
            "",      # status
        )

    question = question.strip()

    # 2. Look up ground truth
    ground_truths = find_ground_truth(question)

    # 3. Run RL pipeline
    status_msg = "Running RL agent..."
    try:
        with torch.no_grad():
            rl_answer, rl_docs, rl_meta = rl_pipeline.answer(
                question, deterministic=True, temperature=0.7
            )
    except Exception as e:
        rl_answer = f"[Error: {e}]"
        rl_docs = []
        rl_meta = {}

    # 4. Run baseline pipeline (always retrieves)
    status_msg = "Running baseline..."
    try:
        baseline_answer, baseline_docs = baseline_pipeline.answer(
            question, should_retrieve=True
        )
    except Exception as e:
        baseline_answer = f"[Error: {e}]"
        baseline_docs = []

    # 5. Compute metrics
    rl_did_retrieve = rl_meta.get("did_retrieve", False)
    rl_topk = rl_meta.get("topk_used", 0)
    rl_retrieval_prob = rl_meta.get("retrieval_probability", None)

    rl_f1 = compute_f1(rl_answer, ground_truths) if ground_truths else None
    rl_em = compute_exact_match(rl_answer, ground_truths) if ground_truths else None
    bl_f1 = compute_f1(baseline_answer, ground_truths) if ground_truths else None
    bl_em = compute_exact_match(baseline_answer, ground_truths) if ground_truths else None

    rl_reward = None
    if ground_truths:
        rl_reward, _ = reward_calc.compute_reward(
            rl_answer, ground_truths, rl_did_retrieve, topk_used=rl_topk
        )

    # 6. Build output strings

    # Question display (shows transcription if voice was used)
    question_display = question

    # RL Policy Decision panel
    policy_md = format_policy_output(rl_meta)

    # Efficiency comparison (prominent)
    baseline_topk = 10
    baseline_cost = 0.05 + 0.01 * baseline_topk  # 0.15
    rl_cost = 0.0
    if rl_did_retrieve and rl_topk > 0:
        rl_cost = 0.05 + 0.01 * rl_topk
    docs_saved = baseline_topk - rl_topk
    pct_saved = (docs_saved / baseline_topk * 100) if baseline_topk > 0 else 0
    cost_saved = ((baseline_cost - rl_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

    efficiency_lines = [
        f"| | RL Agent | Baseline | Saving |",
        f"|---|---|---|---|",
        f"| **Documents Retrieved** | **{rl_topk}** | {baseline_topk} | **{docs_saved} fewer ({pct_saved:.0f}%)** |",
        f"| **Retrieval Cost** | **{rl_cost:.2f}** | {baseline_cost:.2f} | **{cost_saved:.0f}% lower** |",
    ]
    if rl_answer.strip() == baseline_answer.strip():
        efficiency_lines.append("")
        efficiency_lines.append("*Same answer quality with fewer documents — the RL agent learned to be efficient.*")
    efficiency_md = "\n".join(efficiency_lines)

    # Metrics comparison table
    metrics_lines = ["| Metric | RL Agent | Baseline (always retrieve) |",
                     "|--------|----------|---------------------------|"]
    if rl_f1 is not None:
        metrics_lines.append(f"| **F1** | {rl_f1:.4f} | {bl_f1:.4f} |")
    if rl_em is not None:
        metrics_lines.append(f"| **EM** | {rl_em:.1f} | {bl_em:.1f} |")
    metrics_lines.append(
        f"| **Retrieved** | {'Yes' if rl_did_retrieve else 'No'} | Yes |"
    )
    metrics_lines.append(f"| **TopK Used** | {rl_topk} | {baseline_topk} |")
    if rl_reward is not None:
        metrics_lines.append(f"| **Reward** | {rl_reward:.4f} | — |")
    if not ground_truths:
        metrics_lines.append("")
        metrics_lines.append("_No ground truth available — F1/EM cannot be computed._")
    metrics_md = "\n".join(metrics_lines)

    # Retrieved documents — show both side by side
    rl_docs_count = len(rl_docs) if rl_docs else 0
    bl_docs_count = len(baseline_docs) if baseline_docs else 0
    docs_parts = []
    docs_parts.append(f"### RL Agent — {rl_docs_count} document(s) retrieved\n")
    if rl_did_retrieve and rl_docs:
        docs_parts.append(format_docs(rl_docs))
    elif not rl_did_retrieve:
        docs_parts.append("_RL agent chose not to retrieve — answered from LLM knowledge._")
    else:
        docs_parts.append("_No documents._")
    docs_parts.append(f"\n\n---\n\n### Baseline — {bl_docs_count} document(s) retrieved\n")
    docs_parts.append(format_docs(baseline_docs))
    docs_md = "\n".join(docs_parts)

    # Ground truth
    if ground_truths:
        gt_md = "\n\n".join(f"- {a}" for a in ground_truths)
    else:
        gt_md = "_Not available for this question (not in test set)._"

    status_msg = "Done."

    return (
        question_display,
        rl_answer,
        baseline_answer,
        policy_md,
        efficiency_md,
        metrics_md,
        docs_md,
        gt_md,
        status_msg,
    )


def get_random_example():
    """Pick a random example question from the cached set."""
    if not cached_examples:
        return "No examples loaded."
    ex = random.choice(cached_examples)
    return ex["question"]


def get_example_questions() -> list:
    """Return list of example question strings for Gradio Examples component."""
    return [[ex["question"]] for ex in cached_examples] if cached_examples else []


# ---------------------------------------------------------------------------
# TTS helper — uses browser Web Speech API via JavaScript
# ---------------------------------------------------------------------------

TTS_JS = """
async (text) => {
    if (!text || text.trim() === '') return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}
"""

STOP_TTS_JS = """
async () => {
    window.speechSynthesis.cancel();
}
"""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(
        title="RL-Optimized RAG System",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    ) as demo:
        # Header
        gr.Markdown(
            "# RL-Optimized RAG System\n"
            "### Dissertation Demo — Retrieval-Augmented Generation with Reinforcement Learning\n"
            "The RL agent learns *how many documents to retrieve* per query (Dynamic TopK). "
            "It achieves **70% fewer retrievals** than the fixed topk=10 baseline while maintaining answer quality, "
            "by selecting from [0, 1, 3, 5, 7, 10] documents based on query characteristics."
        )

        # ---- Input Section ----
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type a question about academic/research topics...",
                    lines=2,
                )
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Or Record Voice",
                    visible=VOICE_ENABLED,
                )

        with gr.Row():
            run_btn = gr.Button("Run Demo", variant="primary", scale=2)
            random_btn = gr.Button("Random Example", variant="secondary", scale=1)

        # Status
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)

        # ---- Transcription Display ----
        question_display = gr.Textbox(
            label="Question (after transcription)", interactive=False, visible=True
        )

        # ---- Efficiency Highlight ----
        gr.Markdown("### Efficiency Comparison")
        efficiency_output = gr.Markdown(value="")

        # ---- Main Output Section ----
        with gr.Row():
            # Left column: RL Policy Decision
            with gr.Column(scale=1):
                gr.Markdown("### RL Policy Decision")
                policy_output = gr.Markdown(value="")

            # Right column: Metrics Comparison
            with gr.Column(scale=2):
                gr.Markdown("### Quality Metrics")
                metrics_output = gr.Markdown(value="")

        # ---- Answers Side-by-Side ----
        with gr.Row():
            with gr.Column():
                gr.Markdown("### RL Agent Answer")
                rl_answer_box = gr.Textbox(
                    label="RL Agent", interactive=False, lines=5
                )
                with gr.Row():
                    speak_btn = gr.Button("Read Aloud", size="sm")
                    stop_btn = gr.Button("Stop", size="sm")

            with gr.Column():
                gr.Markdown("### Baseline Answer (always retrieve, topk=10)")
                baseline_answer_box = gr.Textbox(
                    label="Baseline", interactive=False, lines=5
                )

        # ---- Tabs: Documents / Policy Details / Ground Truth ----
        with gr.Tabs():
            with gr.TabItem("Retrieved Documents"):
                docs_output = gr.Markdown(value="")
            with gr.TabItem("Ground Truth"):
                gt_output = gr.Markdown(value="")

        # ---- Examples ----
        if cached_examples:
            gr.Examples(
                examples=get_example_questions(),
                inputs=[question_input],
                label="Example Questions (from test set)",
            )

        # ---- Wire up events ----

        # Main run
        run_btn.click(
            fn=run_demo,
            inputs=[question_input, audio_input, gr.State("custom")],
            outputs=[
                question_display,
                rl_answer_box,
                baseline_answer_box,
                policy_output,
                efficiency_output,
                metrics_output,
                docs_output,
                gt_output,
                status_box,
            ],
        )

        # Also run on Enter in the text box
        question_input.submit(
            fn=run_demo,
            inputs=[question_input, audio_input, gr.State("custom")],
            outputs=[
                question_display,
                rl_answer_box,
                baseline_answer_box,
                policy_output,
                efficiency_output,
                metrics_output,
                docs_output,
                gt_output,
                status_box,
            ],
        )

        # Random example
        random_btn.click(fn=get_random_example, outputs=[question_input])

        # TTS buttons
        speak_btn.click(None, inputs=[rl_answer_box], js=TTS_JS)
        stop_btn.click(None, js=STOP_TTS_JS)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RL-RAG Dissertation Demo")
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to RL checkpoint (default: latest)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice input/output features"
    )
    return parser.parse_args()


def main():
    global VOICE_ENABLED, CHECKPOINT_PATH

    args = parse_args()
    VOICE_ENABLED = not args.no_voice
    CHECKPOINT_PATH = args.checkpoint

    # Initialize pipelines
    init_pipelines(checkpoint_path=args.checkpoint)

    # Build and launch UI
    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
