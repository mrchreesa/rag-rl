"""
Extended Metrics Analysis for RAG Experiment Results

Computes additional metrics from existing prediction files (NO re-runs needed):
1. Token Precision & Recall (separate from F1)
2. Sub-ExactMatch (substring containment)
3. Retrieval Recall@k (do retrieved docs contain the answer?)
4. Retrieval Precision@k (fraction of retrieved docs containing the answer)
5. CountToken (average input tokens per query at each k)

All metrics use FlashRAG's normalize_answer for consistency.

Usage:
    python experiments/scripts/analysis/extended_metrics_analysis.py \
        --results-dir experiments/results/controlled_comparison_custom_20260219_105226

    # With W&B logging
    python experiments/scripts/analysis/extended_metrics_analysis.py \
        --results-dir experiments/results/controlled_comparison_custom_20260219_105226 --wandb
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
import string
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


# ── normalize_answer (from FlashRAG evaluator/utils.py) ─────────────────────
def normalize_answer(s):
    """Lower text, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ── Metric functions ─────────────────────────────────────────────────────────

def token_level_scores(prediction: str, golden_answers: list) -> dict:
    """
    Compute token-level F1, Precision, and Recall.
    Matches FlashRAG's F1_Score.token_level_scores exactly.
    """
    final = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    for gold in golden_answers:
        norm_pred = normalize_answer(prediction)
        norm_gold = normalize_answer(gold)
        if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
            continue
        if norm_gold in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
            continue
        pred_tokens = norm_pred.split()
        gold_tokens = norm_gold.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(gold_tokens) if gold_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        for k in ["f1", "precision", "recall"]:
            final[k] = max(locals()[k], final[k])
    return final


def sub_exact_match(prediction: str, golden_answers: list) -> float:
    """
    Sub-ExactMatch: does the prediction contain any gold answer as a substring?
    Matches FlashRAG's Sub_ExactMatch.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    for gold in golden_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold in norm_pred:
            return 1.0
    return 0.0


def retrieval_recall(context: str, golden_answers: list) -> float:
    """
    Retrieval Recall: does ANY part of the retrieved context contain the gold answer?
    Returns 1 if at least one gold answer found in context, else 0.
    """
    if not context:
        return 0.0
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_context = normalize_answer(context)
    for gold in golden_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold and norm_gold in norm_context:
            return 1.0
    return 0.0


def retrieval_precision_per_doc(retrieved_text: str, golden_answers: list) -> float:
    """
    Retrieval Precision: what fraction of retrieved document paragraphs
    contain the gold answer? Uses double-newline as document separator.
    """
    if not retrieved_text:
        return 0.0
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    # Split on double-newline (document boundary in our format)
    docs = [d.strip() for d in retrieved_text.split("\n\n") if d.strip()]
    if not docs:
        return 0.0
    hits = 0
    for doc in docs:
        norm_doc = normalize_answer(doc)
        for gold in golden_answers:
            norm_gold = normalize_answer(gold)
            if norm_gold and norm_gold in norm_doc:
                hits += 1
                break
    return hits / len(docs)


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken (GPT-4 tokenizer). Falls back to whitespace."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


# ── Loading ──────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> list:
    """Load per-sample predictions from JSONL."""
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_config(preds: list, config_name: str) -> dict:
    """Compute all extended metrics for a single configuration."""
    n = len(preds)

    # Check if retrieved_text is available (new format) vs only metadata.context (old format)
    has_retrieved_text = any(p.get("retrieved_text") for p in preds)

    # Token-level metrics
    precisions, recalls, f1s = [], [], []
    sub_ems = []
    ret_recalls = []
    ret_precisions = []
    token_counts = []

    for p in preds:
        prediction = p.get("prediction", "")
        golden = p.get("golden_answers", [])
        topk = p.get("topk_used", 0)

        # Use retrieved_text (from new controlled_comparison.py) or fall back to metadata.context
        retrieved_text = p.get("retrieved_text", "")
        if not retrieved_text:
            retrieved_text = p.get("metadata", {}).get("context", "")

        # Token Precision / Recall / F1
        scores = token_level_scores(prediction, golden)
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])

        # Sub-ExactMatch
        sub_ems.append(sub_exact_match(prediction, golden))

        # Retrieval Recall (does retrieved context contain the answer?)
        if topk > 0 and retrieved_text:
            ret_recalls.append(retrieval_recall(retrieved_text, golden))
            ret_precisions.append(retrieval_precision_per_doc(retrieved_text, golden))
        else:
            ret_recalls.append(0.0)
            ret_precisions.append(0.0)

        # Token count: use actual retrieved text if available
        question = p.get("question", "")
        if has_retrieved_text and retrieved_text:
            prompt_text = question + " " + retrieved_text
        elif topk > 0 and retrieved_text:
            prompt_text = question + " " + retrieved_text
        else:
            prompt_text = question
        token_counts.append(count_tokens_tiktoken(prompt_text))

    result = {
        "config": config_name,
        "n_samples": n,
        "has_retrieved_text": has_retrieved_text,
        "avg_f1": float(np.mean(f1s)),
        "avg_precision": float(np.mean(precisions)),
        "avg_recall": float(np.mean(recalls)),
        "avg_sub_em": float(np.mean(sub_ems)),
        "avg_retrieval_recall": float(np.mean(ret_recalls)),
        "avg_retrieval_precision": float(np.mean(ret_precisions)),
        "avg_input_tokens": float(np.mean(token_counts)),
        "total_input_tokens": int(sum(token_counts)),
        # Per-sample lists for statistical tests
        "per_sample_precision": [float(x) for x in precisions],
        "per_sample_recall": [float(x) for x in recalls],
        "per_sample_sub_em": [float(x) for x in sub_ems],
        "per_sample_ret_recall": [float(x) for x in ret_recalls],
        "per_sample_ret_precision": [float(x) for x in ret_precisions],
        "per_sample_tokens": token_counts,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Extended metrics analysis from prediction files")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to controlled comparison results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to results-dir)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to W&B")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXTENDED METRICS ANALYSIS (Post-Hoc)")
    print("=" * 70)
    print(f"Results dir: {results_dir}")

    # Load all prediction files
    pred_files = sorted(results_dir.glob("predictions_*.jsonl"))
    if not pred_files:
        print(f"ERROR: No prediction files found in {results_dir}")
        return

    # W&B init
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="rl-rag-enhanced",
                name=f"extended_metrics_{results_dir.name}",
                tags=["analysis", "extended-metrics"],
                config={"results_dir": str(results_dir)},
            )
        except Exception as e:
            print(f"W&B init failed: {e}")

    all_results = []
    for pf in pred_files:
        config_name = pf.stem.replace("predictions_", "")
        preds = load_predictions(str(pf))
        print(f"\n  Loaded {len(preds)} predictions for {config_name}")

        result = analyze_config(preds, config_name)
        all_results.append(result)

        print(f"    F1:               {result['avg_f1']:.4f}")
        print(f"    Token Precision:  {result['avg_precision']:.4f}")
        print(f"    Token Recall:     {result['avg_recall']:.4f}")
        print(f"    Sub-ExactMatch:   {result['avg_sub_em']:.4f}")
        print(f"    Retrieval Recall: {result['avg_retrieval_recall']:.4f}")
        print(f"    Avg Input Tokens: {result['avg_input_tokens']:.0f}")

        print(f"    Retrieval Prec:   {result['avg_retrieval_precision']:.4f}")

        # Log to W&B
        if wandb_run:
            import wandb
            wandb.log({
                f"extended/{config_name}/f1": result["avg_f1"],
                f"extended/{config_name}/precision": result["avg_precision"],
                f"extended/{config_name}/recall": result["avg_recall"],
                f"extended/{config_name}/sub_em": result["avg_sub_em"],
                f"extended/{config_name}/retrieval_recall": result["avg_retrieval_recall"],
                f"extended/{config_name}/retrieval_precision": result["avg_retrieval_precision"],
                f"extended/{config_name}/avg_input_tokens": result["avg_input_tokens"],
            })

    # ── Note about data format ─────────────────────────────────────────
    if all_results and not all_results[0].get("has_retrieved_text"):
        print("\n  NOTE: Predictions do not contain retrieved_text field.")
        print("  Retrieval Recall/Precision and token counts are based on dataset metadata.context")
        print("  (same for all configs). Re-run controlled_comparison.py for accurate per-config values.")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n" + "=" * 115)
    print("EXTENDED METRICS SUMMARY")
    print("=" * 115)
    header = f"{'Config':<25}{'F1':<10}{'Prec':<10}{'Recall':<10}{'SubEM':<10}{'RetRecall':<12}{'RetPrec':<10}{'AvgTokens'}"
    print(header)
    print("-" * 115)
    for r in all_results:
        print(f"{r['config']:<25}{r['avg_f1']:<10.4f}{r['avg_precision']:<10.4f}"
              f"{r['avg_recall']:<10.4f}{r['avg_sub_em']:<10.4f}"
              f"{r['avg_retrieval_recall']:<12.4f}{r['avg_retrieval_precision']:<10.4f}"
              f"{r['avg_input_tokens']:<.0f}")

    # ── Token efficiency table ───────────────────────────────────────────
    print("\n\nTOKEN EFFICIENCY")
    print("-" * 70)
    print(f"{'Config':<25}{'F1':<10}{'AvgTokens':<12}{'F1/1K Tokens'}")
    print("-" * 70)
    for r in all_results:
        f1_per_1k = (r['avg_f1'] / r['avg_input_tokens'] * 1000) if r['avg_input_tokens'] > 0 else 0
        print(f"{r['config']:<25}{r['avg_f1']:<10.4f}{r['avg_input_tokens']:<12.0f}{f1_per_1k:.4f}")

    # ── W&B summary table ────────────────────────────────────────────────
    if wandb_run:
        import wandb
        columns = ["Config", "F1", "Precision", "Recall", "Sub-EM",
                    "Retrieval Recall", "Retrieval Prec", "Avg Tokens", "F1/1K Tokens"]
        table = wandb.Table(columns=columns)
        for r in all_results:
            f1_per_1k = (r['avg_f1'] / r['avg_input_tokens'] * 1000) if r['avg_input_tokens'] > 0 else 0
            table.add_data(
                r["config"],
                round(r["avg_f1"], 4),
                round(r["avg_precision"], 4),
                round(r["avg_recall"], 4),
                round(r["avg_sub_em"], 4),
                round(r["avg_retrieval_recall"], 4),
                round(r["avg_retrieval_precision"], 4),
                round(r["avg_input_tokens"]),
                round(f1_per_1k, 4),
            )
        wandb.log({"extended_metrics_table": table})
        wandb.finish()

    # ── Save results (without per-sample lists for cleaner JSON) ─────────
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items()
                  if not k.startswith("per_sample_") and k != "has_retrieved_text"}
        save_r["f1_per_1k_tokens"] = (r['avg_f1'] / r['avg_input_tokens'] * 1000) if r['avg_input_tokens'] > 0 else 0
        save_results.append(save_r)

    out_path = output_dir / "extended_metrics.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Save per-sample details for further statistical analysis ─────────
    detail_path = output_dir / "extended_metrics_per_sample.json"
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved per-sample: {detail_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
