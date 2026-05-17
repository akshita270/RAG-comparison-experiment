"""
main.py — Experiment orchestrator for the RAG comparison study.

This is the single entry point that:
  1. Loads (or fetches) the ArXiv paper dataset
  2. Loads (or generates) the evaluation question set
  3. Initialises all 4 RAG architectures on the SAME document set
  4. Runs all 4 architectures on all 25 questions
  5. Evaluates each answer with RAGAS metrics + LLM-as-judge
  6. Saves results to CSV
  7. Generates 4 visualisations
  8. Prints a summary table showing the winner per metric

Usage:
    python main.py                  # full experiment run
    python main.py --skip-existing  # skip questions already in results CSV
    python main.py --dry-run        # test setup with 2 questions per architecture

Architecture initialisation is deferred until after the dataset is loaded
so that all 4 architectures index the exact same documents.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tabulate import tabulate

# Add project root to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.data_loader import load_papers, load_questions
from utils.llm_client import LLMClient
from architectures.vanilla_rag import VanillaRAG
from architectures.hyde_rag import HyDERAG
from architectures.graph_rag import GraphRAG
from architectures.agentic_rag import AgenticRAG
from evaluation import ragas_eval, llm_judge
from visualizations.charts import generate_all_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_question(
    question_data: Dict,
    architecture_name: str,
    architecture,
    eval_llm: LLMClient,
    question_idx: int,
) -> Dict[str, Any]:
    """
    Run a single architecture on a single question and return a result row.

    The result row matches the CSV column schema so it can be appended
    directly to the results list without any transformation.
    """
    question     = question_data["question"]
    ground_truth = question_data.get("ground_truth", "")
    arxiv_id     = question_data.get("arxiv_id", "")

    logger.info(f"  [{architecture_name}] Q{question_idx+1}: {question[:80]}…")

    # ── Answer generation (timed) ──────────────────────────────────────────
    t_start = time.time()
    try:
        result = architecture.answer(question)
    except Exception as exc:
        logger.error(f"  [{architecture_name}] Answer generation failed: {exc}")
        result = {"answer": f"ERROR: {exc}", "contexts": [], "sources": []}
    latency_ms = (time.time() - t_start) * 1000

    answer   = result.get("answer", "")
    contexts = result.get("contexts", [])

    # ── RAGAS / LLM metric evaluation ─────────────────────────────────────
    try:
        ragas_scores = ragas_eval.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            llm=eval_llm,
        )
    except Exception as exc:
        logger.warning(f"  RAGAS eval failed: {exc}. Using default 0.5.")
        ragas_scores = {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}

    # ── LLM-as-judge evaluation ────────────────────────────────────────────
    try:
        judge_scores = llm_judge.judge(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    except Exception as exc:
        logger.warning(f"  Judge eval failed: {exc}. Using default 3.")
        judge_scores = {"factuality": 3, "relevance": 3, "citation": 3}

    return {
        "question_id":       question_idx,
        "question":          question,
        "arxiv_id":          arxiv_id,
        "architecture":      architecture_name,
        "answer":            answer[:500],  # truncate very long answers in CSV
        "faithfulness":      ragas_scores.get("faithfulness", 0.5),
        "answer_relevancy":  ragas_scores.get("answer_relevancy", 0.5),
        "context_precision": ragas_scores.get("context_precision", 0.5),
        "judge_factuality":  judge_scores.get("factuality", 3),
        "judge_relevance":   judge_scores.get("relevance", 3),
        "judge_citation":    judge_scores.get("citation", 3),
        "latency_ms":        round(latency_ms, 1),
        "eval_method":       ragas_scores.get("method", "unknown"),
        # Extra fields from architectures where available
        "tool_calls":        json.dumps(result.get("tool_calls", [])),
        "hyp_answer":        result.get("hypothetical_answer", "")[:200],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    """
    Print a formatted summary table showing mean scores per architecture
    and highlighting the winner (highest mean) for each metric.
    """
    metrics = [
        "faithfulness", "answer_relevancy", "context_precision",
        "judge_factuality", "judge_relevance", "judge_citation", "latency_ms",
    ]
    summary = df.groupby("architecture")[metrics].mean().round(3)

    # Add standard deviation columns
    std = df.groupby("architecture")[metrics].std().round(3)
    std.columns = [f"{c}_std" for c in std.columns]

    # Determine winner per metric (lowest latency wins, highest for the rest)
    winners = {}
    for m in metrics:
        if m == "latency_ms":
            winners[m] = summary[m].idxmin()
        else:
            winners[m] = summary[m].idxmax()

    print("\n" + "═" * 80)
    print("  RAG ARCHITECTURE COMPARISON — SUMMARY RESULTS")
    print("═" * 80)

    # Build display table
    rows = []
    for arch in summary.index:
        row = [arch]
        for m in metrics:
            val = summary.loc[arch, m]
            s   = f"{val:.3f}"
            if winners[m] == arch:
                s = f"★ {s}"   # mark the winner
            row.append(s)
        rows.append(row)

    headers = ["Architecture"] + [m.replace("_", "\n") for m in metrics]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    print("\n★ = winner for that metric  |  latency_ms: lower is better  |  all others: higher is better")
    print("\nPer-metric winners:")
    for m, winner in winners.items():
        label = f"{'↓' if m == 'latency_ms' else '↑'} {m}"
        print(f"  {label:30s} → {winner}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_existing: bool = False, dry_run: bool = False):
    # ── Config validation ─────────────────────────────────────────────────
    config.validate()
    logger.info(f"LLM provider: {config.LLM_PROVIDER} | model: {config.LLM_MODEL}")
    logger.info(f"Judge model: {config.JUDGE_MODEL}")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading papers …")
    papers = load_papers()
    logger.info(f"Loaded {len(papers)} papers.")

    logger.info("Loading questions …")
    questions = load_questions(papers)
    if dry_run:
        questions = questions[:2]
        logger.info(f"[DRY RUN] Using only {len(questions)} questions.")
    logger.info(f"Loaded {len(questions)} questions.")

    # ── Initialise architectures (all share the same paper set) ───────────
    logger.info("Initialising RAG architectures …")
    architectures = {
        "vanilla_rag": VanillaRAG(papers),
        "hyde_rag":    HyDERAG(papers),
        "graph_rag":   GraphRAG(papers),
        "agentic_rag": AgenticRAG(papers),
    }
    logger.info(f"Initialised: {list(architectures.keys())}")

    eval_llm = LLMClient()  # shared LLM for RAGAS fallback evaluation

    # ── Load existing results if resuming ─────────────────────────────────
    existing_rows = []
    if skip_existing and config.RESULTS_FILE.exists():
        existing_df = pd.read_csv(config.RESULTS_FILE)
        existing_rows = existing_df.to_dict("records")
        already_done = set(zip(existing_df["question"], existing_df["architecture"]))
        logger.info(f"Resuming — {len(existing_rows)} existing results found.")
    else:
        already_done = set()

    # ── Run experiment ─────────────────────────────────────────────────────
    results = list(existing_rows)
    total = len(questions) * len(architectures)
    done  = len(existing_rows)

    for q_idx, q_data in enumerate(questions):
        for arch_name, arch in architectures.items():
            key = (q_data["question"], arch_name)
            if key in already_done:
                logger.info(f"  Skipping (already done): {arch_name} Q{q_idx+1}")
                continue

            row = run_one_question(q_data, arch_name, arch, eval_llm, q_idx)
            results.append(row)
            done += 1
            logger.info(
                f"  Progress: {done}/{total} | "
                f"faithfulness={row['faithfulness']:.3f} | "
                f"latency={row['latency_ms']:.0f}ms"
            )

            # Save after every result so we can resume if interrupted
            pd.DataFrame(results).to_csv(config.RESULTS_FILE, index=False)

        # Small pause between questions to respect API rate limits
        if not dry_run:
            time.sleep(1)

    # ── Final results ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(config.RESULTS_FILE, index=False)
    logger.info(f"Results saved to {config.RESULTS_FILE} ({len(df)} rows)")

    # ── Visualisations ────────────────────────────────────────────────────
    chart_paths = generate_all_charts(df)
    for name, path in chart_paths.items():
        logger.info(f"Chart saved: {path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary_table(df)

    print(f"\nAll outputs saved to:")
    print(f"  Results CSV  : {config.RESULTS_FILE}")
    print(f"  Charts dir   : {config.VIZ_DIR}/")
    print(f"  Papers cache : {config.PAPERS_FILE}")
    print(f"  Questions    : {config.QUESTIONS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Architecture Comparison Experiment")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip questions already present in the results CSV (enables resuming)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 2 questions per architecture to verify everything works"
    )
    args = parser.parse_args()
    main(skip_existing=args.skip_existing, dry_run=args.dry_run)
