"""
evaluation/ragas_eval.py — RAGAS metrics computation with LLM-based fallback.

RAGAS (Retrieval Augmented Generation Assessment) provides three metrics:

  faithfulness      — Does the answer contain only information present
                      in the retrieved contexts? (hallucination detector)
  answer_relevancy  — Does the answer actually address the question?
  context_precision — Are the retrieved chunks relevant to the question?

We first attempt to use the RAGAS library directly. If RAGAS fails (version
mismatch, import error, API error), we fall back to our own LLM-based
implementations of the same three metrics. Both approaches yield 0-1 scores.

The fallback metrics are intentionally simple but interpretable:
  - We ask the LLM to score each dimension 0-10 and normalise to 0-1.
  - This is less rigorous than RAGAS internals but produces comparable rankings.
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS library path (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_ragas_llm():
    """
    Configure RAGAS to use our LLM provider instead of its OpenAI default.
    Returns (llm_wrapper, embeddings_wrapper) or raises if unavailable.
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        llm = LangchainLLMWrapper(ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=config.OPENAI_API_KEY))
    else:
        from langchain_anthropic import ChatAnthropic
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        llm = LangchainLLMWrapper(ChatAnthropic(model=config.LLM_MODEL, api_key=config.ANTHROPIC_API_KEY))
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL))
    return llm, emb


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Optional[Dict[str, float]]:
    """
    Try to evaluate one Q&A pair with the RAGAS library.
    Returns dict of scores or None if RAGAS is unavailable / fails.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        llm_wrapper, emb_wrapper = _configure_ragas_llm()

        # Wire our LLM into each metric — RAGAS metrics are stateful objects
        for metric in [faithfulness, answer_relevancy, context_precision]:
            metric.llm = llm_wrapper
            if hasattr(metric, "embeddings"):
                metric.embeddings = emb_wrapper

        dataset = Dataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts if contexts else [""]],
            "ground_truth": [ground_truth],
        })

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness":      float(result["faithfulness"]),
            "answer_relevancy":  float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "method":            "ragas",
        }
    except Exception as exc:
        logger.debug(f"RAGAS evaluation failed: {exc}. Falling back to LLM judge.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based fallback metrics (used when RAGAS is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """Evaluate whether the following answer is FAITHFUL to the provided context.

A faithful answer:
- Contains ONLY information present in the context passages
- Does NOT introduce facts not found in the context
- Does NOT contradict the context

Context passages:
{context}

Answer to evaluate:
{answer}

Score the faithfulness from 0 to 10:
- 0: Answer contains many fabricated facts not in context
- 5: Answer is partially supported by context
- 10: Every claim in the answer is directly supported by context

Return ONLY valid JSON: {{"score": <0-10>, "reason": "<one sentence>"}}"""

_RELEVANCY_PROMPT = """Evaluate whether the following answer is RELEVANT to the question.

Question: {question}
Answer: {answer}

Score from 0 to 10:
- 0: Answer completely ignores the question
- 5: Answer partially addresses the question
- 10: Answer directly and completely addresses the question

Return ONLY valid JSON: {{"score": <0-10>, "reason": "<one sentence>"}}"""

_PRECISION_PROMPT = """Evaluate how PRECISE the retrieved context is for answering the question.

Question: {question}
Retrieved context passages:
{context}

Score from 0 to 10:
- 0: Context is completely unrelated to the question
- 5: Context is partially relevant
- 10: Context contains exactly the information needed to answer

Return ONLY valid JSON: {{"score": <0-10>, "reason": "<one sentence>"}}"""


def _score_prompt(prompt: str, llm: LLMClient) -> float:
    """Send a scoring prompt and parse the 0-10 score, normalised to 0-1."""
    try:
        raw = llm.complete(prompt, temperature=0.0, max_tokens=128)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        score = float(data.get("score", 5))
        return round(max(0.0, min(10.0, score)) / 10.0, 3)
    except Exception as exc:
        logger.warning(f"Score parsing failed: {exc}. Defaulting to 0.5.")
        return 0.5


def evaluate_with_llm(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    llm: LLMClient,
) -> Dict[str, float]:
    """
    Fallback metric computation using direct LLM scoring.
    Produces 0-1 scores for faithfulness, answer_relevancy, context_precision.
    """
    context_str = "\n\n---\n\n".join(contexts[:3]) if contexts else "No context provided."

    faithfulness_score = _score_prompt(
        _FAITHFULNESS_PROMPT.format(context=context_str, answer=answer), llm
    )
    relevancy_score = _score_prompt(
        _RELEVANCY_PROMPT.format(question=question, answer=answer), llm
    )
    precision_score = _score_prompt(
        _PRECISION_PROMPT.format(question=question, context=context_str), llm
    )

    return {
        "faithfulness":      faithfulness_score,
        "answer_relevancy":  relevancy_score,
        "context_precision": precision_score,
        "method":            "llm_fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    llm: Optional[LLMClient] = None,
) -> Dict[str, float]:
    """
    Compute faithfulness, answer_relevancy, and context_precision for one sample.

    Tries RAGAS first (more rigorous). Falls back to LLM scoring if RAGAS fails.
    Both methods return scores in [0, 1].
    """
    # Attempt RAGAS
    ragas_result = evaluate_with_ragas(question, answer, contexts, ground_truth)
    if ragas_result:
        return ragas_result

    # Fallback to LLM-based scoring
    if llm is None:
        llm = LLMClient()
    return evaluate_with_llm(question, answer, contexts, ground_truth, llm)
