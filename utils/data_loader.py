"""
utils/data_loader.py — Fetch ArXiv papers and generate evaluation questions.

Two public functions:
  load_papers()    → list of paper dicts (fetches from ArXiv if cache missing)
  load_questions() → list of question dicts (generates via LLM if cache missing)

Papers are cached in data/papers.json so you don't hit the ArXiv API on
every run. Questions are cached in data/questions.json for the same reason.
"""
import json
import logging
import tempfile
import time
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paper fetching
# ─────────────────────────────────────────────────────────────────────────────

def _extract_arxiv_id(entry_id: str) -> str:
    """Pull the bare ID (e.g. '2401.12345') out of the full ArXiv URL."""
    return entry_id.rstrip("/").split("/")[-1]


def _try_extract_pdf_text(paper) -> str:
    """
    Attempt to download and extract full text from a paper's PDF.
    Returns empty string on any failure so the caller can fall back
    to the abstract — we never want PDF extraction to crash the run.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        paper.download_pdf(filename=tmp_path)
        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        Path(tmp_path).unlink(missing_ok=True)
        # Keep only the first 8000 chars — enough for good chunking without
        # overwhelming the LLM context windows during question generation.
        return text[:8000].strip()
    except Exception as exc:
        logger.debug(f"PDF extraction failed: {exc}")
        return ""


def fetch_papers(num_papers: int = config.NUM_PAPERS) -> List[Dict[str, Any]]:
    """
    Fetch recent AI/ML papers from ArXiv.
    Returns a list of dicts with keys:
      arxiv_id, title, authors, abstract, full_text, url, published
    """
    import arxiv

    logger.info(f"Fetching {num_papers} papers from ArXiv …")
    search = arxiv.Search(
        query=config.ARXIV_QUERY,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for entry in search.results():
        arxiv_id = _extract_arxiv_id(entry.entry_id)
        abstract  = entry.summary.replace("\n", " ").strip()

        # Try PDF; fall back to abstract-only if extraction fails or is slow.
        full_text = _try_extract_pdf_text(entry)
        if not full_text:
            full_text = abstract   # graceful fallback
            logger.debug(f"Using abstract-only for {arxiv_id}")

        papers.append({
            "arxiv_id":  arxiv_id,
            "title":     entry.title.strip(),
            "authors":   [str(a) for a in entry.authors],
            "abstract":  abstract,
            "full_text": full_text,
            "url":       entry.entry_id,
            "published": entry.published.isoformat() if entry.published else "",
        })
        time.sleep(0.5)  # be polite to ArXiv rate limits

    logger.info(f"Fetched {len(papers)} papers.")
    return papers


def load_papers() -> List[Dict[str, Any]]:
    """
    Return papers from local cache (data/papers.json) if it exists,
    otherwise fetch from ArXiv and save the cache.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.PAPERS_FILE.exists():
        logger.info(f"Loading papers from cache: {config.PAPERS_FILE}")
        with open(config.PAPERS_FILE) as f:
            return json.load(f)

    papers = fetch_papers()
    with open(config.PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Saved {len(papers)} papers to {config.PAPERS_FILE}")
    return papers


# ─────────────────────────────────────────────────────────────────────────────
# Question generation
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_SYSTEM = """You are a research evaluator creating factual quiz questions
from academic papers. Questions must be answerable with specific facts from the
paper text, not general knowledge."""

_QUESTION_PROMPT = """Read the following research paper excerpt and generate exactly
ONE specific, fact-checkable question AND its ground-truth answer.

The question must be about a concrete detail in the paper such as:
- Which dataset was used?
- What baseline method did the authors compare against?
- What was the reported accuracy/F1/BLEU score?
- What architecture or model was proposed?
- What is the key finding or main contribution?

Avoid vague questions like "What is this paper about?"

Paper title: {title}
Paper ID: {arxiv_id}

Paper text:
{text}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "question": "...",
  "ground_truth": "...",
  "arxiv_id": "{arxiv_id}",
  "paper_title": "{title}"
}}"""


def generate_questions(papers: List[Dict], num_questions: int = config.NUM_QUESTIONS) -> List[Dict]:
    """
    Use the LLM to generate one specific, fact-checkable question per paper.
    Papers are sampled if there are more papers than needed.
    """
    llm = LLMClient()
    selected = papers[:num_questions]  # take first N papers
    questions = []

    for i, paper in enumerate(selected):
        logger.info(f"Generating question {i+1}/{len(selected)} for '{paper['title'][:60]}…'")
        # Prefer full text; fall back to abstract. Truncate to avoid huge prompts.
        text = (paper.get("full_text") or paper["abstract"])[:4000]

        prompt = _QUESTION_PROMPT.format(
            title=paper["title"],
            arxiv_id=paper["arxiv_id"],
            text=text,
        )
        try:
            raw = llm.complete(prompt, system=_QUESTION_SYSTEM, temperature=0.3, max_tokens=512)
            # Strip markdown code fences if the model wrapped the JSON
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            q = json.loads(raw)
            # Validate required fields exist
            assert "question" in q and "ground_truth" in q
            questions.append(q)
        except Exception as exc:
            logger.warning(f"Question generation failed for {paper['arxiv_id']}: {exc}")
            # Create a fallback question from the abstract so we always
            # have the right count — better than crashing the pipeline.
            questions.append({
                "question":    f"What is the main contribution of the paper '{paper['title']}'?",
                "ground_truth": paper["abstract"][:500],
                "arxiv_id":   paper["arxiv_id"],
                "paper_title": paper["title"],
            })
        time.sleep(0.5)  # avoid hammering the LLM API

    return questions


def load_questions(papers: List[Dict]) -> List[Dict]:
    """
    Return questions from local cache if it exists, otherwise generate them.
    """
    if config.QUESTIONS_FILE.exists():
        logger.info(f"Loading questions from cache: {config.QUESTIONS_FILE}")
        with open(config.QUESTIONS_FILE) as f:
            return json.load(f)

    questions = generate_questions(papers)
    with open(config.QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    logger.info(f"Saved {len(questions)} questions to {config.QUESTIONS_FILE}")
    return questions
