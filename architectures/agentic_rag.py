"""
architectures/agentic_rag.py — Agentic RAG using a ReAct loop.

Motivation: static retrieval (Vanilla, HyDE, Graph) commits to a single
retrieval strategy at indexing time. An agent can instead *reason* about
what information it needs and issue multiple targeted retrieval calls.

This implements the ReAct (Reason + Act) pattern:
  Thought → Action → Observation → Thought → Action → … → Final Answer

The agent has 3 tools:
  1. vector_search(query)       — semantic search over chunked papers
  2. fetch_paper_by_id(id)      — retrieve full text of a specific paper
  3. search_by_keyword(keyword) — keyword match over paper metadata

The agent decides which tool to call, calls it, sees the result, and
can then call more tools before committing to an answer. This is especially
useful when the first retrieval doesn't have enough information.

All tool calls per question are logged so you can analyse agent behaviour.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)

Can be run independently:
    python architectures/agentic_rag.py
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
from utils.embedder import embed_single
from architectures.vanilla_rag import index_papers, retrieve, get_chroma_collection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def tool_vector_search(query: str, collection, top_k: int = config.TOP_K) -> str:
    """
    Semantic vector search — returns the top-k most relevant text chunks.
    Used when the agent knows roughly what it's looking for but not where.
    """
    chunks = retrieve(query, collection, top_k)
    if not chunks:
        return "No relevant chunks found."
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[Chunk {i+1} | {c['title']} ({c['arxiv_id']}) | score={c['score']}]\n{c['text']}")
    return "\n\n".join(parts)


def tool_fetch_paper_by_id(arxiv_id: str, papers: List[Dict]) -> str:
    """
    Return the full text of a specific paper by ArXiv ID.
    Used when the agent has identified a specific paper likely to contain the answer.
    """
    paper_by_id = {p["arxiv_id"]: p for p in papers}
    arxiv_id = arxiv_id.strip().lower()

    # Try exact match first, then substring match (handles partial IDs)
    paper = paper_by_id.get(arxiv_id)
    if not paper:
        for pid, p in paper_by_id.items():
            if arxiv_id in pid or pid in arxiv_id:
                paper = p
                break

    if not paper:
        return f"Paper '{arxiv_id}' not found in the dataset."
    text = paper.get("full_text") or paper["abstract"]
    return f"[{paper['title']} ({paper['arxiv_id']})]\n\n{text[:3000]}"


def tool_search_by_keyword(keyword: str, papers: List[Dict]) -> str:
    """
    Simple keyword search over paper titles, abstracts, and author lists.
    Faster than vector search when looking for a specific term or name.
    """
    keyword_lower = keyword.lower().strip()
    matches = []
    for p in papers:
        searchable = f"{p['title']} {p['abstract']} {' '.join(p['authors'])}".lower()
        if keyword_lower in searchable:
            matches.append(
                f"• [{p['arxiv_id']}] {p['title']}\n  Authors: {', '.join(p['authors'][:3])}\n  "
                f"Abstract snippet: {p['abstract'][:200]}…"
            )
    if not matches:
        return f"No papers found matching keyword '{keyword}'."
    return f"Found {len(matches)} paper(s) matching '{keyword}':\n\n" + "\n\n".join(matches[:5])


# ─────────────────────────────────────────────────────────────────────────────
# ReAct loop
# ─────────────────────────────────────────────────────────────────────────────

_REACT_SYSTEM = """You are a research assistant with access to a database of ArXiv papers.
You must answer questions by using the available tools to retrieve relevant information.
Think step by step and use tools as needed before writing your final answer.

Available tools:
  vector_search(query: str)
    → Returns the most semantically similar paper chunks to your query.
    → Use when you need to find relevant content but don't know which paper.

  fetch_paper_by_id(arxiv_id: str)
    → Returns the full text of a specific paper.
    → Use when you know the paper ID and want the complete content.

  search_by_keyword(keyword: str)
    → Searches paper titles, abstracts, and authors for an exact keyword.
    → Use for specific terms, model names, dataset names, or author names.

Format EVERY response as:
  Thought: [your reasoning about what to do next]
  Action: [tool_name]
  Action Input: [tool argument]

When you have enough information to answer:
  Thought: I have sufficient information to answer the question.
  Final Answer: [your complete answer, citing specific papers]

Rules:
  - Always start with a Thought
  - Use exactly ONE Action per turn (never skip straight to Final Answer without using a tool first)
  - After seeing an Observation, write another Thought then either another Action or Final Answer
  - If tools return nothing useful after 2-3 tries, say so honestly in your Final Answer"""

_REACT_USER_TEMPLATE = """Question: {question}

{history}"""

_OBSERVATION_TEMPLATE = "Observation: {result}\n\n"


def parse_react_output(text: str) -> Dict[str, Optional[str]]:
    """
    Parse the LLM's ReAct-formatted output.
    Returns dict with keys: thought, action, action_input, final_answer.
    All values may be None if that section wasn't present.
    """
    # Check for final answer first
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {"thought": None, "action": None, "action_input": None,
                "final_answer": final_match.group(1).strip()}

    thought_match      = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL | re.IGNORECASE)
    action_match       = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL | re.IGNORECASE)
    action_input_match = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", text, re.DOTALL | re.IGNORECASE)

    return {
        "thought":      thought_match.group(1).strip()      if thought_match      else None,
        "action":       action_match.group(1).strip()       if action_match       else None,
        "action_input": action_input_match.group(1).strip() if action_input_match else None,
        "final_answer": None,
    }


def execute_tool(action: str, action_input: str, collection, papers: List[Dict]) -> str:
    """Dispatch the parsed tool name to the actual tool function."""
    action_lower = action.lower().strip()

    if "vector_search" in action_lower:
        return tool_vector_search(action_input, collection)
    elif "fetch_paper" in action_lower or "fetch_paper_by_id" in action_lower:
        return tool_fetch_paper_by_id(action_input, papers)
    elif "keyword" in action_lower or "search_by_keyword" in action_lower:
        return tool_search_by_keyword(action_input, papers)
    else:
        return f"Unknown tool '{action}'. Available: vector_search, fetch_paper_by_id, search_by_keyword."


def react_loop(
    question: str,
    collection,
    papers: List[Dict],
    llm: LLMClient,
    max_steps: int = config.AGENT_MAX_STEPS,
) -> Dict[str, Any]:
    """
    Run the ReAct loop until the agent produces a Final Answer or hits max_steps.

    Returns: {answer, contexts, tool_calls}
      - tool_calls is a list of {"action": ..., "input": ..., "result": ...}
        for every tool invocation — useful for analysing agent behaviour.
    """
    history_text = ""
    tool_calls = []
    all_observations = []

    for step in range(max_steps):
        # Build the prompt including everything the agent has seen so far
        user_msg = _REACT_USER_TEMPLATE.format(question=question, history=history_text)

        llm_response = llm.complete(
            user_msg,
            system=_REACT_SYSTEM,
            temperature=0.2,   # Slightly higher for diverse tool selection
            max_tokens=768,
        )

        parsed = parse_react_output(llm_response)
        logger.debug(f"Step {step+1}: action={parsed['action']}, final={bool(parsed['final_answer'])}")

        # Agent has decided it has enough information
        if parsed["final_answer"]:
            return {
                "answer":     parsed["final_answer"],
                "contexts":   all_observations,
                "tool_calls": tool_calls,
            }

        # Execute the requested tool
        action       = parsed.get("action") or "vector_search"
        action_input = parsed.get("action_input") or question

        observation = execute_tool(action, action_input, collection, papers)
        all_observations.append(observation[:500])  # store truncated for RAGAS contexts

        tool_calls.append({
            "step":   step + 1,
            "action": action,
            "input":  action_input,
            "result": observation[:300],  # truncate for logging
        })

        # Append this turn to the history so the next LLM call sees it
        history_text += (
            f"Thought: {parsed.get('thought', '')}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {observation[:1000]}\n\n"  # truncate long observations
        )

    # Reached max steps without a Final Answer — ask LLM to conclude from what it has
    logger.warning(f"Agent hit max_steps={max_steps} without Final Answer. Forcing conclusion.")
    forced_prompt = (
        _REACT_USER_TEMPLATE.format(question=question, history=history_text)
        + "\nYou have used all your tool calls. Based ONLY on what you have observed above, "
          "give your best Final Answer now.\nFinal Answer:"
    )
    forced_answer = llm.complete(forced_prompt, system=_REACT_SYSTEM, temperature=0.1, max_tokens=512)

    return {
        "answer":     forced_answer.strip(),
        "contexts":   all_observations,
        "tool_calls": tool_calls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRAG:
    def __init__(self, papers: List[Dict]):
        self.llm        = LLMClient()
        self.papers     = papers
        self.collection = index_papers(papers)

    def answer(self, question: str) -> Dict[str, Any]:
        return react_loop(question, self.collection, self.papers, self.llm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.data_loader import load_papers, load_questions

    papers = load_papers()
    rag    = AgenticRAG(papers)
    questions = load_questions(papers)

    q = questions[0]["question"]
    print(f"\nQuestion: {q}")
    result = rag.answer(q)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTool calls made:")
    for tc in result["tool_calls"]:
        print(f"  Step {tc['step']}: {tc['action']}({tc['input'][:60]})")
