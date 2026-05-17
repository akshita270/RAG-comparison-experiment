"""
architectures/graph_rag.py — Knowledge Graph RAG (GraphRAG).

Motivation: flat vector retrieval treats all chunks equally and ignores
structure. Research papers have rich relationships: paper A *builds on*
paper B, method X was *evaluated on* dataset Y, model Z *outperforms*
baseline W. A knowledge graph captures these relationships explicitly
so retrieval can traverse them.

Pipeline:
  Indexing:
    1. Extract entities from each paper via LLM
       (datasets, methods, baselines, tasks, findings)
    2. Build a NetworkX directed graph connecting papers to their entities
    3. (Optional) store in Neo4j if available; falls back to NetworkX

  Query:
    1. Extract key entities from the query via LLM
    2. Find those entities in the graph and expand to their neighbours
    3. Collect context from graph-adjacent paper chunks
    4. Combine with vector-retrieved chunks (hybrid context)
    5. Pass combined context + question to LLM for final answer

Can be run independently:
    python architectures/graph_rag.py
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.llm_client import LLMClient
from utils.embedder import embed_single
from architectures.vanilla_rag import index_papers, retrieve, get_chroma_collection

logger = logging.getLogger(__name__)

GRAPH_CACHE_FILE = config.DATA_DIR / "knowledge_graph.json"


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction (runs once during indexing)
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_SYSTEM = "You are an information extraction assistant for research papers."

_ENTITY_PROMPT = """Extract structured information from this research paper excerpt.
Return ONLY valid JSON (no markdown, no extra text):
{{
  "datasets": ["list of dataset names used in experiments"],
  "methods": ["list of methods or models proposed or used"],
  "baselines": ["list of baseline methods this paper compares against"],
  "tasks": ["list of NLP/ML tasks or benchmarks evaluated on"],
  "key_findings": ["1-2 sentence key contributions or findings"]
}}

Paper title: {title}
Paper text (truncated):
{text}"""


def extract_entities(paper: Dict, llm: LLMClient) -> Dict[str, List[str]]:
    """
    Use the LLM to extract named entities and relationships from a paper.
    Returns a dict keyed by entity type; values are lists of entity strings.
    """
    text = (paper.get("full_text") or paper["abstract"])[:3000]
    prompt = _ENTITY_PROMPT.format(title=paper["title"], text=text)
    try:
        raw = llm.complete(prompt, system=_ENTITY_SYSTEM, temperature=0.0, max_tokens=512)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        entities = json.loads(raw)
        # Normalise: ensure all expected keys exist
        for key in ["datasets", "methods", "baselines", "tasks", "key_findings"]:
            entities.setdefault(key, [])
        return entities
    except Exception as exc:
        logger.warning(f"Entity extraction failed for {paper['arxiv_id']}: {exc}")
        return {"datasets": [], "methods": [], "baselines": [], "tasks": [], "key_findings": []}


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(papers: List[Dict], llm: LLMClient) -> nx.DiGraph:
    """
    Build a directed knowledge graph where:
      - Each paper is a node (type='paper')
      - Each extracted entity is a node (type=entity_type)
      - Directed edges encode the relationship: paper → uses → dataset, etc.

    Graph structure enables traversal: "find all papers that use GPT-4"
    or "find all baselines mentioned alongside LoRA" without vector search.
    """
    G = nx.DiGraph()

    for paper in papers:
        paper_node = f"paper:{paper['arxiv_id']}"
        G.add_node(paper_node, type="paper", title=paper["title"],
                   arxiv_id=paper["arxiv_id"], abstract=paper["abstract"][:500])

        logger.info(f"Extracting entities from {paper['arxiv_id']} …")
        entities = extract_entities(paper, llm)

        for dataset in entities.get("datasets", []):
            node = f"dataset:{dataset.lower().strip()}"
            G.add_node(node, type="dataset", name=dataset)
            G.add_edge(paper_node, node, relation="uses_dataset")

        for method in entities.get("methods", []):
            node = f"method:{method.lower().strip()}"
            G.add_node(node, type="method", name=method)
            G.add_edge(paper_node, node, relation="proposes_method")

        for baseline in entities.get("baselines", []):
            node = f"baseline:{baseline.lower().strip()}"
            G.add_node(node, type="baseline", name=baseline)
            G.add_edge(paper_node, node, relation="compares_against")

        for task in entities.get("tasks", []):
            node = f"task:{task.lower().strip()}"
            G.add_node(node, type="task", name=task)
            G.add_edge(paper_node, node, relation="evaluated_on")

    return G


def save_graph(G: nx.DiGraph, path: Path = GRAPH_CACHE_FILE):
    """Serialise graph to JSON so it can be reloaded without re-running LLM extraction."""
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: Path = GRAPH_CACHE_FILE) -> nx.DiGraph:
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True)


def get_or_build_graph(papers: List[Dict], llm: LLMClient) -> nx.DiGraph:
    """Return cached graph if it exists, otherwise build and cache it."""
    if GRAPH_CACHE_FILE.exists():
        logger.info(f"Loading knowledge graph from {GRAPH_CACHE_FILE} …")
        return load_graph()
    G = build_graph(papers, llm)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_graph(G)
    logger.info(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Query-time entity extraction & graph traversal
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_ENTITY_PROMPT = """Extract the key entity names from this research question.
Return ONLY a JSON list of strings — the entity names to look up.

Question: {question}

Examples of entities: dataset names, model names, method names, task names.
Return ONLY valid JSON list, e.g.: ["GPT-4", "SQuAD", "fine-tuning"]"""


def extract_query_entities(question: str, llm: LLMClient) -> List[str]:
    """Extract searchable entities from the question to drive graph traversal."""
    try:
        raw = llm.complete(
            _QUERY_ENTITY_PROMPT.format(question=question),
            temperature=0.0, max_tokens=128,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        entities = json.loads(raw)
        return [str(e).lower().strip() for e in entities if e]
    except Exception:
        # Fall back to using words from the question itself
        return [w.lower() for w in question.split() if len(w) > 4]


def graph_retrieve(question: str, G: nx.DiGraph, llm: LLMClient, papers: List[Dict]) -> List[str]:
    """
    Traverse the knowledge graph to find papers related to the query entities.

    Strategy:
      1. Extract entities from the question
      2. Find matching nodes in the graph (fuzzy substring match)
      3. Collect paper nodes reachable from those entity nodes
      4. Return the abstracts/text snippets of those papers as context
    """
    query_entities = extract_query_entities(question, llm)
    logger.debug(f"Graph query entities: {query_entities}")

    matched_paper_ids: Set[str] = set()

    for entity in query_entities:
        for node in G.nodes():
            # Fuzzy match: entity string appears in the node identifier
            if entity in node.lower():
                # Walk backwards: entity node → paper nodes that reference it
                predecessors = list(G.predecessors(node))
                for pred in predecessors:
                    if G.nodes[pred].get("type") == "paper":
                        matched_paper_ids.add(G.nodes[pred].get("arxiv_id", ""))

    logger.debug(f"Graph matched paper IDs: {matched_paper_ids}")

    # Build a paper lookup dict for fast retrieval
    paper_by_id = {p["arxiv_id"]: p for p in papers}

    contexts = []
    for arxiv_id in list(matched_paper_ids)[:config.TOP_K]:
        paper = paper_by_id.get(arxiv_id)
        if paper:
            text = (paper.get("full_text") or paper["abstract"])[:800]
            contexts.append(f"[Graph match — {paper['title']} ({arxiv_id})]\n{text}")

    return contexts


# ─────────────────────────────────────────────────────────────────────────────
# Full answer pipeline
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """You are a research assistant with access to both a knowledge
graph and retrieved document passages. Answer factually using ONLY the provided
context. If the answer is not present, say so explicitly."""

_ANSWER_PROMPT = """Graph-retrieved context (papers related to your query via entity graph):
{graph_context}

Vector-retrieved context (most similar document chunks):
{vector_context}

---
Question: {question}

Synthesise an answer using the information above. Cite specific papers and facts."""


def answer(
    question: str,
    G: nx.DiGraph,
    collection,
    papers: List[Dict],
    llm: LLMClient,
    top_k: int = config.TOP_K,
) -> Dict[str, Any]:
    """
    Hybrid graph + vector retrieval answer.
    Returns: {answer, contexts, sources, graph_paper_ids}
    """
    # Graph traversal path: entity matching → related papers
    graph_contexts = graph_retrieve(question, G, llm, papers)

    # Vector retrieval path: same as Vanilla RAG (for comparison fairness)
    vector_chunks = retrieve(question, collection, top_k)
    vector_contexts = [c["text"] for c in vector_chunks]

    # Combine both context sources — the union gives broader coverage
    all_contexts = graph_contexts + vector_contexts

    graph_str  = "\n\n---\n\n".join(graph_contexts)  if graph_contexts  else "No graph matches found."
    vector_str = "\n\n---\n\n".join(vector_contexts) if vector_contexts else "No vector matches found."

    llm_answer = llm.complete(
        _ANSWER_PROMPT.format(
            graph_context=graph_str,
            vector_context=vector_str,
            question=question,
        ),
        system=_ANSWER_SYSTEM,
        temperature=0.1,
        max_tokens=512,
    )

    return {
        "answer":          llm_answer,
        "contexts":        all_contexts,
        "sources":         [{"arxiv_id": c["arxiv_id"], "title": c["title"]} for c in vector_chunks],
        "graph_paper_ids": [c.split("(")[-1].rstrip(")]\n") for c in graph_contexts],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(self, papers: List[Dict]):
        self.llm        = LLMClient()
        self.papers     = papers
        self.collection = index_papers(papers)
        self.G          = get_or_build_graph(papers, self.llm)

    def answer(self, question: str) -> Dict[str, Any]:
        return answer(question, self.G, self.collection, self.papers, self.llm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.data_loader import load_papers, load_questions

    papers = load_papers()
    rag    = GraphRAG(papers)
    questions = load_questions(papers)

    q = questions[0]["question"]
    print(f"\nQuestion: {q}")
    result = rag.answer(q)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nGraph-matched papers: {result.get('graph_paper_ids', [])}")
