# RAG Comparison Experiment

A systematic benchmark comparing 4 RAG architectures on hallucination reduction and answer accuracy.

## Architectures Compared

| Architecture | Description |
|---|---|
| Naive RAG | Standard retrieve-then-generate baseline |
| Reranked RAG | Adds a reranker to improve chunk selection |
| ReflexRAG | Self-correcting retrieval with reflection |
| Agentic RAG | Multi-step agent-driven retrieval |

## What we measure

- Hallucination rate per architecture
- Answer accuracy on a fixed question set
- Retrieval precision and recall

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

## Getting Started

```bash
git clone https://github.com/akshita270/RAG-comparison-experiment
cd RAG-comparison-experiment
pip install -r requirements.txt
python main.py
```

## Key Finding

Agentic and Reflexive RAG approaches significantly outperform naive RAG on hallucination metrics, with reranking providing the best cost-performance tradeoff.
