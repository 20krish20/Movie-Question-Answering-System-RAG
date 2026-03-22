# Movie RAG QA System (Production-Style)

A production-style NLP system for movie question answering:
- **Semantic queries**: embeddings + FAISS retrieval
- **Factual queries**: LLM-generated **single-line pandas** executed safely (validated)
- **Router**: automatically selects semantic vs factual pipeline

## Architecture

![Architecture](docs/architecture.png)

## Repository Layout
- `src/movie_rag/` — core application code
- `data/` — datasets (ignored in git)
- `artifacts/` — embeddings + FAISS index (ignored in git)

## Quickstart (Semantic)
1) Prepare cleaned dataset: `data/processed/rich_movies.csv`
2) Build artifacts: `artifacts/faiss.index`, `artifacts/movie_ids.pkl`
3) Run:
```bash
python -m movie_rag.cli --query "movies about aliens attacking humans"
