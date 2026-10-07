"""Loads data, FAISS index and embedder once; shared by the CLI and the web app."""
from movie_rag.config.settings import DEFAULT_TOP_K
from movie_rag.indexing.embedder import Embedder
from movie_rag.io.load_artifacts import load_faiss_index, load_movie_ids
from movie_rag.io.load_data import load_movies
from movie_rag.llm import call_llm
from movie_rag.pipelines.router import answer_any_query

MAX_QUERY_LEN = 300
MAX_K = 20


class MovieQA:
    def __init__(self, device: str = "cpu", call_llm_fn=call_llm):
        self.rich_movies = load_movies()
        self.index = load_faiss_index()
        self.movie_ids = load_movie_ids()
        self.embedder = Embedder(device=device)
        self.call_llm_fn = call_llm_fn

    def ask(self, query: str, k: int = DEFAULT_TOP_K, force_type=None) -> dict:
        query = (query or "").strip()
        if not query:
            raise ValueError("Please enter a question.")
        if len(query) > MAX_QUERY_LEN:
            raise ValueError(f"Question is too long (max {MAX_QUERY_LEN} characters).")
        return answer_any_query(
            query,
            k=max(1, min(int(k), MAX_K)),
            rich_movies=self.rich_movies,
            embedder=self.embedder,
            index=self.index,
            movie_ids=self.movie_ids,
            call_llm_fn=self.call_llm_fn,
            force_type=force_type,
        )
