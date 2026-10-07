import re
from movie_rag.pipelines.semantic import semantic_pipeline
from movie_rag.pipelines.factual import factual_pipeline

def _has_word(q: str, words) -> bool:
    # whole-word match, so "min" doesn't fire on "criminal" or "mean" on "meaning"
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


FACTUAL_KEYWORDS = [
    "average", "mean", "median", "count", "how many", "number of", "total", "sum",
    "highest", "lowest", "maximum", "minimum", "max", "min", "top", "best", "worst",
    "greater than", "less than", "higher than", "lower than", "more than", "fewer than", "over", "under",
    "before", "after", "between", "most", "least", "percentage", "ratio",
]
FACTUAL_COLS = ["rating", "rated", "votes", "voted", "gross", "grossing", "box office", "earned",
                "metascore", "runtime", "minutes", "year", "years", "certificate"]
# phrases that imply an aggregation / numeric column on their own
SELF_FACTUAL = ["how many", "number of", "average", "median", "percentage",
                "longest", "shortest", "oldest", "newest", "highest grossing", "highest rated", "lowest rated"]


def classify_query_type(query: str) -> str:
    q = query.lower().strip()

    if _has_word(q, SELF_FACTUAL):
        return "factual"

    if _has_word(q, FACTUAL_KEYWORDS) and _has_word(q, FACTUAL_COLS):
        return "factual"

    if re.search(r"\b(19|20)\d{2}s?\b", q) and _has_word(q, ["after", "before", "between", "since", "from"]):
        return "factual"

    return "semantic"

def answer_any_query(query: str, *, k: int, rich_movies, embedder, index, movie_ids, call_llm_fn, force_type=None):
    qtype = force_type or classify_query_type(query)

    if qtype == "factual":
        out = factual_pipeline(query, rich_movies=rich_movies, call_llm_fn=call_llm_fn)
        return {
            "query": query,
            "query_type": "factual",
            "pipeline_used": "factual_pipeline",
            **out
        }

    out = semantic_pipeline(query, k=k, embedder=embedder, index=index, movie_ids=movie_ids, rich_movies=rich_movies)
    return {
        "query": query,
        "query_type": "semantic",
        "pipeline_used": "semantic_pipeline",
        **out
    }
