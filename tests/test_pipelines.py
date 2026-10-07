import numpy as np
import pytest

from movie_rag.pipelines.factual import factual_pipeline
from movie_rag.pipelines.semantic import semantic_pipeline


def fake_llm(*responses):
    calls = []

    def _call(prompt):
        calls.append(prompt)
        return responses[min(len(calls) - 1, len(responses) - 1)]
    _call.calls = calls
    return _call


def test_factual_scalar(movies):
    llm = fake_llm('```python\nresult = rich_movies[rich_movies["Director"] == "Christopher Nolan"]["Rating"].mean()\n```')
    out = factual_pipeline("avg nolan rating", rich_movies=movies, call_llm_fn=llm)
    assert out["result_kind"] == "scalar"
    assert out["result"] == pytest.approx(8.833, abs=1e-3)


def test_factual_table_drops_text_column(movies):
    llm = fake_llm('```python\nresult = rich_movies.nlargest(2, "Rating")\n```')
    out = factual_pipeline("top 2", rich_movies=movies, call_llm_fn=llm)
    assert out["result_kind"] == "table"
    assert list(out["result"]["Title"]) == ["The Dark Knight", "Inception"]
    assert "text" not in out["result"].columns


def test_factual_retries_with_error_feedback(movies):
    llm = fake_llm(
        '```python\nresult = rich_movies["Ratingz"].mean()\n```',
        '```python\nresult = rich_movies["Rating"].max()\n```',
    )
    out = factual_pipeline("max rating", rich_movies=movies, call_llm_fn=llm)
    assert out["result"] == 9.0
    assert "KeyError" in llm.calls[1]


def test_factual_unsafe_code_never_runs(movies):
    llm = fake_llm('```python\nresult = rich_movies.to_csv("/tmp/pwned.csv")\n```')
    with pytest.raises(RuntimeError, match="safety validator"):
        factual_pipeline("x", rich_movies=movies, call_llm_fn=llm)


class FakeEmbedder:
    def encode_query(self, q):
        return np.zeros((1, 4), dtype="float32")


class FakeIndex:
    def search(self, q, k):
        idx = np.array([[2, 3, -1]])[:, :k]
        return np.array([[0.9, 0.8, 0.0]])[:, :k], idx


def test_semantic_handles_missing_summary_and_padding(movies):
    out = semantic_pipeline("q", k=3, embedder=FakeEmbedder(), index=FakeIndex(),
                            movie_ids=np.arange(len(movies)), rich_movies=movies)
    titles = [r["Title"] for r in out["retrieved"]]
    assert titles == ["The Dark Knight", "Toy Story"]
    assert out["retrieved"][1]["Summary"] == ""
