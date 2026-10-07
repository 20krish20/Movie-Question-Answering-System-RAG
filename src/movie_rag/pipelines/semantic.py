from typing import List, Dict
import pandas as pd


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def retrieve_movies_for_query(query: str, *, k: int, embedder, index, movie_ids, rich_movies) -> List[Dict]:
    q_emb = embedder.encode_query(query)
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0:  # FAISS pads with -1 when k > index size
            continue
        movie_row_idx = movie_ids[idx]
        row = rich_movies.loc[movie_row_idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "row_idx": int(movie_row_idx),
            "Title": row["Title"],
            "Year": int(row["Year"]) if not pd.isna(row["Year"]) else None,
            "Genres": _text(row.get("Genres")),
            "Director": _text(row.get("Director")),
            "Stars": _text(row.get("Stars")),
            "Rating": float(row["Rating"]) if not pd.isna(row["Rating"]) else None,
            "Summary": _text(row.get("Summary"))
        })
    return results

def semantic_pipeline(query: str, *, k: int, embedder, index, movie_ids, rich_movies):
    retrieved = retrieve_movies_for_query(
        query, k=k, embedder=embedder, index=index, movie_ids=movie_ids, rich_movies=rich_movies
    )

    answer = "Top matches:\n" + "\n".join(
        [f"{r['rank']}. {r['Title']} ({r['Year']}) — score={r['score']:.3f}" for r in retrieved]
    )

    return {"query": query, "retrieved": retrieved, "answer": answer}
