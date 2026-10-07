import pandas as pd


def _clean(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "" if text.lower() == "unknown" else text


def build_movie_text(row):
    """Text that gets embedded for semantic search.

    Only descriptive content (title, year, genres, people, plot). Numbers like votes,
    gross and rating are left out on purpose: they add noise to the embedding and
    measurably hurt retrieval (recall 10/25 -> 13/25 without them), and numeric
    questions are answered exactly by the factual pipeline instead.
    """
    title = _clean(row["Title"])
    year = int(row["Year"]) if not pd.isna(row["Year"]) else None
    genres = _clean(row["Genres"])
    director = _clean(row["Director"])
    stars = _clean(row["Stars"])
    summary = _clean(row["Summary"])

    parts = [f"{title} ({year})." if year else f"{title}."]
    if genres:
        parts.append(f"{genres} movie" + (f" directed by {director}." if director else "."))
    elif director:
        parts.append(f"Directed by {director}.")
    if stars:
        parts.append(f"Starring {stars}.")
    if summary:
        parts.append(summary)
    return " ".join(parts)
