import pandas as pd
from movie_rag.config.settings import MOVIES_CSV

def load_movies():
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {MOVIES_CSV}")
    movies = pd.read_csv(MOVIES_CSV)
    # nullable int so years render as 1994, not 1994.0, even when one year is missing
    movies["Year"] = movies["Year"].astype("Int64")
    return movies
