import pandas as pd
import numpy as np

# Kaggle "IMDB Movies Dataset" (Harshit Shankhdhar, imdb_top_1000.csv) -> project schema
TOP_1000_RENAME = {
    "Series_Title": "Title",
    "Released_Year": "Year",
    "Genre": "Genres",
    "IMDB_Rating": "Rating",
    "Overview": "Summary",
    "Meta_score": "Metascore",
    "No_of_Votes": "Votes",
}


def normalize_schema(movie_data: pd.DataFrame) -> pd.DataFrame:
    """Map known source schemas onto: Title, Year, Genres, Certificate, Runtime, Rating,
    Metascore, Votes, Gross(Million), Director, Stars, Summary. Already-normalized data passes through."""
    movies = movie_data.copy()
    if "Series_Title" not in movies.columns:
        return movies

    movies = movies.rename(columns=TOP_1000_RENAME)
    # "142 min" -> 142
    movies["Runtime"] = movies["Runtime"].astype(str).str.extract(r"(\d+)", expand=False)
    # "28,341,469" (dollars) -> 28.341469 (millions)
    movies["Gross(Million)"] = pd.to_numeric(movies["Gross"].astype(str).str.replace(",", ""), errors="coerce") / 1e6
    star_cols = [c for c in ["Star1", "Star2", "Star3", "Star4"] if c in movies.columns]
    movies["Stars"] = movies[star_cols].apply(lambda r: ", ".join(s for s in r if isinstance(s, str)), axis=1)
    return movies.drop(columns=["Gross", "Poster_Link", *star_cols], errors="ignore")


def clean_movies(movie_data: pd.DataFrame) -> pd.DataFrame:
    movies = normalize_schema(movie_data)

    # Fill missing as 'Unknown' in Stars and Certificate Columns (your original logic)
    movies["Stars"] = movies["Stars"].fillna("Unknown")
    movies["Certificate"] = movies["Certificate"].fillna("Unknown")

    # Ensure numeric types (robust)
    for col in ["Year","Runtime","Rating","Metascore","Votes","Gross(Million)"]:
        if col in movies.columns:
            movies[col] = pd.to_numeric(movies[col], errors="coerce")

    # Fill missing in Runtime column with its mean (your original logic)
    movies["Runtime"] = movies["Runtime"].fillna(movies["Runtime"].mean())

    # Gross(Million) is left as NaN when unknown: imputing a median would make factual
    # answers ("total gross of Nolan films") silently wrong.

    # Basic whitespace cleanup (safe)
    for col in ["Title","Genres","Director","Stars","Certificate","Summary"]:
        if col in movies.columns:
            movies[col] = movies[col].astype("string").fillna("").str.strip().replace({"": np.nan})

    # Restore Unknown for key categoricals if stripping made them NaN
    movies["Stars"] = movies["Stars"].fillna("Unknown")
    movies["Certificate"] = movies["Certificate"].fillna("Unknown")

    return movies
