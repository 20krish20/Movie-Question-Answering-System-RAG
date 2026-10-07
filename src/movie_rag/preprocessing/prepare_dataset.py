import argparse
import pandas as pd

from movie_rag.preprocessing.clean_movies import clean_movies
from movie_rag.preprocessing.text_builder import build_movie_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV, e.g. data/raw/imdb_top_1000.csv")
    parser.add_argument("--output", required=True, help="Path to save cleaned rich_movies.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = clean_movies(df)

    # Build text column here (feature engineering for retrieval)
    df["text"] = df.apply(build_movie_text, axis=1)

    df.to_csv(args.output, index=False)
    print("Saved cleaned dataset:", args.output)
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()
