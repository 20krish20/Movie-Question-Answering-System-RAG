import argparse
import os
import pickle

import numpy as np
import pandas as pd
import faiss

from movie_rag.indexing.embedder import Embedder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data/processed/rich_movies.csv")
    parser.add_argument("--outdir", required=True, help="Directory to save artifacts")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    texts = df["text"].astype(str).tolist()
    movie_ids = df.index.to_numpy()

    embedder = Embedder(device=args.device)
    embeddings_norm = embedder.encode_corpus(texts, batch_size=args.batch_size)

    # FAISS cosine similarity = inner product on normalized vectors
    d = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm)

    faiss.write_index(index, os.path.join(args.outdir, "faiss.index"))
    np.save(os.path.join(args.outdir, "embeddings_norm.npy"), embeddings_norm)

    with open(os.path.join(args.outdir, "movie_ids.pkl"), "wb") as f:
        pickle.dump(movie_ids, f)

    print("Saved FAISS index + embeddings + movie_ids to:", args.outdir)
    print("Vectors in index:", index.ntotal)

if __name__ == "__main__":
    main()
