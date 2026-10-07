from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MOVIES_CSV = DATA_DIR / "rich_movies.csv"

EMBEDDINGS_NPY = ARTIFACTS_DIR / "embeddings_norm.npy"
FAISS_INDEX = ARTIFACTS_DIR / "faiss.index"
MOVIE_IDS_PKL = ARTIFACTS_DIR / "movie_ids.pkl"

# all-mpnet-base-v2 as an 8-bit ONNX export: same retrieval quality as the torch model,
# ~475MB RAM at query time instead of ~860MB, no torch dependency.
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_ONNX_FILE = "onnx/model_quint8_avx2.onnx"
EMBED_DIM = 768
MODEL_CACHE_DIR = str(PROJECT_ROOT / ".model_cache")
DEFAULT_TOP_K = 5
