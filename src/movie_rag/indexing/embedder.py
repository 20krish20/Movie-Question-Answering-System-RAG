import numpy as np
from numpy.linalg import norm
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from movie_rag.config import settings


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    return (emb / np.clip(norm(emb, axis=1, keepdims=True), 1e-9, None)).astype("float32")


def _register_onnx_model(model_name: str):
    if any(m["model"] == model_name for m in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,  # sentence-transformers mpnet uses mean pooling
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=settings.EMBED_DIM,
        model_file=settings.EMBED_ONNX_FILE,
    )


class Embedder:
    """ONNX embeddings via fastembed: no torch, small enough for free-tier hosting."""

    def __init__(self, model_name: str = settings.EMBED_MODEL_NAME, device: str = "cpu"):
        # device is kept for CLI compatibility; fastembed runs on CPU via onnxruntime
        _register_onnx_model(model_name)
        self.model = TextEmbedding(model_name, cache_dir=settings.MODEL_CACHE_DIR)

    def encode_query(self, query: str) -> np.ndarray:
        return _l2_normalize(np.array(list(self.model.embed([query]))))

    def encode_corpus(self, texts, batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        return _l2_normalize(np.array(list(self.model.embed(list(texts), batch_size=batch_size))))
