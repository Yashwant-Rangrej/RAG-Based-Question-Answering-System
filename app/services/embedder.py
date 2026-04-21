"""
Singleton embedding service using sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384-dimensional float32 dense vectors.
  - Fast inference (~14k sentences/second on CPU).
  - Strong performance on semantic textual similarity benchmarks.
  - Fully local — no API key, no network call during query.

Consistency guarantee (FR constraint):
  The model_id is stored per-document in the registry at ingestion time.
  At query time, if the configured model differs from the stored model_id,
  the system should warn/reject to prevent silent embedding-space mismatches
  that would cause semantically incorrect similarity scores.

Batch encoding:
  batch_size=32 is the sweet spot for CPU inference on MiniLM.
  Higher batch sizes do not improve throughput significantly on CPU
  and cause larger peak memory usage.
"""

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from app.config import settings

log = structlog.get_logger(__name__)


class Embedder:
    """Singleton wrapper around a SentenceTransformer model."""

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if not self._loaded:
            log.info("embedder_loading", model=settings.EMBEDDING_MODEL)
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._model_id = settings.EMBEDDING_MODEL
            self._loaded = True
            log.info("embedder_ready", model=self._model_id)

    @property
    def model_id(self) -> str:
        self._load()
        return self._model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings into dense float32 vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) — float32.
        """
        self._load()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        log.info("embedding_batch", count=len(texts), batch_size=settings.EMBEDDING_BATCH_SIZE)
        embeddings = self._model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-norm for cosine similarity via inner product.
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Encode a single string. Returns shape (1, dim)."""
        return self.embed([text])


# Application-wide singleton
embedder = Embedder()
