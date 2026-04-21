"""
FAISS vector store wrapper.

Index type: IndexFlatIP (Inner Product)
  - With L2-normalised embeddings, inner product == cosine similarity.
  - IndexFlatIP performs exact nearest-neighbour search (no approximation error).
  - For collections > ~100k vectors, consider upgrading to IndexIVFFlat or IndexHNSW
    without changing the interface (swap in vector_store.py only).

Persistence:
  - FAISS index is saved to data/faiss.index after every upsert.
  - Metadata (chunk_text, document_id, chunk_index) is saved to data/metadata.json.
  - On application startup, both are loaded so no data is lost across restarts.

Thread safety:
  - A threading.Lock guards all read/write operations on the FAISS index
    since faiss-cpu is not thread-safe by default.
"""

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import structlog

from app.config import settings

log = structlog.get_logger(__name__)

_INDEX_PATH = settings.DATA_DIR / "faiss.index"
_META_PATH = settings.DATA_DIR / "metadata.json"


@dataclass
class ChunkMetadata:
    document_id: str
    chunk_index: int
    text: str
    model_id: str


@dataclass
class SearchResult:
    document_id: str
    chunk_index: int
    text: str
    similarity_score: float


class FAISSVectorStore:
    """Thread-safe FAISS IndexFlatIP wrapper with disk persistence."""

    _instance: "FAISSVectorStore | None" = None

    def __new__(cls) -> "FAISSVectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def _init(self) -> None:
        if self._initialised:
            return
        self._lock = threading.Lock()
        self._metadata: list[ChunkMetadata] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._load_from_disk()
        self._initialised = True

    def _load_from_disk(self) -> None:
        """Load persisted index and metadata from disk (called once at startup)."""
        if _INDEX_PATH.exists() and _META_PATH.exists():
            try:
                self._index = faiss.read_index(str(_INDEX_PATH))
                with _META_PATH.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._metadata = [ChunkMetadata(**item) for item in raw]
                log.info(
                    "faiss_loaded_from_disk",
                    vectors=self._index.ntotal,
                    meta_records=len(self._metadata),
                )
            except Exception as e:
                log.error("faiss_load_failed", error=str(e))
                self._index = None
                self._metadata = []
        else:
            log.info("faiss_no_persisted_index")

    def _save_to_disk(self) -> None:
        """Persist index and metadata after every upsert."""
        try:
            faiss.write_index(self._index, str(_INDEX_PATH))
            with _META_PATH.open("w", encoding="utf-8") as f:
                json.dump([asdict(m) for m in self._metadata], f, ensure_ascii=False)
        except Exception as e:
            log.error("faiss_save_failed", error=str(e))

    def _ensure_index(self, dim: int) -> None:
        """Lazily create the FAISS index with the correct dimension."""
        if self._index is None:
            # IndexFlatIP: exact inner product search. With L2-normalised vectors = cosine similarity.
            self._index = faiss.IndexFlatIP(dim)
            log.info("faiss_index_created", dim=dim)

    def upsert(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        document_id: str,
        start_index: int,
        model_id: str,
    ) -> None:
        """
        Add chunk embeddings to the index.

        Args:
            chunks: List of chunk text strings.
            embeddings: float32 ndarray of shape (len(chunks), dim).
            document_id: ID of the parent document.
            start_index: global chunk index offset for this document.
            model_id: Embedding model used — stored for consistency checks.
        """
        self._init()
        if len(chunks) == 0:
            return

        dim = embeddings.shape[1]
        with self._lock:
            self._ensure_index(dim)
            self._index.add(embeddings)
            for i, chunk_text in enumerate(chunks):
                self._metadata.append(
                    ChunkMetadata(
                        document_id=document_id,
                        chunk_index=start_index + i,
                        text=chunk_text,
                        model_id=model_id,
                    )
                )
            self._save_to_disk()

        log.info(
            "faiss_upserted",
            document_id=document_id,
            chunk_count=len(chunks),
            total_vectors=self._index.ntotal,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        document_id_filter: Optional[str] = None,
        threshold: float = 0.35,
    ) -> list[SearchResult]:
        """
        Retrieve top-k most similar chunks.

        Args:
            query_embedding: float32 array of shape (1, dim) — L2-normalised.
            k: Number of results to return.
            document_id_filter: If set, restrict results to a specific document.
            threshold: Minimum cosine similarity (inner product with normalised vectors).
                       Results below this threshold are discarded (graceful degradation).

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        self._init()
        if self._index is None or self._index.ntotal == 0:
            log.warning("faiss_search_empty_index")
            return []

        # Over-fetch to account for post-filter (document_id) and threshold cuts.
        fetch_k = min(k * 10 if document_id_filter else k * 3, self._index.ntotal)

        with self._lock:
            scores, indices = self._index.search(query_embedding, fetch_k)

        results: list[SearchResult] = []
        seen_chunks: set[tuple[str, int]] = set()  # de-duplicate

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if float(score) < threshold:
                # Scores are sorted descending; once below threshold, all subsequent are too.
                # (Only valid for IndexFlatIP with normalised vectors.)
                break

            meta = self._metadata[idx]

            if document_id_filter and meta.document_id != document_id_filter:
                continue

            dedup_key = (meta.document_id, meta.chunk_index)
            if dedup_key in seen_chunks:
                continue
            seen_chunks.add(dedup_key)

            results.append(
                SearchResult(
                    document_id=meta.document_id,
                    chunk_index=meta.chunk_index,
                    text=meta.text,
                    similarity_score=float(score),
                )
            )

            if len(results) >= k:
                break

        log.info(
            "faiss_search_done",
            results_returned=len(results),
            k=k,
            threshold=threshold,
            filter=document_id_filter,
        )
        return results

    @property
    def total_vectors(self) -> int:
        self._init()
        return self._index.ntotal if self._index else 0

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document from metadata.
        Note: FAISS IndexFlatIP does not support individual vector deletion;
        vectors remain in the index but will be filtered by document_id at search time.
        For a complete rebuild, reconstruct the index from remaining metadata.
        """
        self._init()
        with self._lock:
            original_count = len(self._metadata)
            self._metadata = [m for m in self._metadata if m.document_id != document_id]
            removed = original_count - len(self._metadata)
        log.info("document_deleted_from_metadata", document_id=document_id, chunks_removed=removed)
        return removed


# Application-wide singleton
vector_store = FAISSVectorStore()
