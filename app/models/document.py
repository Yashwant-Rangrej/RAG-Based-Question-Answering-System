"""
Thread-safe in-memory document registry.
Tracks processing status, filename, embedding model used (for mismatch detection), and errors.

Design note: This is an in-memory store suitable for single-process deployments.
For multi-worker / distributed setups, replace with Redis or a shared DB.
The embedding model_id stored per document ensures that if the configured model
changes, queries against old documents will be flagged (FR embedding-consistency constraint).
"""

import threading
from dataclasses import dataclass, field
from typing import Optional
from app.models.schemas import DocumentStatus


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    # model_id stored at ingestion time; validated at query time to detect embedding-space mismatches.
    model_id: Optional[str] = None
    error: Optional[str] = None
    chunk_count: int = 0


class DocumentRegistry:
    """Singleton thread-safe registry for document processing state."""

    def __init__(self) -> None:
        self._store: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()

    def register(self, document_id: str, filename: str) -> DocumentRecord:
        record = DocumentRecord(document_id=document_id, filename=filename)
        with self._lock:
            self._store[document_id] = record
        return record

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            return self._store.get(document_id)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error: Optional[str] = None,
        model_id: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        with self._lock:
            record = self._store.get(document_id)
            if record:
                record.status = status
                if error is not None:
                    record.error = error
                if model_id is not None:
                    record.model_id = model_id
                if chunk_count is not None:
                    record.chunk_count = chunk_count

    def all_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._store.values())

    def remove(self, document_id: str) -> bool:
        with self._lock:
            if document_id in self._store:
                del self._store[document_id]
                return True
            return False


# Application-wide singleton
registry = DocumentRegistry()
