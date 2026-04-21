"""
Pydantic v2 schemas for all API request and response models.
Validation errors are automatically returned as structured 422 responses by FastAPI.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ---------- Upload ----------

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    status: DocumentStatus
    message: str


# ---------- Status ----------

class StatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    filename: Optional[str] = None
    error: Optional[str] = None


# ---------- Ask ----------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Natural-language question")
    document_id: Optional[str] = Field(
        None,
        description="Filter retrieval to a specific document. Omit to search all documents."
    )
    # top_k capped at 20 per FR-03: prevents context-window overflow on smaller LLMs.
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve (1–20)")

    @field_validator("query")
    @classmethod
    def sanitise_query(cls, v: str) -> str:
        """Strip null bytes and leading/trailing whitespace (FR-06 input sanitisation)."""
        return v.replace("\x00", "").strip()


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    query_latency_ms: float
    model_used: str

    model_config = {"protected_namespaces": ()}


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_ok: bool
    faiss_vectors: int
    embedding_model: str
