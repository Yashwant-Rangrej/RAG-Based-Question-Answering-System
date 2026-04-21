"""
Document ingestion pipeline.
Orchestrates text extraction, chunking, embedding, and vector storage.

Runs as a background task to keep the API responsive (FR-02).
"""

import os
from pathlib import Path
import structlog
from app.models.document import registry
from app.models.schemas import DocumentStatus
from app.services.extractor import extract_text
from app.services.chunker import chunk_text
from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.config import settings

log = structlog.get_logger(__name__)

async def run_ingestion_pipeline(document_id: str, file_path: str, mime_type: str):
    """
    Background job to process a document.
    Steps:
      1. Extraction
      2. Chunking
      3. Embedding
      4. Vector Store Upsert
    """
    path = Path(file_path)
    try:
        log.info("pipeline_started", document_id=document_id)
        registry.update_status(document_id, DocumentStatus.PROCESSING)

        # 1. Extraction
        text = extract_text(path, mime_type)
        if not text:
            raise ValueError("No text extracted from document.")

        # 2. Chunking
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("Document yielded zero chunks.")

        # 3. Embedding
        embeddings = embedder.embed(chunks)

        # 4. Vector Store Upsert
        vector_store.upsert(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            start_index=0, # Simplified index for now
            model_id=embedder.model_id
        )

        # Success update
        registry.update_status(
            document_id,
            DocumentStatus.READY,
            model_id=embedder.model_id,
            chunk_count=len(chunks)
        )
        log.info("pipeline_success", document_id=document_id, chunks=len(chunks))

    except Exception as e:
        log.error("pipeline_failed", document_id=document_id, error=str(e))
        registry.update_status(document_id, DocumentStatus.FAILED, error=str(e))
    # finally:
        # Optional: cleanup the raw file if desired
        # if path.exists():
        #     os.remove(path)
