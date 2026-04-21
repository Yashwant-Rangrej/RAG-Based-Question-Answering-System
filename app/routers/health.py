"""
Health check router.
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.llm import llm_service
from app.services.vector_store import vector_store
from app.services.embedder import embedder
from app.config import settings

router = APIRouter(prefix="/health", tags=["System"])

@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Liveness and readiness check (FR-05).
    """
    ollama_ok = await llm_service.check_health()
    
    return HealthResponse(
        status="active",
        version=settings.VERSION,
        ollama_ok=ollama_ok,
        faiss_vectors=vector_store.total_vectors,
        embedding_model=embedder.model_id
    )
