"""
Router for natural language queries.
"""

import time
import structlog
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.models.schemas import AskRequest, SourceChunk
from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.services.llm import llm_service
from app.dependencies import verify_api_key
from app.config import settings

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/ask", tags=["Query"])

@router.post("")
async def ask_question(
    request: AskRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Submit a query and get a streaming grounded answer (FR-03, FR-04, FR-12).
    """
    # 1. Encode Query
    query_embedding = embedder.embed_single(request.query)

    # 2. Search Vector Store
    results = vector_store.search(
        query_embedding=query_embedding,
        k=request.top_k,
        document_id_filter=request.document_id,
        threshold=settings.SIMILARITY_THRESHOLD
    )

    # 3. Handle retrieval (Hybrid fallback)
    # If no results are found, we continue anyway to let the LLM answer from general knowledge.
    # Otherwise, we prepare the sources for the citation chunk.
    sources = []
    if results:
        sources = [
            {
                "document_id": res.document_id,
                "chunk_index": res.chunk_index,
                "text": res.text[:200] + "...",
                "similarity_score": res.similarity_score
            }
            for res in results
        ]

    async def stream_generator():
        # First chunk: Sending metadata and sources (if any)
        yield f"data: {json.dumps({'sources': sources, 'model': settings.OLLAMA_MODEL})}\n\n"
        
        # Stream the response (passing an empty context list if results is empty)
        context_list = [res.text for res in results] if results else []
        async for token in llm_service.stream_answer(request.query, context_list):
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
