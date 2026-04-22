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
from app.services.vector_store import vector_store, SearchResult
from app.services.llm import llm_service
from app.dependencies import verify_api_key
from app.models.document import registry
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

    # 3. Handle 'Summarize' Intent (FR-03/FR-04 extension)
    # If the query contains 'summarize' or 'summary', we may need more context.
    # We also handle 'latest document' if no document_id is provided.
    is_summary_request = any(word in request.query.lower() for word in ["summarize", "summary", "summarise", "main points", "key findings"])
    
    if is_summary_request or not results:
        # If no document_id provided, find the latest ready document
        target_doc_id = request.document_id
        if not target_doc_id:
            all_docs = registry.all_documents()
            ready_docs = [d for d in all_docs if d.status == "ready"]
            if ready_docs:
                # Assuming the last one in the registry list is the latest (since they are appended)
                target_doc_id = ready_docs[-1].document_id
        
        if target_doc_id:
            # Fetch chunks specifically for this document to ensure the LLM has context
            # We take up to 8 chunks for a summary to balance context vs speed
            doc_chunks = [m for m in vector_store._metadata if m.document_id == target_doc_id][:8]
            if doc_chunks and (not results or is_summary_request):
                # Replace or append results with these context chunks
                summary_results = [
                    SearchResult(
                        document_id=m.document_id,
                        chunk_index=m.chunk_index,
                        text=m.text,
                        similarity_score=1.0  # Artificial score for mandatory context
                    )
                    for m in doc_chunks
                ]
                # Merge: keep vector results if they are high quality, but ensure doc chunks are present
                results = summary_results

    # 4. Handle retrieval (Hybrid fallback)
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
