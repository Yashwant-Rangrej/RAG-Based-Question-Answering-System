"""
Router for document status polling.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import StatusResponse
from app.models.document import registry
from app.dependencies import verify_api_key

router = APIRouter(prefix="/status", tags=["Ingestion"])

@router.get("/{document_id}", response_model=StatusResponse)
async def get_document_status(
    document_id: str,
    _api_key: str = Depends(verify_api_key)
):
    """
    Poll processing status of a document (FR-05).
    """
    record = registry.get(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document ID not found."
        )
    
    return StatusResponse(
        document_id=record.document_id,
        status=record.status,
        filename=record.filename,
        error=record.error
    )
