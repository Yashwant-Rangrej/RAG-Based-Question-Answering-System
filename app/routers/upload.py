"""
Router for document uploads.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, status
from app.config import settings
from app.models.schemas import UploadResponse, DocumentStatus
from app.models.document import registry
from app.services.pipeline import run_ingestion_pipeline
from app.dependencies import verify_api_key

router = APIRouter(prefix="/upload", tags=["Ingestion"])

@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _api_key: str = Depends(verify_api_key)
):
    """
    Upload a PDF or TXT file for processing.
    """
    # 1. Validate MIME Type (FR-01)
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file format. Only PDF and TXT are accepted."
        )

    # 2. Validate File Size
    # Note: actual size check usually requires reading or spooling
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB."
        )

    # 3. Create unique ID and save locally
    doc_id = str(uuid.uuid4())
    save_path = settings.STORAGE_DIR / f"{doc_id}_{file.filename}"
    
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 4. Register document and trigger background job (FR-02)
    registry.register(doc_id, file.filename)
    background_tasks.add_task(
        run_ingestion_pipeline,
        doc_id,
        str(save_path),
        file.content_type
    )

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        size_bytes=file_size,
        status=DocumentStatus.PENDING,
        message="Document uploaded and processing started."
    )
