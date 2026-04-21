"""
FastAPI dependencies: API Key validation and Rate Limiting.
"""

import hmac
from fastapi import Header, HTTPException, status
from app.config import settings

def verify_api_key(x_api_key: str = Header(...)):
    """
    Constant-time comparison to prevent timing attacks. (FR-06)
    """
    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key
