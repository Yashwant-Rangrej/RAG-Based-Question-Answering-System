"""
FastAPI application entry point.
"""

import time
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.routers import upload, status, ask, health

# 1. Setup Logging (FR stack)
structlog.configure()
log = structlog.get_logger()

# 2. Setup Rate Limiting (FR-06)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG-Based Question Answering API",
    description="A local-first RAG API using FastAPI, Sentence Transformers, FAISS, and Ollama.",
    version=settings.VERSION,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs latency for every request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    log.info(
        "request_processed",
        path=request.url.path,
        method=request.method,
        duration_ms=round(process_time, 2),
        status_code=response.status_code
    )
    return response

# 4. Routes
app.include_router(upload.router)
app.include_router(status.router)
app.include_router(ask.router)
app.include_router(health.router)

# Serve Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
