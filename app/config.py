"""
Application configuration via pydantic-settings.
All values are loaded from environment variables or .env file.
Rationale for defaults:
  - CHUNK_SIZE=400 tokens: balances context richness vs retrieval precision (300–500 range).
  - CHUNK_OVERLAP=60 tokens: ~15% of 400 — prevents information loss at chunk boundaries.
  - TOP_K=5: returns 5 chunks by default; enough context without exceeding typical LLM token budgets.
  - SIMILARITY_THRESHOLD=0.35: cosine similarity cutoff; below this indicates the query has no
    meaningful match in the corpus, triggering graceful degradation.
  - MAX_FILE_SIZE_MB=20: matches FR-01 requirement; configurable for larger corpora.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- API Security ---
    API_KEY: str = "dev-secret-key-12345"

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["*"]

    # --- File Storage ---
    STORAGE_DIR: Path = Path("storage")
    DATA_DIR: Path = Path("data")
    MAX_FILE_SIZE_MB: int = 20

    # --- Chunking ---
    # 400 tokens: mid-range of 300–500 token recommendation.
    # Balances semantic completeness per chunk with tight concept mapping.
    CHUNK_SIZE: int = 400
    # ~15% overlap prevents information loss at chunk boundaries.
    CHUNK_OVERLAP: int = 60

    # --- Retrieval ---
    # Default top-k=5; configurable up to 20 per FR-03.
    TOP_K: int = 5
    # Cosine similarity below 0.20 indicates no meaningful match — return graceful degradation message.
    SIMILARITY_THRESHOLD: float = 0.20

    # --- Embedding ---
    # all-MiniLM-L6-v2: 384-dim, fast, local, no API key required.
    # Consistent model_id stored per-document to guard against embedding-space mismatches.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32

    # --- LLM (Ollama) ---
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_TIMEOUT: int = 120  # seconds

    # --- Rate Limiting ---
    RATE_LIMIT: str = "60/minute"

    # --- App ---
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()

# Ensure required directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
