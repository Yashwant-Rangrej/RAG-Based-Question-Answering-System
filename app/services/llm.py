"""
Ollama adapter for LLM answer generation.

Design rationale:
  - Communicates with a local Ollama instance via HTTP.
  - Mistral-7B is the default model: good balance of reasoning and speed for a local setup.
  - Grounded prompt engineering: instructs the model to answer ONLY from context or
    admit ignorance, reducing hallucinations (FR-04 requirement).
  - Timeout management: local LLM inference can be slow on CPU; default 120s ensures
    the request doesn't hang indefinitely but gives the model time to reason.
"""

import httpx
import structlog
from app.config import settings

log = structlog.get_logger(__name__)

class OllamaAdapter:
    """Service to interact with local Ollama API."""

    def __init__(self, url: str = settings.OLLAMA_URL, model: str = settings.OLLAMA_MODEL):
        self.url = f"{url.rstrip('/')}/api/generate"
        self.model = model
        self.timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT)

    async def stream_answer(self, query: str, context_chunks: list[str]):
        """
        Generate a grounded answer in a streaming fashion.
        Yields chunks of text as they arrive from Ollama.
        """
        context_str = "\n\n".join([f"--- Context {i+1} ---\n{chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = (
            "You are a professional RAG (Retrieval-Augmented Generation) assistant.\n"
            "Your task is to answer the user's question accurately using the provided context.\n\n"
            "Guidelines:\n"
            "1. If the provided context contains the answer, prioritize it and cite facts exactly as they appear.\n"
            "2. If the user asks to 'answer questions', 'summarize', or 'list facts' based on the document, and the context contains relevant chunks, perform the task using those chunks.\n"
            "3. If the context is empty or completely unrelated to the user's current request, use your general knowledge but clearly state that the information is not grounded in the provided documents.\n"
            "4. Be concise, technical, and professional. Use formatting (bullet points, bold text) for clarity.\n\n"
            f"### Provided Context:\n{context_str}\n\n"
            f"### User Question:\n{query}\n\n"
            "### Answer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096
            }
        }

        log.info("llm_streaming_start", model=self.model)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        import json
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
        except Exception as e:
            log.error("llm_streaming_failed", error=str(e))
            yield f"\n\n[Error during generation: {str(e)}]"

    async def check_health(self) -> bool:
        """Ping Ollama to verify availability."""
        try:
            base_url = self.url.replace("/api/generate", "/api/tags")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(base_url)
                return response.status_code == 200
        except Exception:
            return False

# Global instance
llm_service = OllamaAdapter()
