"""
Token-aware sliding-window text chunker.

Design rationale:
  - Uses tiktoken (cl100k_base tokeniser) for accurate token counts matching OpenAI /
    sentence-transformer tokenisation behaviour.
  - 300–500 token range: balances context richness against embedding signal dilution.
    > 800 tokens → embedding vector averages too many concepts, lowering similarity precision.
    < 100 tokens → loses intra-sentence coherence, fragments context.
  - 15% overlap (default 60 tokens on 400-token chunks): ensures information at chunk
    boundaries is captured by at least two chunks, preventing retrieval gaps.
  - Sentence-boundary awareness: the chunker tries to end chunks at sentence endings
    (., !, ?) to preserve semantic units.
"""

import re
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Cached tokeniser — loaded once at module import time.
_TOKENISER = None


def _get_tokeniser():
    global _TOKENISER
    if _TOKENISER is None:
        import tiktoken
        # cl100k_base: used by GPT-3.5/4 and compatible with sentence-transformers token counts.
        _TOKENISER = tiktoken.get_encoding("cl100k_base")
    return _TOKENISER


def _token_count(text: str) -> int:
    return len(_get_tokeniser().encode(text))


def _find_sentence_boundary(tokens: list[int], target_idx: int, window: int = 20) -> int:
    """
    Search backward from target_idx within 'window' tokens for a sentence-ending token.
    Returns an adjusted index at a sentence boundary, or target_idx if none found.
    """
    enc = _get_tokeniser()
    search_start = max(0, target_idx - window)
    # Decode a small trailing segment to check for sentence endings
    segment = enc.decode(tokens[search_start:target_idx])
    # Walk backwards looking for sentence terminators
    for i, char in enumerate(reversed(segment)):
        if char in ".!?":
            byte_pos = len(segment) - i
            # Re-encode the portion up to the boundary to count tokens
            prefix_tokens = enc.encode(segment[:byte_pos])
            return search_start + len(prefix_tokens)
    return target_idx


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[str]:
    """
    Split text into overlapping token-aware chunks.

    Args:
        text: Normalised plain text to chunk.
        chunk_size: Target chunk size in tokens (recommended 300–500).
        chunk_overlap: Overlap between consecutive chunks in tokens (recommended ~15% of chunk_size).

    Returns:
        List of text chunk strings.
    """
    if not text.strip():
        return []

    enc = _get_tokeniser()
    tokens = enc.encode(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return []

    chunks: list[str] = []
    start = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)

        # Attempt to align end to a sentence boundary (only when not at the final chunk)
        if end < total_tokens:
            end = _find_sentence_boundary(tokens, end)

        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()

        if chunk_text_str:
            chunks.append(chunk_text_str)

        # Advance start by (chunk_size - overlap). Always move forward at least 1 token.
        step = max(1, chunk_size - chunk_overlap)
        start += step

    log.info(
        "text_chunked",
        total_tokens=total_tokens,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
