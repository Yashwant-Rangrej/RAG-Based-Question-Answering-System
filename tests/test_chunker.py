"""
Basic tests for chunking logic.
"""

from app.services.chunker import chunk_text

def test_chunker_basic():
    text = "This is a sentence. This is another sentence. And a third one here."
    # With small chunk size to trigger splits
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)

def test_chunker_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
