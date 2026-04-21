"""
Basic tests for extractor module.
"""

from pathlib import Path
from app.services.extractor import extract_text

def test_normalisation():
    from app.services.extractor import _normalise
    input_text = "Hello\x00World\n\n\n\nNew\r\nLine"
    output = _normalise(input_text)
    assert "\x00" not in output
    assert "\n\n\n" not in output
    assert "HelloWorld" in output
