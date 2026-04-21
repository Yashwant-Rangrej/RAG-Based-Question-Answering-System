"""
Text extraction service for PDF and TXT files.

Design decisions:
  - pdfplumber is the primary PDF extractor: handles table-heavy and multi-column layouts well.
  - PyMuPDF (fitz) is the fallback: faster and handles image-heavy / scanned PDFs better
    (though scanned PDFs still require OCR which is out of scope).
  - Unicode NFKC normalisation: resolves ligatures (ﬁ→fi), full-width chars, and other
    compatibility equivalents that break token boundaries during chunking.
  - Null byte stripping: prevents downstream embedding / DB issues.
"""

import unicodedata
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def _normalise(text: str) -> str:
    """NFKC normalise, strip null bytes, collapse excessive whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    # Collapse runs of blank lines to at most two newlines
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber with PyMuPDF fallback."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
        if pages:
            log.info("pdf_extracted", engine="pdfplumber", pages=len(pages), path=str(path))
            return "\n\n".join(pages)
        raise ValueError("pdfplumber returned empty text — trying fallback")
    except Exception as primary_err:
        log.warning("pdfplumber_failed", error=str(primary_err), path=str(path))
        # --- PyMuPDF fallback ---
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            if any(pages):
                log.info("pdf_extracted", engine="pymupdf", pages=len(pages), path=str(path))
                return "\n\n".join(p for p in pages if p)
        except Exception as fallback_err:
            log.error("pymupdf_failed", error=str(fallback_err), path=str(path))
        raise RuntimeError(f"Could not extract text from PDF: {path}") from primary_err


def _extract_txt(path: Path) -> str:
    """Read plain text file with UTF-8 then latin-1 fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = path.read_text(encoding=encoding)
            log.info("txt_extracted", encoding=encoding, path=str(path))
            return text
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode text file: {path}")


def extract_text(path: Path, mime_type: str) -> str:
    """
    Main entry point: detect file type and extract plain text.

    Args:
        path: Absolute path to the uploaded file.
        mime_type: MIME type detected at upload (e.g. 'application/pdf', 'text/plain').

    Returns:
        Normalised plain text string.

    Raises:
        ValueError: For unsupported MIME types.
        RuntimeError: If extraction fails for all strategies.
    """
    if mime_type == "application/pdf" or path.suffix.lower() == ".pdf":
        raw = _extract_pdf(path)
    elif mime_type in ("text/plain",) or path.suffix.lower() == ".txt":
        raw = _extract_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

    normalised = _normalise(raw)
    log.info(
        "text_extracted",
        path=str(path),
        mime_type=mime_type,
        char_count=len(normalised),
    )
    return normalised
