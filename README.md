# RAG-Based Question Answering System

A local-first backend API for context-aware document QA using Retrieval-Augmented Generation (RAG).

## Features
- **Document Ingestion**: Upload PDF or TXT files.
- **Semantic Retrieval**: FAISS vector store with `all-MiniLM-L6-v2` embeddings.
- **Local LLM**: Grounded Q&A via Ollama (Mistral).
- **Explainable**: Citations and similarity scores returned with every answer.
- **Performant**: Async FastAPI with background processing.

## Tech Stack
- **Framework**: FastAPI
- **Embeddings**: Sentence Transformers
- **Vector DB**: FAISS (local)
- **LLM**: Ollama
- **PDF Extraction**: pdfplumber

## Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama
```bash
ollama pull mistral
```

### 4. Configuration
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

### 5. Run the Application
```bash
uvicorn app.main:app --reload
```

## API Usage

### Upload a Document
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: your-secret-key" \
  -F "file=@/path/to/document.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings in the document?"}'
```

---
Built with transparency and performance in mind. No black-box RAG frameworks.
