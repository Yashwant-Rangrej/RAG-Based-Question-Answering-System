import json
import os
import asyncio
from pathlib import Path

# Mock config to avoid loading full app if needed, but here we want the real services
from app.services.vector_store import vector_store
from app.services.embedder import embedder
from app.models.document import registry
from app.services.pipeline import run_ingestion_pipeline

async def rebuild_index():
    print("Starting index cleanup and rebuild...")
    
    # 1. Identify unique documents in storage
    storage_dir = Path("storage")
    unique_docs = {} # filename -> first_doc_id_seen
    doc_id_to_filename = {}
    
    for p in storage_dir.glob("*"):
        if "_" in p.name:
            parts = p.name.split("_")
            doc_id = parts[0]
            filename = "_".join(parts[1:])
            doc_id_to_filename[doc_id] = filename
            
            if filename not in unique_docs:
                unique_docs[filename] = doc_id
    
    # 2. Identify duplicates to remove
    to_delete_ids = [did for did in doc_id_to_filename if did not in unique_docs.values()]
    print(f"Found {len(unique_docs)} unique documents and {len(to_delete_ids)} duplicates.")

    # 3. Wipe current index files
    idx_path = Path("data/faiss.index")
    meta_path = Path("data/metadata.json")
    
    if idx_path.exists(): os.remove(idx_path)
    if meta_path.exists(): os.remove(meta_path)
    print("Wiped old index files.")

    # 4. Reset Vector Store state so it creates a fresh index
    import app.services.vector_store as vs_module
    # We force reset the singleton and the module-level instance
    vs_module.vector_store = vs_module.FAISSVectorStore()
    vstore = vs_module.vector_store
    
    # 5. Re-ingest unique documents
    for filename, doc_id in unique_docs.items():
        file_path = storage_dir / f"{doc_id}_{filename}"
        print(f"Indexing: {filename}...")
        mime = "application/pdf" if filename.endswith(".pdf") else "text/plain"
        await run_ingestion_pipeline(doc_id, str(file_path), mime)

    # 6. Delete redundant files from storage
    for did in to_delete_ids:
        fname = doc_id_to_filename[did]
        fpath = storage_dir / f"{did}_{fname}"
        if fpath.exists():
            print(f"Deleting duplicate file: {fpath.name}")
            os.remove(fpath)
        # Also remove from registry
        registry.remove(did)
    
    # Skip registry._save() as it's typically in-memory in this implementation
    # but we can call any persistence method if it exists.
    print("Index and storage cleanup complete!")

if __name__ == "__main__":
    asyncio.run(rebuild_index())
