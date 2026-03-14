# ── Embedding model (downloaded once, runs fully offline after that) ──────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Text splitting ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200  # characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 8     # how many chunks to retrieve per query

# ── ChromaDB (persisted to disk so no re-embedding on restart) ─────────────────
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "pdf_docs"

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "llama3.2"
