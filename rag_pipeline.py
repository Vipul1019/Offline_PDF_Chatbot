import os
import shutil
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using the document context provided below.

Guidelines:
- Base your answer primarily on the context.
- If the context contains partial information, use it and mention what is covered.
- Only say you cannot find information if the context is truly completely unrelated to the question.
- Be thorough and detailed in your answer.
- If asked for a summary, cover all key points from the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self):
        print("[RAG] Loading embedding model (first run will download it)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.vector_store  = None
        self.retriever     = None
        self.chain         = None
        self.current_model = config.DEFAULT_MODEL
        self.loaded_pdf    = None
        print("[RAG] Embedding model ready.")

    # ── Ollama helpers ────────────────────────────────────────────────────────

    def get_available_models(self) -> list[str]:
        """Return list of model names installed in Ollama."""
        try:
            resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.ConnectionError:
            pass
        return []

    def is_ollama_running(self) -> bool:
        try:
            requests.get(config.OLLAMA_BASE_URL, timeout=3)
            return True
        except Exception:
            return False

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> int:
        """Load a PDF, chunk it, embed it, persist to ChromaDB. Returns chunk count."""

        # Clear previous collection so old data doesn't bleed into new PDF
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)

        print(f"[RAG] Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[RAG] Split into {len(chunks)} chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_DB_PATH,
            collection_name=config.COLLECTION_NAME,
        )

        self.loaded_pdf = os.path.basename(pdf_path)
        self._build_chain()
        print(f"[RAG] Ingestion complete. {len(chunks)} chunks stored.")
        return len(chunks)

    # ── Chain builder ─────────────────────────────────────────────────────────

    def _build_chain(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.TOP_K_RESULTS, "fetch_k": config.TOP_K_RESULTS * 3},
        )

        llm = ChatOllama(
            model=self.current_model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
        )

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        self.chain = (
            {
                "context":  self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for doc in docs
        )

    # ── Query (streaming) ─────────────────────────────────────────────────────

    def stream_query(self, question: str):
        """Yield response tokens one by one for streaming UI."""
        if not self.chain:
            yield "Please upload and process a PDF first."
            return
        for token in self.chain.stream(question):
            yield token

    # ── Model switching ───────────────────────────────────────────────────────

    def set_model(self, model_name: str):
        if model_name == self.current_model:
            return
        self.current_model = model_name
        if self.vector_store:
            self._build_chain()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def clear(self):
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)
        self.vector_store = None
        self.retriever    = None
        self.chain        = None
        self.loaded_pdf   = None
        print("[RAG] Cleared all stored data.")
