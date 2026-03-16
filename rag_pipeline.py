import os
import shutil
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

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
        # ── Embedding model (BGE — better retrieval accuracy) ─────────────────
        print("[RAG] Loading embedding model (first run will download it)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # required for BGE
        )
        print("[RAG] Embedding model ready.")

        # ── Cross-encoder re-ranker ────────────────────────────────────────────
        print("[RAG] Loading re-ranker model (first run will download it)...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)
        print("[RAG] Re-ranker ready.")

        self.vector_store  = None
        self.current_model = config.DEFAULT_MODEL
        self.loaded_pdf    = None

    # ── Ollama helpers ────────────────────────────────────────────────────────

    def get_available_models(self) -> list[str]:
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
        """Load PDF → chunk → embed → store in ChromaDB. Returns chunk count."""

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
        print(f"[RAG] Ingestion complete. {len(chunks)} chunks stored.")
        return len(chunks)

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def _rerank(self, question: str, docs: list) -> list:
        """Score every (question, chunk) pair and return top RERANKER_TOP_K docs."""
        if not docs:
            return docs
        pairs  = [(question, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:config.RERANKER_TOP_K]]

    # ── Format docs ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
            for doc in docs
        )

    # ── Query (streaming) ─────────────────────────────────────────────────────

    def stream_query(self, question: str):
        """Retrieve → re-rank → stream answer."""
        if not self.vector_store:
            yield "Please upload and process a PDF first."
            return

        # Step 1: fetch a large candidate pool
        fetch_k = config.TOP_K_RESULTS * 3
        candidates = self.vector_store.similarity_search(question, k=fetch_k)

        # Step 2: re-rank and keep the best ones
        best_docs = self._rerank(question, candidates)
        context   = self._format_docs(best_docs)

        # Step 3: stream answer from LLM
        llm    = ChatOllama(
            model=self.current_model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain  = prompt | llm | StrOutputParser()

        for token in chain.stream({"context": context, "question": question}):
            yield token

    # ── Model switching ───────────────────────────────────────────────────────

    def set_model(self, model_name: str):
        self.current_model = model_name

    # ── Reset ─────────────────────────────────────────────────────────────────

    def clear(self):
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)
        self.vector_store = None
        self.loaded_pdf   = None
        print("[RAG] Cleared all stored data.")
