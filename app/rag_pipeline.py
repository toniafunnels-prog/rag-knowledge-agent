"""
RAG Pipeline — core retrieval-augmented generation logic.

Architecture:
  1. Documents are chunked and embedded, stored in a local Chroma vector store.
  2. On query: embed the query, retrieve top-k relevant chunks.
  3. Chunks + query are sent to an LLM (Anthropic Claude) with a grounding prompt.
  4. Response includes citations back to source chunks (source filename + snippet).

This is intentionally provider-agnostic where possible so it can be swapped
between Anthropic / OpenAI / AWS Bedrock with minimal changes (see README
"Deploying on AWS Bedrock" section for that swap).
"""

import os
import glob
from dataclasses import dataclass
from typing import List

import chromadb
from anthropic import Anthropic
import voyageai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 120       # overlap between chunks to preserve context across boundaries
TOP_K = 4                 # number of chunks retrieved per query
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a knowledge-base assistant. Answer the user's question
using ONLY the provided context. If the context does not contain the answer,
say so clearly instead of guessing. Always cite which source document(s)
your answer comes from using the format [Source: filename].
Keep answers concise and directly useful."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


class VoyageEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Embedding function using Voyage AI — Anthropic's recommended embedding
    provider (Anthropic does not offer its own embedding model). This keeps
    the entire stack Claude-ecosystem with zero OpenAI dependency.

    Requires a Voyage AI API key: https://www.voyageai.com (separate signup,
    generous free tier as of this writing). Set VOYAGE_API_KEY in your
    environment.

    Model used: voyage-3.5-lite (fast, cheap, strong general-purpose
    retrieval quality) — swap to "voyage-3.5" for higher accuracy on
    harder/longer documents.
    """

    def __init__(self, model: str = "voyage-3.5-lite"):
        self._client = voyageai.Client()  # reads VOYAGE_API_KEY from environment
        self._model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        # input_type differs for documents vs queries; Chroma calls __call__
        # for documents and embed_query (falls back to __call__) for queries.
        # We use "document" here as the default; see embed_query override below.
        result = self._client.embed(input, model=self._model, input_type="document")
        return result.embeddings

    def embed_query(self, input: List[str]) -> List[List[float]]:
        result = self._client.embed(input, model=self._model, input_type="query")
        return result.embeddings

    def name(self) -> str:
        return "voyage_embedding"


class RAGPipeline:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Voyage AI embeddings — Anthropic's recommended embedding provider.
        # Keeps this project 100% Claude-ecosystem, zero OpenAI dependency.
        self.embed_fn = VoyageEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn
        )
        self.llm = Anthropic()  # reads ANTHROPIC_API_KEY from environment

    # -- Ingestion ----------------------------------------------------------
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return [c.strip() for c in chunks if c.strip()]

    def ingest_directory(self, directory: str = DATA_DIR) -> int:
        """Reads all .txt/.md files in a directory, chunks, and stores them."""
        files = glob.glob(os.path.join(directory, "*.txt")) + glob.glob(
            os.path.join(directory, "*.md")
        )
        total_chunks = 0
        for filepath in files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self._chunk_text(text)
            ids = [f"{filename}::{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename} for _ in chunks]
            self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
            total_chunks += len(chunks)
        return total_chunks

    # -- Retrieval + generation ----------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[RetrievedChunk]:
        results = self.collection.query(query_texts=[query], n_results=k)
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(RetrievedChunk(text=doc, source=meta["source"], distance=dist))
        return chunks

    def answer(self, query: str) -> dict:
        retrieved = self.retrieve(query)
        context_block = "\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in retrieved
        )
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        response = self.llm.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return {
            "answer": answer_text,
            "sources": list({c.source for c in retrieved}),
            "retrieved_chunks": [
                {"source": c.source, "text": c.text[:200], "relevance": round(1 - c.distance, 3)}
                for c in retrieved
            ],
        }
