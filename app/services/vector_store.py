from __future__ import annotations

import hashlib
import logging
from typing import Any

import chromadb
import requests
from chromadb.api.models.Collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.services.ingestion import ParsedDocument


LOGGER = logging.getLogger(__name__)
COLLECTION_NAME = "document_assistant_sources"


class VectorStoreError(RuntimeError):
    """Raised when vector persistence or retrieval fails."""


class OllamaEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embedding_model
        self.timeout = 90

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.append(self._embed_one(text))
        return embeddings

    def _embed_one(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout)
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise VectorStoreError("Ollama returned an empty embedding")
            return [float(value) for value in embedding]
        except requests.RequestException as exc:
            raise VectorStoreError(
                f"Unable to reach Ollama embeddings at {self.base_url}. "
                f"Confirm 'ollama serve' is running and '{self.model}' is pulled."
            ) from exc


class VectorStoreService:
    def __init__(self) -> None:
        settings = get_settings()
        settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_client = OllamaEmbeddingClient()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def add_document(self, document: ParsedDocument) -> int:
        try:
            chunks = [chunk.strip() for chunk in self.splitter.split_text(document.content) if chunk.strip()]
            if not chunks:
                raise VectorStoreError(f"No chunks generated for {document.source_name}")

            embeddings = self.embedding_client.embed(chunks)
            ids = [self._chunk_id(document.source_name, index, chunk) for index, chunk in enumerate(chunks)]
            metadatas = [
                {
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                }
                for index in range(len(chunks))
            ]

            self.collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            return len(chunks)
        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise
            LOGGER.exception("Failed to add document %s", document.source_name)
            raise VectorStoreError(f"Failed to add document {document.source_name}: {exc}") from exc

    def query(self, query_text: str, k: int = 3) -> list[dict[str, Any]]:
        try:
            if not query_text.strip():
                raise VectorStoreError("Query text cannot be empty")
            query_embedding = self.embedding_client.embed([query_text])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, k),
                include=["documents", "metadatas", "distances"],
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            return [
                {
                    "content": document,
                    "metadata": metadata or {},
                    "distance": distance,
                    "source": (metadata or {}).get("source", "unknown"),
                }
                for document, metadata, distance in zip(documents, metadatas, distances)
            ]
        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise
            LOGGER.exception("Vector query failed")
            raise VectorStoreError(f"Vector query failed: {exc}") from exc

    @staticmethod
    def _chunk_id(source_name: str, index: int, chunk: str) -> str:
        digest = hashlib.sha256(f"{source_name}:{index}:{chunk}".encode("utf-8")).hexdigest()
        return f"{source_name}:{index}:{digest[:16]}"
