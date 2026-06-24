from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import chromadb
import chromadb.telemetry.product.posthog as chroma_posthog
import requests
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.services.ingestion import ParsedDocument


LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _disable_chroma_telemetry() -> None:
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    chroma_posthog.posthog.disabled = True
    chroma_posthog.Posthog._direct_capture = lambda self, event: None


_disable_chroma_telemetry()


class VectorStoreError(RuntimeError):
    """Raised when vector persistence or retrieval fails."""


class OllamaEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embedding_model
        self.timeout = 90

    @property
    def model_name(self) -> str:
        return self.model

    def status(self) -> str:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return "healthy" if response.ok else "unhealthy"
        except requests.RequestException:
            return "unhealthy"

    def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
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


class GeminiEmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key.strip()
        self.model = settings.gemini_embedding_model.strip()
        self.base_url = settings.gemini_api_base_url
        self.api_version = settings.gemini_embedding_api_version
        self.timeout = settings.llm_timeout_seconds
        self.batch_size = settings.gemini_embedding_batch_size
        self.batch_delay_seconds = settings.gemini_embedding_batch_delay_seconds
        self.max_retries = settings.gemini_embedding_max_retries

    @property
    def model_name(self) -> str:
        return self.model

    def status(self) -> str:
        return "configured" if self._has_api_key() else "missing_api_key"

    def embed(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        embeddings: list[list[float]] = []
        batches = range(0, len(texts), self.batch_size)
        for batch_index, start in enumerate(batches):
            if batch_index and self.batch_delay_seconds:
                time.sleep(self.batch_delay_seconds)
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size], task_type=task_type))
        return embeddings

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not self._has_api_key():
            raise VectorStoreError("GEMINI_API_KEY or GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        if not texts:
            return []

        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in texts
            ]
        }
        url = f"{self.base_url}/{self.api_version}/{model_path}:batchEmbedContents"
        try:
            response = self._post_with_retries(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise VectorStoreError("Gemini returned an invalid embedding batch")

            vectors: list[list[float]] = []
            for embedding in embeddings:
                values = (embedding or {}).get("values")
                if not isinstance(values, list) or not values:
                    raise VectorStoreError("Gemini returned an empty embedding")
                vectors.append([float(value) for value in values])
            return vectors
        except requests.RequestException as exc:
            LOGGER.exception("Gemini embedding request failed")
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 429:
                raise VectorStoreError(
                    "Gemini embedding quota was exceeded while indexing this document. "
                    "Wait a few minutes and retry, or lower GEMINI_EMBEDDING_BATCH_SIZE / increase "
                    "GEMINI_EMBEDDING_BATCH_DELAY_SECONDS in .env."
                ) from exc
            raise VectorStoreError(
                f"Unable to generate Gemini embeddings with '{self.model}'. "
                "Check GEMINI_API_KEY and network access."
            ) from exc

    def _has_api_key(self) -> bool:
        key = self.api_key.strip()
        return bool(key and not key.lower().startswith(("replace-", "your-")))

    def _post_with_retries(self, url: str, **kwargs: Any) -> requests.Response:
        last_response: requests.Response | None = None
        for attempt in range(self.max_retries):
            response = requests.post(url, **kwargs)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response
            last_response = response
            retry_after = self._retry_after_seconds(response)
            delay_seconds = retry_after if retry_after is not None else min(60.0, 2.0 * (attempt + 1) ** 2)
            LOGGER.warning(
                "Gemini embedding request returned %s; retrying in %.1f seconds",
                response.status_code,
                delay_seconds,
            )
            time.sleep(delay_seconds)
        return last_response or requests.post(url, **kwargs)

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None


class VectorStoreService:
    def __init__(self) -> None:
        settings = get_settings()
        self.embedding_provider = settings.embedding_provider
        self.embedding_client = self._build_embedding_client()
        self.collection_name = self._collection_name(settings.chroma_collection_name, self.embedding_provider, self.embedding_client.model_name)
        settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.vectorstore_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_client.model_name,
            },
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def add_document(self, document: ParsedDocument) -> int:
        try:
            chunks = [chunk.strip() for chunk in self.splitter.split_text(document.content) if chunk.strip()]
            if not chunks:
                raise VectorStoreError(f"No chunks generated for {document.source_name}")

            embeddings = self.embedding_client.embed(chunks, task_type="RETRIEVAL_DOCUMENT")
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

    def query(self, query_text: str, k: int | None = None) -> list[dict[str, Any]]:
        try:
            if not query_text.strip():
                raise VectorStoreError("Query text cannot be empty")
            top_k = k or get_settings().top_k_results
            query_embedding = self.embedding_client.embed([query_text], task_type="RETRIEVAL_QUERY")[0]
            available_count = self.collection.count()
            if available_count == 0:
                return []
            n_results = min(max(1, top_k), max(1, available_count))
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
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

    @staticmethod
    def _build_embedding_client() -> OllamaEmbeddingClient | GeminiEmbeddingClient:
        settings = get_settings()
        if settings.embedding_provider == "gemini":
            return GeminiEmbeddingClient()
        return OllamaEmbeddingClient()

    @staticmethod
    def _collection_name(base_name: str, provider: str, model: str) -> str:
        if provider == "ollama":
            return base_name
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{provider}_{model}").strip("_").lower()
        name = f"{base_name}_{suffix}"
        return name[:63].strip("_-") or base_name
