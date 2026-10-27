"""Persistent Chroma retrieval for trusted, public static parking documents only."""

from threading import Lock

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from openai import APIError, OpenAIError
from pydantic import ValidationError

from storage.config import StorageConfig


class _LazyAzureEmbeddings(Embeddings):
    """Allow opening/inspecting an empty database without credentials or model downloads."""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._client: Embeddings | None = None
        self._lock = Lock()

    def _get_client(self) -> Embeddings:
        with self._lock:
            if self._client is None:
                try:
                    self._client = AzureOpenAIEmbeddings(
                        azure_deployment=self._config.embedding_deployment,
                        model=self._config.embedding_model,
                        # Send text directly to Azure/DIAL, without a tiktoken model download.
                        check_embedding_ctx_length=False,
                        request_timeout=60,
                        max_retries=2,
                    )
                except (ValidationError, OpenAIError):
                    raise RuntimeError("Unable to initialize embeddings. Check the embedding configuration.") from None
            return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._get_client().embed_documents(texts)
        except APIError:
            raise RuntimeError("Unable to embed parking information. Check configuration or retry.") from None

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._get_client().embed_query(text)
        except APIError:
            raise RuntimeError("Unable to embed the query. Check configuration or retry.") from None


class StaticParkingStore:
    """Lazy, persistent vector store. Ingestion is a trusted Python API, not an LLM tool.

    Only static-topic metadata is accepted. This is not a PII detector: callers must
    review document text and sources before ingestion. Never index chats/reservations.
    """

    def __init__(self, config: StorageConfig, *, embeddings: Embeddings | None = None) -> None:
        self.config = config
        self._embeddings = embeddings if embeddings is not None else _LazyAzureEmbeddings(config)
        self._store: Chroma | None = None
        self._lock = Lock()

    def _get_store(self) -> Chroma:
        with self._lock:
            if self._store is None:
                self.config.chroma_directory.mkdir(parents=True, exist_ok=True)
                self._store = Chroma(
                    collection_name=self.config.chroma_collection,
                    persist_directory=str(self.config.chroma_directory),
                    embedding_function=self._embeddings,
                    client_settings=Settings(anonymized_telemetry=False),
                )
            return self._store

    @staticmethod
    def validate_documents(documents: list[Document]) -> list[str]:
        """Validate a complete batch without creating resources or calling embeddings."""
        ids: list[str] = []
        for document in documents:
            if not document.id or not document.id.strip() or not document.page_content.strip():
                raise ValueError("Static documents require a nonblank ID and text.")
            metadata = document.metadata
            if set(metadata) != {"location_id", "source", "topic"}:
                raise ValueError("Static metadata must contain only location_id, source, and topic.")
            if any(not isinstance(value, str) or not value.strip() for value in metadata.values()):
                raise ValueError("Static metadata values must be nonblank strings.")
            if metadata["topic"] not in ("general", "location", "booking_process"):
                raise ValueError("Only general, location, and booking_process topics belong in static storage.")
            ids.append(document.id)
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique within a batch.")
        return ids

    def upsert_documents(self, documents: list[Document]) -> list[str]:
        """Insert/update by stable document ID; validate the entire batch before embedding."""
        ids = self.validate_documents(documents)
        if not documents:
            return []
        return self._get_store().add_documents(documents, ids=ids)

    def search(self, query: str, *, location_id: str, k: int = 4) -> list[Document]:
        """Search one explicit location; empty stores return no invented information."""
        if not query.strip() or not location_id.strip():
            raise ValueError("Query and location_id must not be blank.")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")
        store = self._get_store()
        # Avoid embedding calls on unseeded stores and clamp k to available documents.
        available = store.get(where={"location_id": location_id}, limit=k, include=[])["ids"]
        if not available:
            return []
        return store.similarity_search(query, k=len(available), filter={"location_id": location_id})

    def delete_documents(self, ids: list[str]) -> None:
        """Delete explicit IDs only; an empty list must never clear a collection."""
        if any(not identifier.strip() for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be nonblank and unique.")
        if ids:
            self._get_store().delete(ids=ids)
