"""Persistent Chroma retrieval for trusted public parking-space inventory only."""

from threading import Lock

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from openai import APIError, OpenAIError
from pydantic import ValidationError

from guardrails import PrivacyGuard
from storage.config import StorageConfig
from storage.spaces import ParkingSpace, SpaceFilters


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

    Metadata and detected PII are checked before embedding. Trusted source review is
    still required: automatic detection cannot identify all PII. Never index chats.
    """

    def __init__(
        self, config: StorageConfig, *, embeddings: Embeddings | None = None, privacy_guard: PrivacyGuard | None = None
    ) -> None:
        self.config = config
        self.privacy_guard = privacy_guard if privacy_guard is not None else PrivacyGuard()
        self._embeddings = embeddings if embeddings is not None else _LazyAzureEmbeddings(config)
        self._store: Chroma | None = None
        self._lock = Lock()
        self._seed_lock = Lock()

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

    def initialize(self) -> None:
        """Open the collection and immediately seed it only if missing/empty.

        A populated collection is reused unchanged, without even loading the fixture.
        This is a server-startup operation, never a conversation/graph operation.
        """
        from storage.seed import load_documents

        with self._seed_lock:
            store = self._get_store()
            if not store.get(limit=1, include=[])["ids"]:
                self.upsert_documents(load_documents())

    def list_space_ids(self, location_id: str) -> list[str]:
        """Read scoped document IDs only, without embeddings or inventory prose."""
        where = SpaceFilters().where(location_id)
        ids = self._get_store().get(where=where, include=[])["ids"]
        for identifier in ids:
            prefix = f"{location_id}:space:"
            if not identifier.startswith(prefix):
                raise ValueError("Static inventory contains a noncanonical space ID.")
            SpaceFilters(space_id=identifier.removeprefix(prefix))
        return ids

    def get_spaces(self, ids: list[str], *, location_id: str) -> list[Document]:
        """Read selected scoped inventory records, preserving the selected order."""
        where = SpaceFilters().where(location_id)
        if not ids:
            return []
        if len(set(ids)) != len(ids):
            raise ValueError("Selected space IDs must be unique.")
        result = self._get_store().get(ids=ids, where=where, include=["documents", "metadatas"])
        documents = {
            identifier: Document(id=identifier, page_content=text, metadata=metadata)
            for identifier, text, metadata in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        }
        if set(documents) != set(ids):
            raise RuntimeError("Selected parking inventory is missing or outside the configured location.")
        for document in documents.values():
            ParkingSpace.validate_document(document)
        return [documents[identifier] for identifier in ids]

    def hourly_price_summary(self, location_id: str) -> dict:
        """Aggregate scoped numeric metadata only, never embed or expose all inventory prose."""
        result = self._get_store().get(where=SpaceFilters().where(location_id), include=["metadatas"])
        rates = []
        for identifier, metadata in zip(result["ids"], result["metadatas"], strict=True):
            ParkingSpace.validate_document(
                Document(id=identifier, page_content="Public rate metadata.", metadata=metadata)
            )
            self.privacy_guard.check_public(metadata)
            rates.append(metadata["hourly_price_cents"])
        return (
            {"status": "inventory", "currency": "USD", "minimum_cents": min(rates), "maximum_cents": max(rates)}
            if rates
            else {"status": "missing", "currency": "USD"}
        )

    @staticmethod
    def validate_documents(documents: list[Document]) -> list[str]:
        """Validate a complete batch without creating resources or calling embeddings."""
        ids: list[str] = []
        for document in documents:
            ParkingSpace.validate_document(document)
            assert document.id is not None
            ids.append(document.id)
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique within a batch.")
        return ids

    def upsert_documents(self, documents: list[Document]) -> list[str]:
        """Insert/update by stable document ID; validate the entire batch before embedding."""
        ids = self.validate_documents(documents)
        for document in documents:
            self.privacy_guard.check_document(document)
        if not documents:
            return []
        store = self._get_store()
        existing = store.get(ids=ids, include=["metadatas"])
        if any(
            metadata.get("schema_version") != documents[0].metadata["schema_version"]
            for metadata in existing["metadatas"]
        ):
            raise ValueError(
                "Choose a fresh Chroma collection for a schema migration. Existing records were not changed."
            )
        return store.add_documents(documents, ids=ids)

    def search(
        self,
        query: str,
        *,
        location_id: str,
        k: int = 4,
        filters: SpaceFilters | None = None,
        eligible_ids: list[str] | None = None,
    ) -> list[Document]:
        """Rank scoped inventory, optionally restricted to Python-selected eligible IDs.

        This store does not decide availability. The caller derives eligibility from
        dynamic data; the additional filter cannot broaden location/schema scope.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonblank text.")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")
        if filters is not None and not isinstance(filters, SpaceFilters):
            raise TypeError("filters must be validated SpaceFilters, not an arbitrary metadata query.")
        where = (filters if filters is not None else SpaceFilters()).where(location_id)
        if eligible_ids is not None:
            if len(set(eligible_ids)) != len(eligible_ids):
                raise ValueError("Eligible document IDs must be unique.")
            space_ids = []
            prefix = f"{location_id}:space:"
            for identifier in eligible_ids:
                if not isinstance(identifier, str) or not identifier.startswith(prefix):
                    raise ValueError("Eligible document IDs must belong to the configured location.")
                space_id = identifier.removeprefix(prefix)
                SpaceFilters(space_id=space_id)
                space_ids.append(space_id)
            if not space_ids:
                return []
            where["$and"].append({"space_id": {"$in": space_ids}})
        store = self._get_store()
        # Avoid embedding calls on unseeded stores and clamp k to available documents.
        available = store.get(where=where, limit=k, include=[])["ids"]
        if not available:
            return []
        return store.similarity_search(query, k=len(available), filter=where)

    def delete_documents(self, ids: list[str]) -> None:
        """Delete explicit IDs only; an empty list must never clear a collection."""
        if any(not identifier.strip() for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be nonblank and unique.")
        if ids:
            self._get_store().delete(ids=ids)
