from __future__ import annotations

import importlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import SecretStr

from .models import SourceDocument


class PineconeHistoryStore:
    """PineconeVectorStore adapter with safe batch/resume support."""

    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        namespace: str,
        openai_api_key: str,
        embedding_model: str,
        dimension: int = 3072,
    ) -> None:
        self.pc: Any = Pinecone(api_key=pinecone_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.namespace = namespace
        self.embedder = OpenAIEmbeddings(
            api_key=SecretStr(openai_api_key), model=embedding_model, dimensions=dimension
        )
        self.dimension = dimension
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _vectorstore_class() -> type[Any]:
        """Load lazily so local dry-runs work before the approved install."""
        try:
            module = importlib.import_module("langchain_pinecone")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Live operations require langchain-pinecone==0.2.13."
            ) from exc
        return module.PineconeVectorStore

    def ensure_index(self, allow_create: bool = False) -> None:
        if not self.pc.has_index(self.index_name):
            if not allow_create:
                raise RuntimeError(
                    f"Pinecone index does not exist: {self.index_name}. "
                    "Create it only after approval or pass --allow-index-create."
                )
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="disabled",
            )
            for _ in range(60):
                description = self.pc.describe_index(self.index_name)
                if description.status.get("ready"):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Timed out while waiting for the Pinecone index.")

        description = self.pc.describe_index(self.index_name)
        if description.dimension is None:
            raise RuntimeError("Could not verify the Pinecone index dimension.")
        if description.dimension != self.dimension:
            raise RuntimeError(
                f"Index dimension {description.dimension} does not match {self.dimension}."
            )
        if str(description.metric) != "cosine":
            raise RuntimeError(f"Pinecone metric must be cosine, not {description.metric}.")

    @staticmethod
    def _retryable(error: Exception) -> bool:
        status = getattr(error, "status", None) or getattr(error, "status_code", None)
        name = type(error).__name__.lower()
        return status == 429 or (isinstance(status, int) and status >= 500) or any(
            token in name for token in ("timeout", "connection", "ratelimit", "internalserver")
        )

    @classmethod
    def _with_retry(
        cls, operation: Callable[[], Any], attempts: int = 4, base_delay: float = 1.0
    ) -> Any:
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as error:
                if attempt + 1 >= attempts or not cls._retryable(error):
                    raise
                time.sleep(base_delay * (2**attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def metadata_for(document: SourceDocument) -> dict[str, Any]:
        return {
            **document.metadata,
            "text": document.text,
            "document_id": document.id,
            "title": document.title,
            "source_url": document.source_url,
        }

    @staticmethod
    def langchain_document(document: SourceDocument) -> Document:
        metadata = PineconeHistoryStore.metadata_for(document)
        metadata.pop("text")
        return Document(page_content=document.text, metadata=metadata)

    @classmethod
    def validate_metadata(cls, document: SourceDocument, limit_bytes: int = 40_960) -> int:
        metadata = cls.metadata_for(document)
        allowed_scalar = (str, int, float, bool)
        for key, value in metadata.items():
            valid_list = isinstance(value, list) and all(isinstance(item, str) for item in value)
            if not isinstance(value, allowed_scalar) and not valid_list:
                raise TypeError(
                    f"{document.id}: unsupported Pinecone metadata type "
                    f"{key}={type(value).__name__}"
                )
        size = len(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if size > limit_bytes:
            raise ValueError(
                f"{document.id}: metadata {size} bytes exceeds {limit_bytes} bytes"
            )
        return size

    def iter_batches(
        self,
        documents: list[SourceDocument],
        start_offset: int = 0,
        max_documents: int = 50,
        max_tokens: int = 200_000,
    ) -> Iterator[tuple[int, list[SourceDocument]]]:
        batch: list[SourceDocument] = []
        tokens = 0
        batch_start = start_offset
        for offset, document in enumerate(documents[start_offset:], start_offset):
            document_tokens = len(self.encoding.encode(document.text, disallowed_special=()))
            if batch and (len(batch) >= max_documents or tokens + document_tokens > max_tokens):
                yield batch_start, batch
                batch = []
                tokens = 0
                batch_start = offset
            batch.append(document)
            tokens += document_tokens
        if batch:
            yield batch_start, batch

    def upsert(
        self,
        documents: list[SourceDocument],
        batch_size: int = 50,
        start_offset: int = 0,
        allow_index_create: bool = False,
        progress: Callable[[int], None] | None = None,
    ) -> int:
        self.ensure_index(allow_create=allow_index_create)
        vectorstore_class = self._vectorstore_class()
        count = 0
        for start, batch in self.iter_batches(
            documents, start_offset=start_offset, max_documents=batch_size
        ):
            for document in batch:
                self.validate_metadata(document)
            langchain_documents = [self.langchain_document(document) for document in batch]
            ids = [document.id for document in batch]
            self._with_retry(
                lambda documents=langchain_documents, document_ids=ids: (
                    vectorstore_class.from_documents(
                        documents=documents,
                        embedding=self.embedder,
                        ids=document_ids,
                        index_name=self.index_name,
                        namespace=self.namespace,
                        pinecone_api_key=self.pinecone_api_key,
                    )
                )
            )
            count += len(batch)
            if progress is not None:
                progress(start + len(batch))
        return count

    def namespace_count(self) -> int:
        stats = self.pc.Index(self.index_name).describe_index_stats()
        namespace = (stats.namespaces or {}).get(self.namespace)
        return int(namespace.vector_count) if namespace else 0

    @staticmethod
    def _sample_documents_by_dataset(
        documents: list[SourceDocument], samples_per_dataset: int
    ) -> list[SourceDocument]:
        grouped: dict[str, list[SourceDocument]] = defaultdict(list)
        for document in documents:
            grouped[str(document.metadata.get("dataset_key", "unknown"))].append(document)
        selected = []
        for dataset_documents in grouped.values():
            ordered = sorted(dataset_documents, key=lambda item: item.id)
            positions = {0, len(ordered) // 2, len(ordered) - 1}
            selected.extend(
                ordered[position] for position in sorted(positions)[:samples_per_dataset]
            )
        return selected

    def verify_remote_corpus(
        self, documents: list[SourceDocument], samples_per_dataset: int = 3
    ) -> dict[str, Any]:
        """Verify every remote ID and representative full records without re-embedding."""
        expected_by_id = {document.id: document for document in documents}
        if len(expected_by_id) != len(documents):
            raise ValueError("Local corpus contains duplicate document IDs.")

        index = self.pc.Index(self.index_name)

        def list_all_ids() -> tuple[list[str], int]:
            remote_ids: list[str] = []
            pages = 0
            for page in index.list(namespace=self.namespace):
                pages += 1
                remote_ids.extend(str(item) for item in page)
            return remote_ids, pages

        remote_ids, list_pages = self._with_retry(list_all_ids)
        remote_id_set = set(remote_ids)
        expected_id_set = set(expected_by_id)
        missing = sorted(expected_id_set - remote_id_set)
        unexpected = sorted(remote_id_set - expected_id_set)
        if len(remote_ids) != len(remote_id_set) or missing or unexpected:
            raise RuntimeError(
                "Remote ID verification failed: "
                f"listed={len(remote_ids)}, unique={len(remote_id_set)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"missing_examples={missing[:5]}, unexpected_examples={unexpected[:5]}"
            )

        remote_by_dataset: Counter[str] = Counter(
            str(expected_by_id[document_id].metadata.get("dataset_key", "unknown"))
            for document_id in remote_ids
        )
        expected_by_dataset: Counter[str] = Counter(
            str(document.metadata.get("dataset_key", "unknown")) for document in documents
        )
        if remote_by_dataset != expected_by_dataset:
            raise RuntimeError(
                "Remote dataset counts do not match: "
                f"expected={dict(expected_by_dataset)}, actual={dict(remote_by_dataset)}"
            )

        sampled = self._sample_documents_by_dataset(documents, samples_per_dataset)
        sample_ids = [document.id for document in sampled]
        fetch_response = self._with_retry(
            lambda: index.fetch(ids=sample_ids, namespace=self.namespace)
        )
        fetched = fetch_response.vectors or {}
        for expected in sampled:
            record = fetched.get(expected.id)
            if record is None:
                raise RuntimeError(f"Sample record was not fetched: {expected.id}")
            metadata = dict(record.metadata or {})
            required = {
                "document_id": expected.id,
                "dataset_key": str(expected.metadata.get("dataset_key", "unknown")),
                "source_url": expected.source_url,
                "title": expected.title,
                "text": expected.text,
            }
            mismatched = [key for key, value in required.items() if metadata.get(key) != value]
            if mismatched:
                raise RuntimeError(
                    f"Sample metadata mismatch for {expected.id}: {mismatched}"
                )

        return {
            "remote_ids": len(remote_ids),
            "remote_by_dataset": dict(sorted(remote_by_dataset.items())),
            "list_pages": list_pages,
            "sample_records_verified": len(sampled),
            "fetch_requests": 1 if sampled else 0,
        }

    def _vectorstore(self) -> Any:
        return self._vectorstore_class()(
            index=self.pc.Index(self.index_name),
            embedding=self.embedder,
            namespace=self.namespace,
            text_key="text",
        )

    def as_retriever(
        self, top_k: int = 12, metadata_filter: dict[str, object] | None = None
    ) -> Any:
        search_kwargs: dict[str, Any] = {"k": top_k}
        if metadata_filter is not None:
            search_kwargs["filter"] = metadata_filter
        return self._vectorstore().as_retriever(search_kwargs=search_kwargs)

    def search(
        self, query: str, top_k: int = 12, metadata_filter: dict[str, object] | None = None
    ) -> list[SourceDocument]:
        kwargs: dict[str, Any] = {"k": top_k}
        if metadata_filter is not None:
            kwargs["filter"] = metadata_filter
        results = self._with_retry(
            lambda: self._vectorstore().similarity_search_with_score(query, **kwargs)
        )
        documents = []
        for document, score in results:
            metadata = dict(document.metadata)
            catalog_url = str(metadata.pop("source_url", ""))
            official_record_url = str(metadata.get("official_record_url", ""))
            if official_record_url:
                source_url = official_record_url
                metadata["source_url_type"] = "official_record"
                metadata["catalog_url"] = catalog_url
            else:
                source_url = catalog_url
                metadata["source_url_type"] = "dataset_catalog"
            documents.append(
                SourceDocument(
                    id=str(metadata.pop("document_id", metadata.pop("id", ""))),
                    text=document.page_content,
                    title=str(metadata.pop("title", "제목 없음")),
                    source_url=source_url,
                    metadata=metadata,
                    score=float(score),
                )
            )
        return documents
