from __future__ import annotations

import re

from .models import SourceDocument

REQUIRED_CORPUS_FIELDS = {
    "dataset",
    "dataset_key",
    "literature",
    "era",
    "article_id",
    "document_type",
    "provider",
}


def stable_document_id(dataset_key: str, article_id: str) -> str:
    """Return a deterministic Pinecone-safe ID unique across datasets."""
    key = re.sub(r"[^a-z0-9_-]+", "-", dataset_key.lower()).strip("-")
    article = re.sub(r"\s+", "-", article_id.strip())
    if not key or not article:
        raise ValueError("dataset_key와 article_id는 비어 있을 수 없습니다.")
    return f"{key}:{article}"


def validate_corpus(documents: list[SourceDocument]) -> None:
    """Validate the shared historical-source contract before external transfer."""
    seen: set[str] = set()
    for document in documents:
        missing = REQUIRED_CORPUS_FIELDS - document.metadata.keys()
        if missing:
            raise ValueError(f"{document.id}: 공통 메타데이터 누락: {sorted(missing)}")
        if not document.text.strip():
            raise ValueError(f"{document.id}: 본문이 비어 있습니다.")
        unverified_sillok_repair = (
            document.metadata["dataset_key"] == "sillok"
            and "𥘼" in document.text
            and document.metadata.get("text_verification_status") != "verified_official_match"
        )
        if unverified_sillok_repair:
            raise ValueError(
                f"{document.id}: 공식 원문 대조 전인 확장한자 복원 문서는 적재할 수 없습니다."
            )
        if document.id in seen:
            raise ValueError(f"중복 문서 ID: {document.id}")
        seen.add(document.id)
        article_id = str(document.metadata["article_id"])
        chunk_index = document.metadata.get("chunk_index")
        id_article = (
            article_id
            if chunk_index is None
            else f"{article_id}__chunk_{int(chunk_index):04d}"
        )
        expected = stable_document_id(str(document.metadata["dataset_key"]), id_article)
        if document.id != expected:
            raise ValueError(f"{document.id}: 공통 ID 규칙과 다릅니다(예상: {expected}).")
