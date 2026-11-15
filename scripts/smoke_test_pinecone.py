from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from history_rag.config import Settings
from history_rag.vectorstore import PineconeHistoryStore
from ingest_pinecone import (
    file_sha256,
    load_documents,
    load_resume_offset,
    local_preflight,
    write_checkpoint,
)

TEST_NAMESPACE = "korean-history-smoke-v1"


def wait_for_count(store: PineconeHistoryStore, minimum: int) -> int:
    actual = 0
    for _ in range(30):
        actual = store.namespace_count()
        if actual >= minimum:
            return actual
        time.sleep(1)
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description="14종 임베딩·Pinecone 소규모 실동작 시험")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rights-registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--allow-index-create", action="store_true")
    parser.add_argument("--confirm-external-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_test:
        raise SystemExit("외부 소규모 시험 승인 후 --confirm-external-test를 지정하세요.")

    settings = Settings.from_env()
    settings.validate_live()
    documents = load_documents(args.input)
    if len({document.metadata["dataset_key"] for document in documents}) != 14:
        raise ValueError("시험 파일에 14종 사료가 각각 포함되어야 합니다.")
    local_preflight(documents, args.rights_registry)
    fingerprint = file_sha256(args.input)
    checkpoint_base = {
        "input_sha256": fingerprint,
        "namespace": TEST_NAMESPACE,
        "total": len(documents),
    }
    store = PineconeHistoryStore(
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        TEST_NAMESPACE,
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )
    store.ensure_index(allow_create=args.allow_index_create)
    before = store.namespace_count()
    if before not in {0, len(documents)}:
        raise RuntimeError(f"시험 namespace에 예상하지 못한 {before}개 벡터가 있습니다.")

    midpoint = len(documents) // 2
    if not args.checkpoint.exists():
        store.upsert(documents[:midpoint])
        write_checkpoint(
            args.checkpoint,
            {**checkpoint_base, "next_offset": midpoint, "status": "interrupted_test"},
        )
    resume_offset = load_resume_offset(
        args.checkpoint, fingerprint, TEST_NAMESPACE, len(documents)
    )
    store.upsert(documents, start_offset=resume_offset)
    write_checkpoint(
        args.checkpoint,
        {**checkpoint_base, "next_offset": len(documents), "status": "complete"},
    )
    after_upload = wait_for_count(store, len(documents))
    if after_upload != len(documents):
        raise RuntimeError(f"시험 적재 수 불일치: {after_upload} != {len(documents)}")

    for source in documents:
        matches = store.search(
            source.text[:500],
            top_k=3,
            metadata_filter={"dataset_key": {"$eq": source.metadata["dataset_key"]}},
        )
        matched = next((item for item in matches if item.id == source.id), None)
        if matched is None:
            raise RuntimeError(f"검색 결과에서 원문을 찾지 못함: {source.id}")
        if matched.text != source.text or matched.source_url != source.source_url:
            raise RuntimeError(f"검색 결과 원문·출처 불일치: {source.id}")
        if matched.metadata.get("article_id") != source.metadata.get("article_id"):
            raise RuntimeError(f"검색 결과 기사 ID 불일치: {source.id}")

    store.upsert(documents)
    after_reupload = wait_for_count(store, len(documents))
    if after_reupload != after_upload:
        raise RuntimeError("동일 ID 재업로드 후 벡터 수가 증가했습니다.")
    print(
        json.dumps(
            {
                "status": "passed",
                "namespace": TEST_NAMESPACE,
                "documents": len(documents),
                "resumed_from": resume_offset,
                "count_after_upload": after_upload,
                "count_after_reupload": after_reupload,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
