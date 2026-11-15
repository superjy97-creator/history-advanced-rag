from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken

from history_rag.config import PROJECT_ROOT, Settings
from history_rag.corpus import validate_corpus
from history_rag.models import SourceDocument
from history_rag.rights import validate_external_use, validate_rights_registry
from history_rag.vectorstore import PineconeHistoryStore


def load_documents(path: Path) -> list[SourceDocument]:
    documents = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                documents.append(SourceDocument(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: 잘못된 JSONL 문서") from exc
    return documents


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_resume_offset(
    checkpoint_path: Path, fingerprint: str, namespace: str, total: int
) -> int:
    if not checkpoint_path.exists():
        return 0
    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected = {"input_sha256": fingerprint, "namespace": namespace, "total": total}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"체크포인트 {key}가 현재 입력과 다릅니다.")
    return int(state.get("next_offset", 0))


def local_preflight(documents: list[SourceDocument], registry_path: Path) -> dict[str, Any]:
    validate_corpus(documents)
    validate_external_use(documents)
    validate_rights_registry(documents, registry_path)
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    largest_metadata = 0
    total_metadata_bytes = 0
    dataset_chunks: Counter[str] = Counter()
    source_articles: set[tuple[str, str]] = set()
    for document in documents:
        total_tokens += len(encoding.encode(document.text, disallowed_special=()))
        metadata_bytes = PineconeHistoryStore.validate_metadata(document)
        largest_metadata = max(largest_metadata, metadata_bytes)
        total_metadata_bytes += metadata_bytes
        dataset_key = str(document.metadata.get("dataset_key", "unknown"))
        dataset_chunks[dataset_key] += 1
        article_id = str(document.metadata.get("article_id", document.id))
        source_articles.add((dataset_key, article_id))
    vector_bytes = len(documents) * 3072 * 4
    return {
        "documents": len(documents),
        "source_articles": len(source_articles),
        "datasets": len(dataset_chunks),
        "dataset_chunks": dict(sorted(dataset_chunks.items())),
        "tokens": total_tokens,
        "estimated_embedding_usd": round(total_tokens * 0.13 / 1_000_000, 6),
        "estimated_vector_bytes_float32": vector_bytes,
        "metadata_bytes": total_metadata_bytes,
        "estimated_payload_bytes": vector_bytes + total_metadata_bytes,
        "largest_metadata_bytes": largest_metadata,
        "metadata_limit_bytes": 40_960,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="승인된 역사 사료를 Pinecone에 안전하게 적재")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--rights-registry",
        type=Path,
        default=PROJECT_ROOT / "reports" / "rights_registry_20261114_final.json",
    )
    parser.add_argument("--namespace")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-log", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-index-create", action="store_true")
    parser.add_argument("--confirm-external-upload", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    namespace = args.namespace or settings.pinecone_namespace
    documents = load_documents(args.input)
    preflight = local_preflight(documents, args.rights_registry)
    fingerprint = file_sha256(args.input)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    **preflight,
                    "vectorstore": "langchain_pinecone.PineconeVectorStore",
                    "pinecone_sdk_required": ">=7.3,<8",
                    "langchain_pinecone_required": "0.2.13",
                    "embedding_model": settings.embedding_model,
                    "dimension": settings.embedding_dimension,
                    "index_name_configured": bool(settings.pinecone_index_name),
                    "openai_key_loaded": bool(settings.openai_api_key),
                    "pinecone_key_loaded": bool(settings.pinecone_api_key),
                    "namespace": namespace,
                    "input_sha256": fingerprint,
                    "external_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.confirm_external_upload:
        raise SystemExit("외부 전송 승인 후 --confirm-external-upload를 지정하세요.")
    settings.validate_live()

    checkpoint = args.checkpoint or args.input.with_suffix(args.input.suffix + ".checkpoint.json")
    batch_log = args.batch_log or checkpoint.with_suffix(checkpoint.suffix + ".batches.jsonl")
    start_offset = (
        load_resume_offset(checkpoint, fingerprint, namespace, len(documents))
        if args.resume
        else 0
    )
    checkpoint_base = {
        "input": str(args.input),
        "input_sha256": fingerprint,
        "namespace": namespace,
        "total": len(documents),
        "embedding_model": settings.embedding_model,
        "dimension": settings.embedding_dimension,
    }
    uploaded_by_dataset: Counter[str] = Counter(
        str(document.metadata.get("dataset_key", "unknown"))
        for document in documents[:start_offset]
    )
    previous_offset = start_offset

    store = PineconeHistoryStore(
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        namespace,
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )

    def save_progress(next_offset: int) -> None:
        nonlocal previous_offset
        batch_counts = Counter(
            str(document.metadata.get("dataset_key", "unknown"))
            for document in documents[previous_offset:next_offset]
        )
        uploaded_by_dataset.update(batch_counts)
        write_checkpoint(
            checkpoint,
            {
                **checkpoint_base,
                "next_offset": next_offset,
                "status": "in_progress",
                "uploaded_by_dataset": dict(sorted(uploaded_by_dataset.items())),
            },
        )
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        with batch_log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "completed_at": datetime.now(UTC).isoformat(),
                        "start_offset": previous_offset,
                        "next_offset": next_offset,
                        "batch_documents": next_offset - previous_offset,
                        "batch_by_dataset": dict(sorted(batch_counts.items())),
                        "uploaded_by_dataset": dict(sorted(uploaded_by_dataset.items())),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        previous_offset = next_offset

    try:
        upserted = store.upsert(
            documents,
            batch_size=args.batch_size,
            start_offset=start_offset,
            allow_index_create=args.allow_index_create,
            progress=save_progress,
        )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        write_checkpoint(
            checkpoint,
            {
                **checkpoint_base,
                "next_offset": previous_offset,
                "status": "interrupted",
                "uploaded_by_dataset": dict(sorted(uploaded_by_dataset.items())),
                "batch_log": str(batch_log),
                "last_error": error,
            },
        )
        with batch_log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "failed_at": datetime.now(UTC).isoformat(),
                        "next_offset": previous_offset,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        raise
    actual = 0
    for _ in range(30):
        actual = store.namespace_count()
        if actual >= len(documents):
            break
        time.sleep(1)
    if actual < len(documents):
        raise RuntimeError(f"적재 검증 실패: 예상 {len(documents)}건 이상, 확인 {actual}건")
    write_checkpoint(
        checkpoint,
        {
            **checkpoint_base,
            "next_offset": len(documents),
            "status": "verifying",
            "uploaded_by_dataset": dict(sorted(uploaded_by_dataset.items())),
            "batch_log": str(batch_log),
        },
    )
    verification = store.verify_remote_corpus(documents)
    write_checkpoint(
        checkpoint,
        {
            **checkpoint_base,
            "next_offset": len(documents),
            "status": "complete",
            "uploaded_by_dataset": dict(sorted(uploaded_by_dataset.items())),
            "batch_log": str(batch_log),
            "remote_verification": verification,
        },
    )
    print(
        f"upserted={upserted} resumed_from={start_offset} "
        f"verified={actual} namespace={namespace} "
        f"sample_records={verification['sample_records_verified']}"
    )


if __name__ == "__main__":
    main()
