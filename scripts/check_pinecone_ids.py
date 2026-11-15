from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from history_rag.config import Settings
from history_rag.vectorstore import PineconeHistoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinecone 원격 ID와 로컬 JSONL ID 비교")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    local_records = [
        json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()
    ]
    local_by_id = {str(record["id"]): record for record in local_records}
    settings = Settings.from_env()
    store = PineconeHistoryStore(
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        settings.pinecone_namespace,
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )
    remote_ids = [
        str(item)
        for page in store.pc.Index(settings.pinecone_index_name).list(
            namespace=settings.pinecone_namespace
        )
        for item in page
    ]
    local_ids = set(local_by_id)
    remote_id_set = set(remote_ids)
    missing = sorted(local_ids - remote_id_set)
    unexpected = sorted(remote_id_set - local_ids)
    remote_by_dataset = Counter(
        str(local_by_id[item]["metadata"].get("dataset_key", "unknown"))
        for item in remote_id_set & local_ids
    )
    print(
        json.dumps(
            {
                "local": len(local_ids),
                "remote_listed": len(remote_ids),
                "remote_unique": len(remote_id_set),
                "missing_count": len(missing),
                "missing_ids": missing,
                "unexpected_count": len(unexpected),
                "unexpected_ids": unexpected,
                "remote_by_dataset": dict(sorted(remote_by_dataset.items())),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
