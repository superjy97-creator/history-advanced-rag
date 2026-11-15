from __future__ import annotations

import argparse
import json

from history_rag.config import Settings
from history_rag.vectorstore import PineconeHistoryStore

INDEX_NAME = "history-advanced-rag"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="한국사 RAG용 일반 Dense Pinecone 인덱스를 확인하거나 생성합니다."
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="인덱스가 없을 때만 새로 생성합니다. 생략하면 조회만 수행합니다.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY가 설정되지 않았습니다.")

    store = PineconeHistoryStore(
        pinecone_api_key=settings.pinecone_api_key,
        index_name=INDEX_NAME,
        namespace="korean-history-v1",
        openai_api_key=settings.openai_api_key or "not-used-for-index-management",
        embedding_model="text-embedding-3-large",
        dimension=3072,
    )
    existed_before = store.pc.has_index(INDEX_NAME)
    if not existed_before and not args.create:
        print(json.dumps({"exists": False, "created": False, "name": INDEX_NAME}))
        return

    # 수업 노트북 Part 4의 Pinecone + ServerlessSpec + create_index 흐름은
    # PineconeHistoryStore.ensure_index()에 구현되어 있으며, 여기서는 이를 재사용한다.
    # --create가 없으면 allow_create=False이므로 기존 인덱스의 설정만 검증한다.
    store.ensure_index(allow_create=args.create)
    details = store.pc.describe_index(INDEX_NAME).to_dict()
    stats = store.pc.Index(INDEX_NAME).describe_index_stats().to_dict()
    serverless = details.get("spec", {}).get("serverless", {})
    status = details.get("status", {})
    print(
        json.dumps(
            {
                "exists": True,
                "created": not existed_before,
                "name": details.get("name", INDEX_NAME),
                "vector_type": details.get("vector_type", "dense"),
                "dimension": details.get("dimension"),
                "metric": details.get("metric"),
                "cloud": serverless.get("cloud"),
                "region": serverless.get("region"),
                "ready": status.get("ready"),
                "state": status.get("state"),
                "total_vector_count": stats.get("total_vector_count", 0),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
