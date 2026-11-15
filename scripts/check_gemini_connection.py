from __future__ import annotations

import argparse
import json
import sys

from history_rag.config import Settings
from history_rag.llm import GeminiHistoryLLM
from history_rag.models import SourceDocument


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Gemini 단계 1·2 최소 연결 테스트")
    parser.add_argument("--confirm-external-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_test:
        raise SystemExit("외부 Gemini 호출 승인 시 --confirm-external-test를 지정하세요.")

    settings = Settings.from_env()
    if not settings.gemini_api_key or not settings.gemini_model:
        raise RuntimeError("GEMINI_API_KEY와 GEMINI_MODEL이 필요합니다.")

    gemini = GeminiHistoryLLM(settings.gemini_api_key, settings.gemini_model)
    question = "세종은 훈민정음을 왜 만들었는가?"
    transformed = gemini.transform_query(question)
    source = SourceDocument(
        id="connection-test:1",
        text="세종은 백성이 말하고자 하는 바를 쉽게 표현하도록 훈민정음을 창제하였다.",
        title="Gemini 연결 확인용 문장",
        source_url="https://example.invalid/connection-test",
        metadata={"dataset_key": "connection_test"},
    )
    reranked = gemini.rerank(question, [source])
    print(
        json.dumps(
            {
                "status": "passed",
                "model": settings.gemini_model,
                "query_transformation": {
                    "queries": transformed.queries,
                    "era": transformed.era,
                    "people": transformed.people,
                    "events": transformed.events,
                    "keywords": transformed.keywords,
                    "intent": transformed.intent,
                },
                "reranking": {
                    "input_documents": 1,
                    "relevant_documents": len(reranked),
                    "evidence": (
                        reranked[0].metadata.get("rerank_evidence") if reranked else None
                    ),
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
