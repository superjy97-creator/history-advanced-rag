from __future__ import annotations

import argparse
import json
import sys

from history_rag.config import Settings
from history_rag.llm import OpenAIHistoryLLM
from history_rag.models import SourceDocument


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="GPT 생성·근거 검증 최소 연결 테스트")
    parser.add_argument("--confirm-external-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_test:
        raise SystemExit("외부 OpenAI 호출 승인 시 --confirm-external-test를 지정하세요.")

    settings = Settings.from_env()
    if not settings.openai_api_key or not settings.chat_model:
        raise RuntimeError("OPENAI_API_KEY와 OPENAI_CHAT_MODEL이 필요합니다.")

    source = SourceDocument(
        id="openai-connection-test:1",
        text="세종은 백성이 말하고자 하는 바를 쉽게 표현하도록 훈민정음을 창제하였다.",
        title="GPT 연결 확인용 사료",
        source_url="https://example.invalid/openai-connection-test",
        metadata={"dataset_key": "connection_test"},
    )
    question = "세종이 훈민정음을 만든 목적은 무엇인가?"
    llm = OpenAIHistoryLLM(settings.openai_api_key, settings.chat_model)
    draft = llm.generate(question, [source])
    verification = llm.verify_semantics(question, draft, [source])
    citation_compatible = all(
        citation.source_id == source.id and citation.quote in source.text
        for citation in draft.citations
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "model": settings.chat_model,
                "generation": {
                    "answer": draft.answer,
                    "citations": [
                        {"source_id": item.source_id, "quote": item.quote}
                        for item in draft.citations
                    ],
                    "data_model_compatible": isinstance(draft.answer, str)
                    and citation_compatible,
                },
                "verification": {
                    "answerable": verification.get("answerable"),
                    "faithfulness": verification.get("faithfulness"),
                    "claims": verification.get("claims"),
                    "revised_answer": verification.get("revised_answer"),
                    "json_compatible": isinstance(verification, dict),
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
