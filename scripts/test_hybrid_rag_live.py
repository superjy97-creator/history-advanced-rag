from __future__ import annotations

import argparse
import json
import sys

from langchain_google_genai import ChatGoogleGenerativeAI

from history_rag.config import Settings
from history_rag.llm import GeminiHistoryLLM, OpenAIHistoryLLM, RoutedHistoryLLM
from history_rag.pipeline import AdvancedRAGPipeline
from history_rag.tool_calling import HybridHistoryRetriever
from history_rag.vectorstore import PineconeHistoryStore


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="하이브리드 Advanced RAG 실제 연결 시험")
    parser.add_argument("question")
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--confirm-external-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_test:
        raise SystemExit("유료·외부 API 시험 승인 후 --confirm-external-test를 지정하세요.")

    settings = Settings.from_env()
    settings.validate_rag()
    store = PineconeHistoryStore(
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        settings.pinecone_namespace,
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )
    retriever = HybridHistoryRetriever(
        ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        ),
        store,
    )
    llm = RoutedHistoryLLM(
        GeminiHistoryLLM(settings.gemini_api_key, settings.gemini_model),
        OpenAIHistoryLLM(settings.openai_api_key, settings.chat_model),
    )
    result = AdvancedRAGPipeline(llm, retriever).run(
        args.question,
        retrieval_k=max(1, min(args.retrieval_k, 20)),
        final_k=max(1, min(args.final_k, 8)),
    )
    print(
        json.dumps(
            {
                "question": args.question,
                "models": {
                    "query_and_rerank": settings.gemini_model,
                    "generation_and_verification": settings.chat_model,
                    "embedding": settings.embedding_model,
                },
                "retrieved": len(result.retrieved),
                "reranked": len(result.reranked),
                "retrieval_paths": sorted(
                    {str(item.metadata.get("retrieval_path", "")) for item in result.retrieved}
                ),
                "source_ids": [item.id for item in result.reranked],
                "source_urls": [item.source_url for item in result.reranked],
                "verification": {
                    "status": result.verification.status,
                    "faithfulness": result.verification.faithfulness,
                    "supported_citations": len(result.verification.supported_citations),
                    "issues": result.verification.issues,
                },
                "answer": result.verification.final_answer,
                "trace": result.trace,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
