from __future__ import annotations

from typing import Protocol

from .models import (
    DraftAnswer,
    PipelineResult,
    SourceDocument,
    TransformedQuery,
    VerificationResult,
)


class HistoryLLM(Protocol):
    def transform_query(self, question: str) -> TransformedQuery: ...
    def rerank(self, question: str, documents: list[SourceDocument]) -> list[SourceDocument]: ...
    def generate(self, question: str, documents: list[SourceDocument]) -> DraftAnswer: ...
    def verify_semantics(
        self, question: str, draft: DraftAnswer, documents: list[SourceDocument]
    ) -> dict: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 12) -> list[SourceDocument]: ...


class AdvancedRAGPipeline:
    """변환 → 검색 → 리랭킹 → 생성 → 검증을 순서대로 실행한다."""

    def __init__(self, llm: HistoryLLM, retriever: Retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def run(self, question: str, retrieval_k: int = 12, final_k: int = 5) -> PipelineResult:
        if not question.strip():
            raise ValueError("질문을 입력해 주세요.")

        transformed = self.llm.transform_query(question.strip())
        retrieved_by_id = {}
        for query in transformed.queries:
            for document in self.retriever.search(query, top_k=retrieval_k):
                existing = retrieved_by_id.get(document.id)
                if existing is None or document.score > existing.score:
                    retrieved_by_id[document.id] = document
        retrieved = list(retrieved_by_id.values())
        rerank_limit = max(final_k, 5)
        rerank_candidates = sorted(retrieved, key=lambda item: item.score, reverse=True)[
            :rerank_limit
        ]
        reranked = self.llm.rerank(question, rerank_candidates)[:final_k]
        if reranked:
            draft = self.llm.generate(question, reranked)
            verification = self._verify(question, draft, reranked)
        else:
            draft = DraftAnswer("제공된 문서에서 확인되지 않는다.", ())
            verification = VerificationResult(
                status="insufficient_evidence",
                final_answer=draft.answer,
                supported_citations=(),
                issues=("리랭킹을 통과한 사료가 없습니다.",),
                faithfulness=0.0,
            )

        trace = {
            "query_transformation": "completed",
            "retrieval": f"completed:{len(retrieved)}",
            "rerank_candidates": len(rerank_candidates),
            "reranking": f"completed:{len(reranked)}",
            "generation": "completed",
            "verification": verification.status,
        }
        retrieval_trace = getattr(self.retriever, "last_trace", None)
        if retrieval_trace:
            trace["retrieval_tools"] = retrieval_trace

        return PipelineResult(
            query=transformed,
            retrieved=tuple(retrieved),
            reranked=tuple(reranked),
            draft=draft,
            verification=verification,
            trace=trace,
        )

    def _verify(
        self, question: str, draft: DraftAnswer, documents: list[SourceDocument]
    ) -> VerificationResult:
        by_id = {d.id: d for d in documents}
        valid = []
        issues = []
        for citation in draft.citations:
            source = by_id.get(citation.source_id)
            if source is None:
                issues.append(f"존재하지 않는 출처 ID: {citation.source_id}")
            elif not citation.quote.strip():
                issues.append(f"빈 인용문: {citation.source_id}")
            elif citation.quote not in source.text:
                issues.append(f"원문과 일치하지 않는 인용문: {citation.source_id}")
            else:
                valid.append(citation)

        semantic = self.llm.verify_semantics(question, draft, documents)
        issues.extend(str(item) for item in semantic.get("removed_reason", []))
        claims = semantic.get("claims", [])
        unsupported = [str(c.get("claim", "")) for c in claims if not c.get("supported")]
        issues.extend(f"근거 없는 주장 제거: {claim}" for claim in unsupported if claim)
        faithfulness = float(semantic.get("faithfulness", 0.0))
        refusal = draft.answer == "제공된 문서에서 확인되지 않는다." and not documents
        supported = (bool(semantic.get("answerable")) and bool(valid) and not issues) or refusal
        if supported:
            status = "passed"
            final_answer = str(semantic.get("revised_answer") or draft.answer)
        else:
            status = "needs_review"
            corrected = str(semantic.get("revised_answer") or draft.answer)
            final_answer = (
                corrected
                + "\n\n※ 일부 주장 또는 인용은 원문 근거가 충분하지 않아 검토가 필요합니다."
            )
        return VerificationResult(status, final_answer, tuple(valid), tuple(issues), faithfulness)
