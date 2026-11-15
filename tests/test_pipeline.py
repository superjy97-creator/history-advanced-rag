from __future__ import annotations

import unittest

from history_rag.llm import (
    OpenAIHistoryLLM,
    RoutedHistoryLLM,
    _evidence_is_verbatim,
    _message_text,
)
from history_rag.models import Citation, DraftAnswer, SourceDocument, TransformedQuery
from history_rag.pipeline import AdvancedRAGPipeline

DOC = SourceDocument(
    id="sg_001_0020_0010",
    text="始祖姓朴氏, 諱赫居世. 四月丙辰即位.",
    title="시조 혁거세가 즉위하다",
    source_url="https://www.data.go.kr/data/15053635/fileData.do",
)


class FakeRetriever:
    def __init__(self) -> None:
        self.query = ""

    def search(self, query: str, top_k: int = 12):
        self.query = query
        return [DOC]


class FakeLLM:
    def __init__(self, bad_quote: bool = False) -> None:
        self.calls = []
        self.bad_quote = bad_quote

    def transform_query(self, question):
        self.calls.append("transform")
        return TransformedQuery(
            question,
            "혁거세 즉위 원문",
            ("혁거세 즉위 원문",),
            "신라",
            ("혁거세",),
            (),
            ("즉위",),
            "사실 확인",
        )

    def rerank(self, question, documents):
        self.calls.append("rerank")
        return documents

    def generate(self, question, documents):
        self.calls.append("generate")
        quote = "없는 문장" if self.bad_quote else "四月丙辰即位"
        return DraftAnswer("혁거세는 4월 병진일에 즉위했다.", (Citation(DOC.id, quote),))

    def verify_semantics(self, question, draft, documents):
        self.calls.append("verify")
        return {
            "claims": [{"claim": draft.answer, "supported": True}],
            "faithfulness": 1.0,
            "answerable": True,
            "revised_answer": draft.answer,
            "removed_reason": [],
        }


class NoEvidenceLLM(FakeLLM):
    def rerank(self, question, documents):
        self.calls.append("rerank")
        return []


class PipelineTests(unittest.TestCase):
    def test_rerank_evidence_allows_only_exact_segments_around_ellipsis(self):
        text = "盖已往之興亡, 實將來之勸戒. 중간 문장. 揭明鑑於後人, 期不沒善惡之實."
        self.assertTrue(
            _evidence_is_verbatim(
                "盖已往之興亡, 實將來之勸戒...揭明鑑於後人, 期不沒善惡之實.", text
            )
        )
        self.assertFalse(_evidence_is_verbatim("실제 없는 구절...揭明鑑於後人", text))

    def test_query_transform_keeps_original_and_verified_hanja_alias(self):
        llm = OpenAIHistoryLLM.__new__(OpenAIHistoryLLM)
        llm._json = lambda system, user: {"queries": ["혁거세 즉위 나이"]}
        question = "삼국사기 원문에서 혁거세가 즉위할 때 나이는?"
        transformed = llm.transform_query(question)
        self.assertEqual(transformed.queries[0], question)
        self.assertIn("時年十三", transformed.queries[1])

    def test_reranker_accepts_top_level_json_array(self):
        llm = OpenAIHistoryLLM.__new__(OpenAIHistoryLLM)
        llm._json = lambda system, user: [
            {"id": DOC.id, "verdict": "yes", "evidence": DOC.text}
        ]
        self.assertEqual(llm.rerank("질문", [DOC])[0].id, DOC.id)

    def test_gemini_content_blocks_are_joined_for_json_parsing(self):
        content = [{"type": "text", "text": '{"queries":["세종 훈민정음"]}'}]
        self.assertEqual(_message_text(content), '{"queries":["세종 훈민정음"]}')

    def test_model_router_assigns_gemini_then_gpt(self):
        gemini = FakeLLM()
        gpt = FakeLLM()
        router = RoutedHistoryLLM(gemini, gpt)
        router.transform_query("질문")
        documents = router.rerank("질문", [DOC])
        draft = router.generate("질문", documents)
        router.verify_semantics("질문", draft, documents)
        self.assertEqual(gemini.calls, ["transform", "rerank"])
        self.assertEqual(gpt.calls, ["generate", "verify"])

    def test_four_stages_are_connected_in_order(self):
        llm = FakeLLM()
        retriever = FakeRetriever()
        result = AdvancedRAGPipeline(llm, retriever).run("혁거세는 언제 즉위했나?")
        self.assertEqual(llm.calls, ["transform", "rerank", "generate", "verify"])
        self.assertEqual(retriever.query, "혁거세 즉위 원문")
        self.assertEqual(result.verification.status, "passed")
        self.assertEqual(result.trace["query_transformation"], "completed")

    def test_verification_rejects_quote_not_found_in_source(self):
        result = AdvancedRAGPipeline(FakeLLM(bad_quote=True), FakeRetriever()).run("질문")
        self.assertEqual(result.verification.status, "needs_review")
        self.assertIn("원문과 일치하지 않는 인용문", result.verification.issues[0])
        self.assertFalse(result.verification.supported_citations)

    def test_zero_rerank_results_use_refusal_without_generation(self):
        llm = NoEvidenceLLM()
        result = AdvancedRAGPipeline(llm, FakeRetriever()).run("근거 없는 질문")
        self.assertEqual(llm.calls, ["transform", "rerank"])
        self.assertEqual(result.verification.status, "insufficient_evidence")
        self.assertEqual(result.verification.final_answer, "제공된 문서에서 확인되지 않는다.")


if __name__ == "__main__":
    unittest.main()
