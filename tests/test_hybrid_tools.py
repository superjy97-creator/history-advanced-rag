from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from history_rag.models import SourceDocument
from history_rag.official_search import OfficialHistoryClient
from history_rag.tool_calling import HybridHistoryRetriever

SEARCH_HTML = """
<a href="javascript:goView('kda_12512030_002', 0);" class="subject">
1. 세종실록 102권 / <span>훈민정음</span>을 창제하다</a>
"""
DETAIL_HTML = """
<div class="title-head"><div class="title">
<p class="date">세종실록102권, 세종 25년 12월 30일 / 1443년</p>
<h3>훈민정음을 창제하다</h3>
</div></div>
<h4 class="view-title">국역</h4><div class="view-text"><p>임금이 언문 28자를 지었다.</p></div>
<h4 class="view-title">원문</h4><div class="view-text"><p>上親制諺文二十八字。</p></div>
"""


class FakeToolModel:
    def __init__(self) -> None:
        self.invocations = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_pinecone_history",
                        "args": {"query": "훈민정음", "top_k": 2},
                        "id": "call-pinecone",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_official_history",
                        "args": {"query": "훈민정음", "top_k": 1},
                        "id": "call-official",
                        "type": "tool_call",
                    },
                ],
            )
        return AIMessage(content="도구 결과를 확인했습니다.")


class SequentialToolModel:
    def __init__(self, request_official: bool = True) -> None:
        self.invocations = 0
        self.request_official = request_official

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_pinecone_history",
                        "args": {"query": "사건 시기 확인", "top_k": 2},
                        "id": "call-pinecone-first",
                        "type": "tool_call",
                    }
                ],
            )
        if self.invocations == 2 and self.request_official:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_official_history",
                        "args": {"query": "훈민정음", "top_k": 2},
                        "id": "call-official-after-review",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="근거 검색을 마쳤습니다.")


class FakeStore:
    def search(self, query, top_k=12):
        return [
            SourceDocument(
                "local-1",
                "훈민정음 관련 사료",
                "로컬 사료",
                "https://example.test/local-1",
                {"dataset_key": "local"},
                0.9,
            )
        ]


def fake_fetch(request, timeout):
    return SEARCH_HTML if "searchResultList" in request.full_url else DETAIL_HTML


class HybridToolsTests(unittest.TestCase):
    def test_official_search_keeps_article_id_original_and_url(self):
        client = OfficialHistoryClient(fetcher=fake_fetch)
        results = client.search("훈민정음", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["article_id"], "kda_12512030_002")
        self.assertIn("上親制諺文二十八字", results[0].text)
        self.assertIn("[공식 기사 일자]", results[0].text)
        self.assertEqual(results[0].metadata["western_years"], ("1443",))
        self.assertIn("세종 25년 12월 30일", results[0].metadata["record_date"])
        self.assertEqual(results[0].source_url, "https://sillok.history.go.kr/id/kda_12512030_002")

    def test_external_failure_returns_empty_results(self):
        def fail(request, timeout):
            raise TimeoutError("offline")

        client = OfficialHistoryClient(fetcher=fail)
        self.assertEqual(client.search("훈민정음"), [])
        self.assertIn("TimeoutError", client.last_trace.error)

    def test_gemini_requests_python_executes_and_results_return(self):
        client = OfficialHistoryClient(fetcher=fake_fetch)
        model = FakeToolModel()
        retriever = HybridHistoryRetriever(model, FakeStore(), client)
        results = retriever.search("훈민정음은 언제 창제되었나?", top_k=5)
        self.assertEqual(
            {item.metadata["retrieval_path"] for item in results}, {"pinecone", "official_live"}
        )
        self.assertEqual(model.invocations, 2)
        self.assertTrue(retriever.last_trace["tool_results_returned_to_gemini"])
        self.assertTrue(all(item["executed"] for item in retriever.last_trace["executions"]))

    def test_gemini_can_request_official_after_reviewing_pinecone(self):
        model = SequentialToolModel(request_official=True)
        retriever = HybridHistoryRetriever(
            model,
            FakeStore(),
            OfficialHistoryClient(fetcher=fake_fetch),
        )

        results = retriever.search("사건의 연도와 월일은 언제인가?", top_k=5)

        self.assertEqual(
            retriever.last_trace["requested_tools"],
            ["search_pinecone_history", "search_official_history"],
        )
        self.assertEqual(model.invocations, 3)
        self.assertEqual(
            {item.metadata["retrieval_path"] for item in results},
            {"pinecone", "official_live"},
        )
        official = next(
            item for item in results if item.metadata["retrieval_path"] == "official_live"
        )
        self.assertEqual(official.metadata["western_years"], ("1443",))

    def test_sufficient_pinecone_evidence_avoids_official_search(self):
        def must_not_fetch(request, timeout):
            raise AssertionError("공식 검색을 불필요하게 실행했습니다.")

        model = SequentialToolModel(request_official=False)
        client = OfficialHistoryClient(fetcher=must_not_fetch)
        retriever = HybridHistoryRetriever(model, FakeStore(), client)

        results = retriever.search("이미 근거가 충분한 사건 시기", top_k=5)

        self.assertEqual(model.invocations, 2)
        self.assertEqual(retriever.last_trace["requested_tools"], ["search_pinecone_history"])
        self.assertEqual(client.last_trace.result_count, 0)
        self.assertEqual({item.metadata["retrieval_path"] for item in results}, {"pinecone"})

    def test_official_failure_keeps_pinecone_results_and_error_trace(self):
        def fail(request, timeout):
            raise TimeoutError("offline")

        model = SequentialToolModel(request_official=True)
        client = OfficialHistoryClient(fetcher=fail)
        retriever = HybridHistoryRetriever(model, FakeStore(), client)

        results = retriever.search("조선 사건의 정확한 연도는?", top_k=5)

        self.assertEqual({item.metadata["retrieval_path"] for item in results}, {"pinecone"})
        self.assertIn("TimeoutError", retriever.last_trace["official_search"]["error"])

    def test_conflicting_official_dates_are_preserved_for_later_verification(self):
        class ConflictingOfficialClient(OfficialHistoryClient):
            def search(self, query: str, top_k: int = 3):
                return [
                    SourceDocument(
                        "official-a",
                        "[공식 기사 일자] 왕 1년 / 1500년\n사건 기록",
                        "첫 번째 기록",
                        "https://sillok.history.go.kr/id/official-a",
                        {
                            "article_id": "official-a",
                            "retrieval_path": "official_live",
                            "record_date": "왕 1년 / 1500년",
                            "western_years": ("1500",),
                        },
                        1.0,
                    ),
                    SourceDocument(
                        "official-b",
                        "[공식 기사 일자] 왕 2년 / 1501년\n사건 기록",
                        "두 번째 기록",
                        "https://sillok.history.go.kr/id/official-b",
                        {
                            "article_id": "official-b",
                            "retrieval_path": "official_live",
                            "record_date": "왕 2년 / 1501년",
                            "western_years": ("1501",),
                        },
                        1.0,
                    ),
                ][:top_k]

        retriever = HybridHistoryRetriever(
            SequentialToolModel(request_official=True),
            FakeStore(),
            ConflictingOfficialClient(),
        )

        results = retriever.search("조선 사건의 연도는?", top_k=5)
        dates = {
            item.metadata["record_date"]
            for item in results
            if item.metadata.get("retrieval_path") == "official_live"
        }

        self.assertEqual(dates, {"왕 1년 / 1500년", "왕 2년 / 1501년"})


if __name__ == "__main__":
    unittest.main()
