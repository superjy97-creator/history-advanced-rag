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
<h3>훈민정음을 창제하다</h3>
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


if __name__ == "__main__":
    unittest.main()
