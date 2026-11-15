from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .models import SourceDocument
from .official_search import OfficialHistoryClient, OfficialSearchTrace


@dataclass(frozen=True)
class ToolExecution:
    requested: str
    executed: bool
    result_count: int
    error: str = ""


def _document_payload(document: SourceDocument) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "text": document.text[:1600],
        "source_url": document.source_url,
        "source_url_type": document.metadata.get("source_url_type", "official_record"),
        "dataset_key": document.metadata.get("dataset_key", ""),
        "retrieval_path": document.metadata.get("retrieval_path", ""),
        "record_date": document.metadata.get("record_date", ""),
        "event_title": document.metadata.get("event_title", document.title),
        "score": document.score,
    }


class HybridHistoryRetriever:
    """Gemini의 도구 요청을 Python에서 실행하고 SourceDocument로 합친다."""

    def __init__(
        self,
        tool_model: Any,
        pinecone_store: Any,
        official_client: OfficialHistoryClient | None = None,
        max_tool_calls: int = 2,
    ) -> None:
        self.pinecone_store = pinecone_store
        self.official_client = official_client or OfficialHistoryClient()
        self.max_tool_calls = max(1, min(max_tool_calls, 2))
        self.last_trace: dict[str, Any] = {}
        self._results: dict[str, list[SourceDocument]] = {}
        tools = [
            StructuredTool.from_function(
                self._search_pinecone,
                name="search_pinecone_history",
                description=(
                    "Search the complete 14-source Korean-history Pinecone corpus. "
                    "Use for every Korean-history evidence search. "
                    "Do not restrict era or dynasty."
                ),
            ),
            StructuredTool.from_function(
                self._search_official,
                name="search_official_history",
                description=(
                    "Search the official National Institute of Korean History "
                    "Joseon Dynasty Annals site. It returns live corroboration, "
                    "official record dates, stable URLs, Korean translations and original text. "
                    "Use a concise event or proper-name query when Pinecone evidence does not "
                    "directly establish every year, month, day, or period requested."
                ),
            ),
        ]
        self._bound_model = tool_model.bind_tools(tools)

    def _search_pinecone(self, query: str, top_k: int = 12) -> list[dict[str, Any]]:
        """Search the local project's Pinecone Korean-history namespace."""
        documents = self.pinecone_store.search(query, top_k=max(1, min(int(top_k), 20)))
        documents = [
            SourceDocument(
                d.id,
                d.text,
                d.title,
                d.source_url,
                {**d.metadata, "retrieval_path": "pinecone"},
                d.score,
            )
            for d in documents
        ]
        self._results["search_pinecone_history"] = documents
        return [_document_payload(document) for document in documents]

    def _search_official(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search official NIKH Sillok pages; never persists live results to Pinecone."""
        sillok_markers = (
            "조선",
            "실록",
            "태조",
            "세종",
            "훈민정음",
            "연산군",
            "선조",
            "광해군",
            "인조",
            "숙종",
            "영조",
            "정조",
            "고종",
            "임진왜란",
            "병자호란",
            "경국대전",
        )
        if not any(marker in query for marker in sillok_markers):
            self.official_client.last_trace = OfficialSearchTrace(
                query, 0, 0, "skipped:not_sillok_scope"
            )
            self._results["search_official_history"] = []
            return []
        documents = self.official_client.search(query, top_k=max(1, min(int(top_k), 3)))
        self._results["search_official_history"] = documents
        return [_document_payload(document) for document in documents]

    def search(self, query: str, top_k: int = 12) -> list[SourceDocument]:
        clean_query = " ".join(query.split())
        if not clean_query or len(clean_query) > 500:
            raise ValueError("검색어는 1~500자여야 합니다.")
        self._results = {}
        messages: list[Any] = [
            SystemMessage(
                content=(
                    "You route evidence searches for a Korean-history RAG. First request "
                    "search_pinecone_history. Inspect its returned evidence. Decide whether the "
                    "question requires an exact year, month, day, sequence, or period. If every "
                    "requested temporal fact is already explicit, do not search again. If such "
                    "facts are missing and the Joseon Annals can directly corroborate them, then "
                    "request search_official_history with a concise event or proper-name query. "
                    "Never invent a date. Request at most two tools in total and do not answer "
                    "the history question yourself."
                )
            ),
            HumanMessage(content=clean_query),
        ]
        executions: list[ToolExecution] = []
        requested_tools: list[str] = []
        executed_tools: set[str] = set()
        conversation = list(messages)
        response = self._bound_model.invoke(conversation)
        model_invocations = 1
        returned_to_model = False

        while len(executions) < self.max_tool_calls:
            remaining = self.max_tool_calls - len(executions)
            tool_calls = list(getattr(response, "tool_calls", []) or [])[:remaining]
            if not tool_calls:
                break

            tool_messages: list[ToolMessage] = []
            for call in tool_calls:
                name = str(call.get("name", ""))
                args = dict(call.get("args") or {})
                call_id = str(call.get("id") or name)
                requested_tools.append(name)
                if name in executed_tools:
                    executions.append(ToolExecution(name, False, 0, "skipped:duplicate"))
                    payload: Any = {"error": "duplicate tool call skipped"}
                else:
                    try:
                        if name == "search_pinecone_history":
                            payload = self._search_pinecone(**args)
                        elif name == "search_official_history":
                            payload = self._search_official(**args)
                        else:
                            raise ValueError(f"허용되지 않은 도구: {name}")
                        executed_tools.add(name)
                        executions.append(ToolExecution(name, True, len(payload)))
                    except Exception as exc:
                        executions.append(
                            ToolExecution(name, False, 0, f"{type(exc).__name__}: {exc}")
                        )
                        payload = {"error": "tool execution failed"}
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=call_id,
                    )
                )

            conversation.extend([response, *tool_messages])
            response = self._bound_model.invoke(conversation)
            model_invocations += 1
            returned_to_model = True

        if "search_pinecone_history" not in self._results:
            try:
                payload = self._search_pinecone(clean_query, top_k)
                executions.append(
                    ToolExecution("search_pinecone_history:fallback", True, len(payload))
                )
            except Exception as exc:
                executions.append(
                    ToolExecution(
                        "search_pinecone_history:fallback",
                        False,
                        0,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        merged: dict[str, SourceDocument] = {}
        for documents in self._results.values():
            for document in documents:
                key = str(document.metadata.get("article_id") or document.id)
                existing = merged.get(key)
                if existing is None or document.score > existing.score:
                    merged[key] = document
        self.last_trace = {
            "requested_tools": requested_tools,
            "executions": [asdict(item) for item in executions],
            "tool_results_returned_to_gemini": returned_to_model,
            "tool_model_invocations": model_invocations,
            "official_search": asdict(self.official_client.last_trace),
        }
        return list(merged.values())
