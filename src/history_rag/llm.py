from __future__ import annotations

import importlib
import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .models import Citation, DraftAnswer, SourceDocument, TransformedQuery

SEARCH_ALIASES = {
    "혁거세": "始祖姓朴氏 諱赫居世 即位 號居西干 時年十三",
    "고려사 찬진표": "進高麗史箋 已往之興亡 將來之勸戒",
}


def _json_object(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def _evidence_is_verbatim(evidence: str, text: str) -> bool:
    if evidence in text:
        return True
    parts = [part.strip() for part in re.split(r"(?:\.{3}|…+)", evidence) if part.strip()]
    return len(parts) > 1 and all(part in text for part in parts)


class GeminiHistoryLLM:
    """Run the existing stage prompts through Google's Gemini integration."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key or not model:
            raise ValueError("GEMINI_API_KEY and GEMINI_MODEL must be explicitly configured.")
        try:
            module = importlib.import_module("langchain_google_genai")
        except ModuleNotFoundError as exc:
            raise RuntimeError("langchain-google-genai==4.5.0 is required.") from exc
        delegate = OpenAIHistoryLLM.__new__(OpenAIHistoryLLM)
        delegate.client = module.ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
        ).bind(response_mime_type="application/json")
        self.delegate = delegate

    def transform_query(self, question: str) -> TransformedQuery:
        return self.delegate.transform_query(question)

    def rerank(self, question: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        return self.delegate.rerank(question, documents)


class RoutedHistoryLLM:
    """Route stages 1-2 to Gemini and stages 3-4 to GPT."""

    def __init__(self, gemini: Any, gpt: Any) -> None:
        self.gemini = gemini
        self.gpt = gpt

    def transform_query(self, question: str) -> TransformedQuery:
        return self.gemini.transform_query(question)

    def rerank(self, question: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        return self.gemini.rerank(question, documents)

    def generate(self, question: str, documents: list[SourceDocument]) -> DraftAnswer:
        return self.gpt.generate(question, documents)

    def verify_semantics(
        self, question: str, draft: DraftAnswer, documents: list[SourceDocument]
    ) -> dict[str, Any]:
        return self.gpt.verify_semantics(question, draft, documents)


class OpenAIHistoryLLM:
    """네 단계 가운데 변환, 리랭킹, 생성, 의미 검증을 담당한다."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = ChatOpenAI(api_key=SecretStr(api_key), model=model).bind(
            response_format={"type": "json_object"}
        )

    def _json(self, system: str, user: str) -> Any:
        response = self.client.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return _json_object(_message_text(response.content) or "{}")

    def transform_query(self, question: str) -> TransformedQuery:
        data = self._json(
            "한국사 사료 검색 질의 재작성기다. JSON 외에는 출력하지 않는다.",
            "질문을 탈맥락화하고, 구어를 사료의 문어 표현으로 확장하라. 고유명사·문헌명·"
            "날짜·번호는 그대로 보존하고 원문에 없는 조건은 추가하지 않는다. 서로 다른 "
            "사실을 요구할 때만 최대 3개로 분해한다. 키: queries(JSON 배열, 1~3개), "
            "era(null 가능), people, events, keywords, intent. "
            f"질문: {question}",
        )
        candidates = [question]
        candidates.extend(alias for term, alias in SEARCH_ALIASES.items() if term in question)
        candidates.extend(str(q) for q in data.get("queries") or [])
        queries = tuple(dict.fromkeys(candidates))[:3]
        return TransformedQuery(
            original=question,
            search_query=" ".join(queries),
            queries=queries,
            era=data.get("era"),
            people=tuple(data.get("people") or []),
            events=tuple(data.get("events") or []),
            keywords=tuple(data.get("keywords") or []),
            intent=str(data.get("intent") or "역사 사료 기반 설명"),
        )

    def rerank(self, question: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        payload = [{"id": d.id, "title": d.title, "text": d.text[:1800]} for d in documents]
        data = self._json(
            "한국사 사료 관련성 판정기다. JSON 외에는 출력하지 않는다. 질문의 답 또는 "
            "답의 일부가 명시된 문서만 yes다. 주제나 용어만 유사하거나 애매하면 no다.",
            f"질문: {question}\n후보: {json.dumps(payload, ensure_ascii=False)}\n"
            "키: results. 값: [{id, verdict(yes/no), evidence(원문 그대로)}]",
        )
        result_items = data if isinstance(data, list) else data.get("results", [])
        verdicts = {
            str(item.get("id")): item for item in result_items if isinstance(item, dict)
        }
        relevant = []
        for d in documents:
            item = verdicts.get(d.id, {})
            evidence = str(item.get("evidence", ""))
            if item.get("verdict") == "yes" and evidence and _evidence_is_verbatim(
                evidence, d.text
            ):
                metadata = {**d.metadata, "rerank_evidence": evidence}
                relevant.append(SourceDocument(d.id, d.text, d.title, d.source_url, metadata, 1.0))
        return relevant

    def generate(self, question: str, documents: list[SourceDocument]) -> DraftAnswer:
        context = [
            {"id": d.id, "title": d.title, "text": d.text, "url": d.source_url} for d in documents
        ]
        data = self._json(
            "한국사 사료 해설자다. 컨텍스트 밖의 지식·수치·추측은 금지한다. 서로 상충하면 "
            "상충 사실을 밝히고 어느 사료를 따랐는지 설명한다. 근거가 전혀 없으면 정확히 "
            "'제공된 문서에서 확인되지 않는다.'라고 답한다. 일부만 확인되면 확인된 부분과 "
            "미확인 부분을 구분한다. 한 문장에 하나의 사실만 쓰고 모든 사실에 출처 ID와 "
            "원문 인용을 붙인다. JSON 외에는 출력하지 않는다.",
            f"질문: {question}\n사료: {json.dumps(context, ensure_ascii=False)}\n"
            "키: answer, citations. citations 값: [{source_id, quote}]",
        )
        citations = tuple(
            Citation(str(c.get("source_id", "")), str(c.get("quote", "")))
            for c in data.get("citations", [])
        )
        return DraftAnswer(str(data.get("answer") or "근거가 부족합니다."), citations)

    def verify_semantics(
        self, question: str, draft: DraftAnswer, documents: list[SourceDocument]
    ) -> dict[str, Any]:
        context = [{"id": d.id, "text": d.text} for d in documents]
        return self._json(
            "역사 RAG 답변 검증자다. 생성과 별도 호출로 검증한다. 초안을 원자적 사실 "
            "주장으로 분해하고, 사료에 명시된 경우에만 supported=true로 판정한다. "
            "상식적으로 타당해도 문서에 없거나 조건 누락·과장이 있으면 false다. "
            "지원되지 않는 주장은 수정본에서 제거한다. JSON 외에는 출력하지 않는다.",
            f"질문: {question}\n초안: {draft.answer}\n사료: "
            f"{json.dumps(context, ensure_ascii=False)}\n"
            "키: claims([{claim,supported,doc_id,quote}]), faithfulness(0~1), "
            "answerable(boolean), revised_answer, removed_reason(문자열 배열)",
        )
