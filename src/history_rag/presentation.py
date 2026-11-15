from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import Citation


def citation_number_map(citations: Iterable[Citation]) -> dict[str, int]:
    """검증된 인용의 내부 ID를 화면용 번호에 안정적으로 연결한다."""
    numbers: dict[str, int] = {}
    for citation in citations:
        if citation.source_id and citation.source_id not in numbers:
            numbers[citation.source_id] = len(numbers) + 1
    return numbers


def public_citation_text(
    text: str,
    numbers: Mapping[str, int],
    known_source_ids: Iterable[str] = (),
) -> str:
    """검증된 ID만 번호로 바꾸고, 미검증 ID에는 번호를 부여하지 않는다."""
    public_text = text
    for source_id in sorted(numbers, key=len, reverse=True):
        public_text = public_text.replace(f"[{source_id}]", f"[{numbers[source_id]}]")

    unresolved = set(known_source_ids) - set(numbers)
    for source_id in sorted(unresolved, key=len, reverse=True):
        public_text = public_text.replace(f"[{source_id}]", "[출처 확인 필요]")
    return public_text


def public_issue_text(text: str, source_titles: Mapping[str, str]) -> str:
    """검증 경고의 내부 ID를 사용자가 이해할 수 있는 사료 제목으로 바꾼다."""
    public_text = text
    for source_id in sorted(source_titles, key=len, reverse=True):
        public_text = public_text.replace(source_id, f"‘{source_titles[source_id]}’")
    return public_text


def public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """원본 메타데이터를 바꾸지 않고 화면에서 내부 식별자 필드만 감춘다."""
    return {
        key: value
        for key, value in metadata.items()
        if key.lower() != "id" and not key.lower().endswith("_id")
    }


def public_trace(value: Any) -> Any:
    """실행 기록의 구조는 유지하되 외부 서비스 원시 오류문은 공개하지 않는다."""
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            if key.lower() in {"error", "exception"}:
                cleaned[key] = "외부 서비스 오류 발생" if item else ""
            else:
                cleaned[key] = public_trace(item)
        return cleaned
    if isinstance(value, list):
        return [public_trace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_trace(item) for item in value)
    return value

