from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    id: str
    text: str
    title: str
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass(frozen=True)
class TransformedQuery:
    original: str
    search_query: str
    queries: tuple[str, ...]
    era: str | None
    people: tuple[str, ...]
    events: tuple[str, ...]
    keywords: tuple[str, ...]
    intent: str


@dataclass(frozen=True)
class Citation:
    source_id: str
    quote: str


@dataclass(frozen=True)
class DraftAnswer:
    answer: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class VerificationResult:
    status: str
    final_answer: str
    supported_citations: tuple[Citation, ...]
    issues: tuple[str, ...]
    faithfulness: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    query: TransformedQuery
    retrieved: tuple[SourceDocument, ...]
    reranked: tuple[SourceDocument, ...]
    draft: DraftAnswer
    verification: VerificationResult
    trace: dict[str, Any]
