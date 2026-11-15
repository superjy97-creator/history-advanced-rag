from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .models import SourceDocument

SILLOK_SEARCH_URL = "https://sillok.history.go.kr/search/searchResultList.do"
SILLOK_ITEM_URL = "https://sillok.history.go.kr/id/{article_id}"
USER_AGENT = "history-rag-course-project/1.0 (educational source verification)"


def _plain(fragment: str) -> str:
    value = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _section(page: str, heading: str) -> str:
    match = re.search(
        rf'<h4 class="view-title">{heading}</h4>\s*<div class="view-text">(.*?)</div>',
        page,
        re.DOTALL,
    )
    return _plain(match.group(1)) if match else ""


def _record_header(page: str) -> tuple[str, str]:
    """공식 상세 화면의 기사 일자와 제목을 구조화된 영역에서 읽는다."""
    match = re.search(
        r'<div class="title-head">.*?<p class="date">(.*?)</p>\s*<h3>(.*?)</h3>',
        page,
        re.DOTALL,
    )
    if not match:
        return "", ""
    return _plain(match.group(1)), _plain(match.group(2))


@dataclass(frozen=True)
class OfficialSearchTrace:
    query: str
    result_count: int
    detail_requests: int
    error: str = ""


class OfficialHistoryClient:
    """국사편찬위원회 조선왕조실록의 공개 검색 화면을 낮은 빈도로 조회한다."""

    def __init__(
        self,
        timeout_seconds: float = 12.0,
        fetcher: Callable[[urllib.request.Request, float], str] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._fetcher = fetcher or self._fetch
        self.last_trace = OfficialSearchTrace("", 0, 0)

    @staticmethod
    def _fetch(request: urllib.request.Request, timeout: float) -> str:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")

    def search(self, query: str, top_k: int = 3) -> list[SourceDocument]:
        clean_query = re.sub(r"\s+", " ", query).strip()
        if not clean_query or len(clean_query) > 200:
            raise ValueError("공식 사료 검색어는 1~200자여야 합니다.")
        limit = max(1, min(int(top_k), 3))
        try:
            page = self._search_page(clean_query)
            candidates = self._parse_candidates(page)[:limit]
            query_used = clean_query
            if not candidates and " " in clean_query:
                tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", clean_query)
                fallback = max(tokens, key=len, default="")
                if fallback and fallback != clean_query:
                    page = self._search_page(fallback)
                    candidates = self._parse_candidates(page)[:limit]
                    query_used = f"{clean_query} -> {fallback}"
            documents = [self._fetch_detail(article_id, title) for article_id, title in candidates]
            self.last_trace = OfficialSearchTrace(query_used, len(documents), len(candidates))
            return documents
        except Exception as exc:  # 외부 검색 장애가 Pinecone 검색까지 중단시키면 안 된다.
            self.last_trace = OfficialSearchTrace(clean_query, 0, 0, f"{type(exc).__name__}: {exc}")
            return []

    def _search_page(self, query: str) -> str:
        payload = urllib.parse.urlencode(
            {"topSearchWord": query, "topSearchWord_ime": query}
        ).encode("utf-8")
        request = urllib.request.Request(
            SILLOK_SEARCH_URL,
            data=payload,
            headers={"User-Agent": USER_AGENT},
            method="POST",
        )
        return self._fetcher(request, self.timeout_seconds)

    @staticmethod
    def _parse_candidates(page: str) -> list[tuple[str, str]]:
        pattern = re.compile(
            r'<a href="javascript:goView\(\'([^\']+)\',\s*\d+\);" class="subject">(.*?)</a>',
            re.DOTALL,
        )
        seen: set[str] = set()
        results = []
        for article_id, title_html in pattern.findall(page):
            if article_id in seen:
                continue
            seen.add(article_id)
            results.append((article_id, re.sub(r"^\d+\.\s*", "", _plain(title_html))))
        return results

    def _fetch_detail(self, article_id: str, fallback_title: str) -> SourceDocument:
        source_url = SILLOK_ITEM_URL.format(article_id=urllib.parse.quote(article_id, safe="_"))
        request = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
        page = self._fetcher(request, self.timeout_seconds)
        record_date, record_title = _record_header(page)
        title_match = re.search(r"<h3>(.*?)</h3>", page, re.DOTALL)
        title = record_title or (_plain(title_match.group(1)) if title_match else fallback_title)
        translation = _section(page, "국역")
        original = _section(page, "원문")
        text = translation
        if original:
            text = f"{translation}\n\n[원문]\n{original}" if translation else original
        if not text:
            raise ValueError(f"공식 기사 본문을 찾지 못했습니다: {article_id}")
        if record_date:
            text = f"[공식 기사 일자] {record_date}\n[기사 제목] {title}\n\n{text}"
        western_years = tuple(dict.fromkeys(re.findall(r"(?<!\d)(\d{4})년", record_date)))
        return SourceDocument(
            id=f"official-sillok-{article_id}",
            text=text,
            title=title,
            source_url=source_url,
            metadata={
                "article_id": article_id,
                "dataset_key": "joseon_sillok",
                "provider": "국사편찬위원회 조선왕조실록",
                "retrieval_path": "official_live",
                "has_original_text": bool(original),
                "record_date": record_date,
                "event_title": title,
                "western_years": western_years,
            },
            score=1.0,
        )
