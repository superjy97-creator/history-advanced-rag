"""조선시대 XML 사료를 공통 RAG 문서 형식으로 변환한다.

원본 XML은 변경하지 않는다. 조선왕조실록 일부 파일의 UTF-16 surrogate 숫자
참조만 메모리에서 유니코드 확장 한자로 결합한 뒤 XML을 해석한다.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from history_rag.corpus import stable_document_id

DATASETS: dict[str, dict[str, str]] = {
    "sillok": {
        "name": "교육부 국사편찬위원회_조선왕조실록 정보_실록원문_20260821",
        "document": "조선왕조실록",
        "era": "조선",
        "catalog_url": "https://www.data.go.kr/data/15053647/fileData.do",
    },
    "laws": {
        "name": "교육부 국사편찬위원회_한국사데이터베이스 정보_조선시대 법령자료_20260812",
        "document": "조선시대 법령자료",
        "era": "조선",
        "catalog_url": "https://www.data.go.kr/data/15060103/fileData.do",
    },
    "gojong": {
        "name": "교육부 국사편찬위원회_한국사데이터베이스 정보_고종시대사_20260812",
        "document": "고종시대사",
        "era": "근대",
        "catalog_url": "https://www.data.go.kr/data/15138768/fileData.do",
    },
}

_SURROGATE_PAIR = re.compile(r"&#(\d+);&#(\d+);")
_SPACE = re.compile(r"\s+")
_RECORD_TAGS = {
    "sillok": ("level3", "level4", "level5"),
    "laws": ("level2", "level3", "level4", "level5"),
    "gojong": ("level2", "level5"),
}


def _combine_surrogate_pair(match: re.Match[str]) -> str:
    high, low = (int(value) for value in match.groups())
    if 0xD800 <= high <= 0xDBFF and 0xDC00 <= low <= 0xDFFF:
        return chr(0x10000 + ((high - 0xD800) << 10) + low - 0xDC00)
    return match.group(0)


def parse_xml_preserving_hanja(path: Path) -> ET.Element:
    """UTF-8 XML을 읽고 잘못 직렬화된 surrogate 쌍만 안전하게 복원한다."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    if "&#" in text:
        text = _SURROGATE_PAIR.sub(_combine_surrogate_pair, text)
    return ET.fromstring(text)


def _clean_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _SPACE.sub(" ", "".join(element.itertext())).strip()


def _first_text(element: ET.Element, paths: Iterable[str]) -> str:
    for path in paths:
        value = _clean_text(element.find(path))
        if value:
            return value
    return ""


def _date(element: ET.Element) -> str:
    dates = element.findall("./front/biblioData/date/dateOccured")
    for date in dates:
        value = date.get("date", "")
        if value:
            return value.removesuffix("L0")
    return next((_clean_text(date) for date in dates if _clean_text(date)), "")


def _document(record: ET.Element, xml_file: Path, kind: str) -> dict[str, Any] | None:
    info = DATASETS[kind]
    article_id = record.get("id", "").strip()
    body = _clean_text(record.find("./text/content"))
    if not article_id or not body:
        return None
    title = _first_text(
        record,
        ("./front/biblioData/title/mainTitle", "./front/biblioData/title/seriesTitle"),
    )
    record_url = f"https://db.history.go.kr/id/{article_id}"
    return {
        "id": stable_document_id(kind, article_id),
        "text": body,
        "title": title or article_id,
        "source_url": info["catalog_url"],
        "metadata": {
            "dataset": info["name"],
            "dataset_key": kind,
            "literature": info["document"],
            "document": info["document"],
            "era": info["era"],
            "article_id": article_id,
            "date": _date(record),
            "xml_file": xml_file.name,
            "document_type": "원문",
            "provider": "교육부 국사편찬위원회",
            "catalog_url": info["catalog_url"],
            "official_record_url": record_url,
            "license": "공공누리 제1유형 (출처표시)",
            "license_source_url": "https://www.kogl.or.kr/info/license.do",
            "license_ai_designation": "인공지능 학습용",
            "rights_status": "conditional_allowed",
            "ai_training_marking": "explicit",
            "rights_verified_at": "2026-11-14",
            "rights_evidence_url": info["catalog_url"],
        },
    }


def parse_file(path: Path, kind: str) -> Iterator[dict[str, Any]]:
    """한 XML 파일의 기사 단위 문서를 반환한다."""
    if kind not in DATASETS:
        raise ValueError(f"지원하지 않는 자료 유형: {kind}")
    root = parse_xml_preserving_hanja(path)
    # 자료마다 기사 깊이가 일정하지 않다. 하위 기사의 본문을 합치지 않고,
    # 해당 계층에 직접 text/content가 있는 요소만 독립 문서로 취급한다.
    records = (
        record
        for tag in _RECORD_TAGS[kind]
        for record in root.iter(tag)
        if record.find("./text/content") is not None
    )
    for record in records:
        document = _document(record, path, kind)
        if document is not None:
            yield document


def iter_dataset(dataset_dir: Path, kind: str) -> Iterator[dict[str, Any]]:
    """파일명 순서로 전체 데이터셋을 결정적으로 순회한다."""
    for path in sorted(dataset_dir.rglob("*.xml")):
        yield from parse_file(path, kind)


def write_jsonl(documents: Iterable[dict[str, Any]], output: Path) -> int:
    """기존 파일을 덮어쓰지 않고 JSONL을 기록한다."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=DATASETS)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = write_jsonl(iter_dataset(args.dataset_dir, args.kind), args.output)
    print(f"{count:,}건 기록: {args.output}")


if __name__ == "__main__":
    main()
