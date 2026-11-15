from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from history_rag.corpus import stable_document_id

PROVIDER = "교육부 국사편찬위원회"
LICENSE = "공공누리 제1유형(인공지능 학습용)"
LICENSE_URL = "https://www.kogl.or.kr/info/licenseType1.do"
VERIFIED_AT = "2026-11-14"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    directory_name: str
    literature: str
    source_url: str
    era: str


DATASETS = {
    spec.key: spec
    for spec in (
        DatasetSpec(
            "samguk_yusa",
            "교육부 국사편찬위원회_한국사데이터베이스 정보_삼국유사 원문_20260810",
            "삼국유사",
            "https://www.data.go.kr/data/15053634/fileData.do",
            "고대",
        ),
        DatasetSpec(
            "goryeosa",
            "교육부 국사편찬위원회_한국사데이터베이스 정보_고려사 원문_20260810",
            "고려사",
            "https://www.data.go.kr/data/15053637/fileData.do",
            "고려",
        ),
        DatasetSpec(
            "ancient_inscriptions",
            "교육부 국사편찬위원회_한국사데이터베이스 정보_한국고대금석문_20260810",
            "한국고대금석문",
            "https://www.data.go.kr/data/15053630/fileData.do",
            "고대",
        ),
    )
}

LEVEL_TAG = re.compile(r"level\d+$")


def normalized_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _direct_title(node: ET.Element) -> str:
    return normalized_text(node.find("./front/biblioData/title/mainTitle"))


def _direct_date(node: ET.Element) -> str:
    dates = {
        normalized_text(date) or date.attrib.get("date", "")
        for date in node.findall("./front/biblioData/date/dateOccured")
    }
    return "; ".join(sorted(date for date in dates if date))


def _paragraphs(node: ET.Element) -> list[str]:
    paragraphs = [
        normalized_text(paragraph)
        for paragraph in node.findall("./text/content/paragraph")
    ]
    return [paragraph for paragraph in paragraphs if paragraph]


def parse_xml(raw: bytes, filename: str, spec: DatasetSpec) -> list[dict]:
    """Parse article units that own a direct text/content section.

    The National Institute of Korean History XML does not use one fixed level for
    articles. Selecting only level3, for example, drops most Samguk Yusa and
    Goryeosa records. Direct-child paths prevent a parent article from duplicating
    the text of nested articles.
    """
    root = ET.fromstring(raw)
    parent = {child: node for node in root.iter() for child in node}
    root_title = _direct_title(root)
    rows: list[dict] = []

    for node in root.iter():
        if not LEVEL_TAG.fullmatch(node.tag):
            continue
        paragraphs = _paragraphs(node)
        source_id = node.attrib.get("id", "").strip()
        if not source_id or not paragraphs:
            continue

        titles: list[str] = []
        date = ""
        cursor: ET.Element | None = node
        while cursor is not None:
            title = _direct_title(cursor)
            if title and title not in titles:
                titles.append(title)
            if not date:
                date = _direct_date(cursor)
            cursor = parent.get(cursor)
        title = titles[0] if titles else root_title or source_id
        hierarchy = list(reversed(titles))
        rows.append(
            {
                "id": stable_document_id(spec.key, source_id),
                "text": "\n".join(paragraphs),
                "title": title,
                "source_url": spec.source_url,
                "metadata": {
                    "dataset": spec.directory_name,
                    "dataset_key": spec.key,
                    "literature": spec.literature,
                    "document_type": "원문",
                    "era": spec.era,
                    "article_id": source_id,
                    "date": date,
                    "title_hierarchy": hierarchy,
                    "xml_file": filename,
                    "provider": PROVIDER,
                    "license": LICENSE,
                    "license_ai_designation": "인공지능 학습용",
                    "license_source_url": LICENSE_URL,
                    "license_verified_at": VERIFIED_AT,
                    "rights_status": "conditional_allowed",
                    "ai_training_marking": "explicit",
                    "rights_verified_at": VERIFIED_AT,
                    "rights_evidence_url": spec.source_url,
                    "attribution_required": True,
                },
            }
        )
    return rows


def parse_dataset(raw_root: Path, spec: DatasetSpec) -> list[dict]:
    dataset_dir = raw_root / spec.directory_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"데이터셋 디렉터리가 없습니다: {dataset_dir}")

    rows: list[dict] = []
    for xml_path in sorted(dataset_dir.glob("*.xml")):
        rows.extend(parse_xml(xml_path.read_bytes(), xml_path.name, spec))

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"중복 기사 ID: {duplicates[:10]}")
    return rows


def write_jsonl(rows: Iterable[dict], output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"기존 파일은 덮어쓰지 않습니다: {output}")
    materialized = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in materialized:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description="고대·고려 원문 XML을 공통 JSONL로 변환")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = parse_dataset(args.raw_root, DATASETS[args.dataset])
    count = write_jsonl(rows, args.output)
    print(f"prepared={count} dataset={args.dataset} output={args.output}")


if __name__ == "__main__":
    main()
