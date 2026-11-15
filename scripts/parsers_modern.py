from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from history_rag.corpus import stable_document_id

PROVIDER = "교육부 국사편찬위원회"
LICENSE = "공공누리 제1유형 (출처표시)"
LICENSE_URL = "https://www.kogl.or.kr/info/license.do"
VERIFIED_AT = "2026-11-14"


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    source_url: str
    period: str


# 권한 대장에서 '조건부 허용'과 'AI 명시적 허용'이 함께 확인된 본문형 자료만 포함한다.
DATASETS: dict[str, DatasetSpec] = {
    "국외항일운동자료(일본외무성 기록)": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_국외항일운동자료(일본외무성 기록)_20260810",
        "https://www.data.go.kr/data/15083502/fileData.do",
        "일제강점기",
    ),
    "반민특위조사기록 원문": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_반민특위조사기록 원문_20260812",
        "https://www.data.go.kr/data/15114319/fileData.do",
        "현대",
    ),
    "삼일운동DB 데이터_격문": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_삼일운동DB 데이터_격문_20260812",
        "https://www.data.go.kr/data/15080861/fileData.do",
        "일제강점기",
    ),
    "일제침략하 한국36년사": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_일제침략하 한국36년사_20260810",
        "https://www.data.go.kr/data/15100189/fileData.do",
        "일제강점기",
    ),
    "자료대한민국사": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_자료대한민국사_20220222",
        "https://www.data.go.kr/data/15099108/fileData.do",
        "현대",
    ),
    "한국근대사자료집성 원문": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_한국근대사자료집성 원문_20260812",
        "https://www.data.go.kr/data/15053628/fileData.do",
        "근대",
    ),
    "한민족독립운동사자료집": DatasetSpec(
        "교육부 국사편찬위원회_한국사데이터베이스 정보_한민족독립운동사자료집_20260810",
        "https://www.data.go.kr/data/15100193/fileData.do",
        "일제강점기",
    ),
}


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _normalized(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _direct(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _tag(child) == name), None)


def _descendant(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((child for child in element.iter() if _tag(child) == name), None)


def _record(node: ET.Element, xml_file: str, spec: DatasetSpec) -> dict | None:
    front = _direct(node, "front")
    text_node = _direct(node, "text")
    if front is None or text_node is None:
        return None
    source_id = node.attrib.get("id", "").strip()
    if not source_id:
        return None
    paragraphs = [
        _normalized(element)
        for element in text_node.iter()
        if _tag(element) == "paragraph"
    ]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    text = "\n".join(paragraphs)
    # 목록 표제나 빈 껍데기를 검색 문서로 만들지 않는다.
    if len(text) < 40:
        return None
    title = _normalized(_descendant(front, "mainTitle")) or source_id
    date = _normalized(_descendant(front, "dateOccured")) or _normalized(
        _descendant(front, "dateIssued")
    )
    literature = next(name for name, item in DATASETS.items() if item is spec)
    dataset_key = "data_" + spec.source_url.split("/data/", 1)[1].split("/", 1)[0]
    return {
        "id": stable_document_id(dataset_key, source_id),
        "text": text,
        "title": title,
        "source_url": spec.source_url,
        "metadata": {
            "dataset": spec.dataset,
            "dataset_key": dataset_key,
            "literature": literature,
            "document_name": title,
            "era": spec.period,
            "period": spec.period,
            "date": date,
            "article_id": source_id,
            "xml_file": xml_file,
            "document_type": "원문",
            "provider": PROVIDER,
            "license": LICENSE,
            "license_source_url": LICENSE_URL,
            "license_ai_designation": "공공누리 AI 학습용",
            "license_verified_at": VERIFIED_AT,
            "rights_status": "conditional_allowed",
            "ai_training_marking": "explicit",
            "rights_verified_at": VERIFIED_AT,
            "rights_evidence_url": spec.source_url,
        },
    }


def parse_xml_file(path: Path, spec: DatasetSpec) -> Iterator[dict]:
    """국사편찬위원회 history XML에서 본문 단위만 메모리 안전하게 읽는다."""
    for _event, node in ET.iterparse(path, events=("end",)):
        row = _record(node, path.name, spec)
        if row is not None:
            yield row
            node.clear()


def find_dataset(raw_root: Path, key: str) -> Path:
    matches = [path for path in raw_root.iterdir() if path.is_dir() and key in path.name]
    if len(matches) != 1:
        raise ValueError(f"데이터셋 폴더를 하나로 결정할 수 없습니다: {key} ({len(matches)}개)")
    return matches[0]


def prepare_dataset(raw_root: Path, key: str, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {output}")
    spec = DATASETS[key]
    dataset_dir = find_dataset(raw_root, key)
    xml_files = sorted(dataset_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"XML 원문이 없습니다: {dataset_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for path in xml_files:
            for row in parse_xml_file(path, spec):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="권한 확인된 근현대 XML 사료를 공통 JSONL로 변환")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare_dataset(args.raw_root, args.dataset, args.output)
    print(f"prepared={count} output={args.output}")


if __name__ == "__main__":
    main()
