from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from history_rag.corpus import stable_document_id

SOURCE_URL = "https://www.data.go.kr/data/15053635/fileData.do"
LICENSE_URL = "https://www.kogl.or.kr/info/license.do"
DATASET = "교육부 국사편찬위원회_한국사데이터베이스 정보_삼국사기 원문_20260812"


def normalized_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def parse_xml(raw: bytes, filename: str) -> list[dict]:
    root = ET.fromstring(raw)
    volume = normalized_text(root.find("./front/biblioData/title/mainTitle"))
    rows = []
    for node in root.iter("level3"):
        source_id = node.attrib.get("id", "")
        title = normalized_text(node.find("./front/biblioData/title/mainTitle")) or volume
        date = normalized_text(node.find("./front/biblioData/date/dateOccured"))
        paragraphs = [normalized_text(p) for p in node.findall("./text/content/paragraph")]
        paragraphs = [p for p in paragraphs if p]
        if not source_id or not paragraphs:
            continue
        text = "\n".join(paragraphs)
        rows.append(
            {
                "id": stable_document_id("samguksagi", source_id),
                "text": text,
                "title": title,
                "source_url": SOURCE_URL,
                "metadata": {
                    "dataset": DATASET,
                    "dataset_key": "samguksagi",
                    "literature": "삼국사기",
                    "era": "고대",
                    "article_id": source_id,
                    "volume": volume,
                    "date": date,
                    "xml_file": filename,
                    "document_type": "원문",
                    "provider": "교육부 국사편찬위원회",
                    "license": "공공누리 제1유형 (출처표시)",
                    "license_ai_designation": "인공지능 학습용",
                    "license_source_url": LICENSE_URL,
                    "license_verified_at": "2026-11-14",
                    "rights_status": "conditional_allowed",
                    "ai_training_marking": "explicit",
                    "rights_verified_at": "2026-11-14",
                    "rights_evidence_url": SOURCE_URL,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="검증된 삼국사기 ZIP을 JSONL로 변환")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {args.output}")
    rows = []
    with zipfile.ZipFile(args.input) as archive:
        bad = archive.testzip()
        if bad:
            raise zipfile.BadZipFile(f"CRC 오류: {bad}")
        for info in archive.infolist():
            if info.filename.lower().endswith(".xml"):
                rows.extend(parse_xml(archive.read(info), info.filename))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
