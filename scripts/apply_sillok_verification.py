from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_CHARACTER = "𥘼"


def apply_verification(input_path: Path, audit_path: Path, output_path: Path) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(f"기존 파일은 덮어쓰지 않습니다: {output_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verified = {
        item["article_id"]: item
        for item in audit["results"]
        if item["status"] == "verified_official_match"
    }
    if len(verified) != 45 or audit["occurrences"] != 49:
        raise ValueError("45개 기사·49개 문자 전체가 공식 대조 완료되지 않았습니다.")

    documents = verified_documents = occurrences = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            row = json.loads(line)
            documents += 1
            count = row["text"].count(TARGET_CHARACTER)
            if count:
                article_id = str(row["metadata"]["article_id"])
                result = verified.get(article_id)
                if result is None or result["occurrences"] != count:
                    raise ValueError(f"검증 근거가 없거나 문자 수가 다른 기사: {article_id}")
                metadata = dict(row["metadata"])
                metadata.update(
                    {
                        "text_verification_status": "verified_official_match",
                        "text_verification_checked_at": audit["checked_at"],
                        "text_verification_source_url": result["official_url"],
                        "official_record_url": result["official_url"],
                        "verified_character": "U+2563C 𥘼",
                        "verified_character_occurrences": count,
                    }
                )
                row["metadata"] = metadata
                verified_documents += 1
                occurrences += count
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "documents": documents,
        "verified_documents": verified_documents,
        "verified_occurrences": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="공식 실록 대조 결과를 적재 메타데이터에 반영")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            apply_verification(args.input, args.audit, args.output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
