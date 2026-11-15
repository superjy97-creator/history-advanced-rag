from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMGUKSAGI_NAME = "교육부 국사편찬위원회_한국사데이터베이스 정보_삼국사기 원문_20260812"
SAMGUKSAGI_SOURCE = "https://www.data.go.kr/data/15053635/fileData.do"
KOGL_TERMS = "https://www.kogl.or.kr/info/license.do"


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터셋별 이용권한 검토대장 생성")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {args.output}")

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    records = []
    for dataset in audit["datasets"]:
        record = {
            "dataset": dataset["dataset"],
            "source_file": dataset["source_file"],
            "provider": dataset["dataset"].split("_", 1)[0],
            "official_page": None,
            "license": "미확인",
            "license_evidence": None,
            "ai_use": "미확인",
            "openai_transfer": "금지(확인 전)",
            "pinecone_transfer": "금지(확인 전)",
            "review_status": "pending",
            "reviewed_at": None,
            "notes": "공식 상세 페이지의 현재 이용허락범위와 AI 활용 조건 확인 필요",
        }
        if dataset["dataset"] == SAMGUKSAGI_NAME:
            record.update(
                {
                    "official_page": SAMGUKSAGI_SOURCE,
                    "license": "공공누리 제1유형(출처표시), 인공지능 학습용 표시",
                    "license_evidence": KOGL_TERMS,
                    "ai_use": "허용(출처표시 및 AI유형 조건 준수)",
                    "openai_transfer": "승인 대기(권한 확인 완료, 사용자 승인 필요)",
                    "pinecone_transfer": "승인 대기(권한 확인 완료, 사용자 승인 필요)",
                    "review_status": "verified",
                    "reviewed_at": "2026-11-14",
                    "notes": (
                        "공식 페이지가 제1유형과 인공지능 학습용을 함께 표시. "
                        "직접 인용 시 출처 표시, 동일·실질 유사 산출 방지 조건을 적용."
                    ),
                }
            )
        records.append(record)

    output = {
        "generated_at": "2026-11-14",
        "policy": "verified가 아닌 데이터셋은 외부 API와 Pinecone에 전송하지 않는다.",
        "verified_count": sum(r["review_status"] == "verified" for r in records),
        "pending_count": sum(r["review_status"] == "pending" for r in records),
        "datasets": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verified={output['verified_count']} pending={output['pending_count']}")


if __name__ == "__main__":
    main()
