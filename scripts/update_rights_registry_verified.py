from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("reports/rights_registry_20261114.json")
OUTPUT = Path("reports/rights_registry_20261114_v2.json")

VERIFIED = {
    "교육부 국사편찬위원회_우리역사넷 정보_오늘의역사(연표)_20211028": {
        "official_page": "https://www.data.go.kr/data/15053642/fileData.do",
        "license": "공공누리 제4유형(출처표시·상업적 이용금지·변경금지)",
        "license_evidence": "https://www.kogl.or.kr/info/license.do",
        "ai_use": "제한적/불명확(공식 페이지에 AI유형 표시 없음)",
        "openai_transfer": "금지(변경금지 자료의 외부 임베딩 전송 해석 미확인)",
        "pinecone_transfer": "금지(변경금지 자료의 청킹·벡터 저장 해석 미확인)",
        "review_status": "verified_restricted",
        "reviewed_at": "2026-11-14",
        "notes": "공식 상세 페이지가 제4유형을 명시. 별도 허가 전 외부 전송에서 제외.",
    },
    "교육부 국사편찬위원회_우리역사넷_한국사연대기_20260901": {
        "official_page": "https://www.data.go.kr/data/15155532/fileData.do",
        "license": "공공누리 제4유형(출처표시·상업적 이용금지·변경금지)",
        "license_evidence": "https://www.kogl.or.kr/info/license.do",
        "ai_use": "제한적/불명확(공식 페이지에 AI유형 표시 없음)",
        "openai_transfer": "금지(변경금지 자료의 외부 임베딩 전송 해석 미확인)",
        "pinecone_transfer": "금지(변경금지 자료의 청킹·벡터 저장 해석 미확인)",
        "review_status": "verified_restricted",
        "reviewed_at": "2026-11-14",
        "notes": (
            "설명에 AI 개발 활용 가능성이 언급되지만 이용허락범위는 제4유형. 별도 허가 전 제외."
        ),
    },
    "교육부 국사편찬위원회_조선왕조실록 정보_부가정보 인물 데이터_20260827": {
        "official_page": "https://www.data.go.kr/data/15053645/fileData.do",
        "license": "공공누리 제1유형(출처표시), 인공지능 학습용 표시",
        "license_evidence": "https://www.kogl.or.kr/info/license.do",
        "ai_use": "허용(출처표시 및 AI유형 조건 준수)",
        "openai_transfer": "승인 대기(권한 확인 완료, 사용자 승인 필요)",
        "pinecone_transfer": "승인 대기(권한 확인 완료, 사용자 승인 필요)",
        "review_status": "verified",
        "reviewed_at": "2026-11-14",
        "notes": "공식 상세 페이지가 제1유형과 인공지능 학습용을 함께 표시.",
    },
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {OUTPUT}")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    for record in data["datasets"]:
        update = VERIFIED.get(record["dataset"])
        if update:
            record.update(update)
    data["generated_at"] = "2026-11-14"
    data["verified_count"] = sum(r["review_status"] == "verified" for r in data["datasets"])
    data["verified_restricted_count"] = sum(
        r["review_status"] == "verified_restricted" for r in data["datasets"]
    )
    data["pending_count"] = sum(r["review_status"] == "pending" for r in data["datasets"])
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"verified={data['verified_count']} "
        f"verified_restricted={data['verified_restricted_count']} "
        f"pending={data['pending_count']}"
    )


if __name__ == "__main__":
    main()
