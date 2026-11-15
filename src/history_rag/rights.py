from __future__ import annotations

import json
from pathlib import Path

from .models import SourceDocument

REQUIRED_RIGHTS_FIELDS = {
    "dataset",
    "provider",
    "license",
    "license_source_url",
    "rights_evidence_url",
    "rights_status",
    "ai_training_marking",
    "rights_verified_at",
}


def validate_external_use(documents: list[SourceDocument]) -> None:
    """외부 임베딩·Pinecone 적재 전에 권한 메타데이터를 강제한다."""
    for document in documents:
        missing = REQUIRED_RIGHTS_FIELDS - document.metadata.keys()
        if missing:
            raise ValueError(f"{document.id}: 권한 메타데이터 누락: {sorted(missing)}")
        if document.metadata["rights_status"] != "conditional_allowed":
            raise ValueError(f"{document.id}: 외부 사용 승인 대상이 아닙니다.")
        if document.metadata["ai_training_marking"] != "explicit":
            raise ValueError(f"{document.id}: AI 학습용 직접 표시가 확인되지 않았습니다.")
        if not document.source_url.startswith("https://www.data.go.kr/data/"):
            raise ValueError(f"{document.id}: 공식 데이터셋 상세 URL이 아닙니다.")
        if document.metadata["rights_evidence_url"] != document.source_url:
            raise ValueError(f"{document.id}: 권한 근거 URL과 데이터셋 상세 URL이 다릅니다.")


def validate_rights_registry(documents: list[SourceDocument], registry_path: Path) -> None:
    """문서의 자체 메타데이터를 검토 완료된 권한 대장과 교차 확인한다."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    by_dataset = {item["dataset"]: item for item in registry["datasets"]}
    for document in documents:
        dataset = str(document.metadata["dataset"])
        record = by_dataset.get(dataset)
        if record is None:
            raise ValueError(f"{document.id}: 권한 대장에 없는 데이터셋입니다: {dataset}")
        if record.get("rights_status") != "조건부 허용":
            raise ValueError(f"{document.id}: 권한 대장이 외부 적재를 허용하지 않습니다.")
        if record.get("ai_use") != "명시적 허용":
            raise ValueError(f"{document.id}: 권한 대장에 AI 직접 허용 표시가 없습니다.")
        if record.get("official_page") != document.source_url:
            raise ValueError(f"{document.id}: 문서와 권한 대장의 공식 URL이 다릅니다.")


def attribution_text(document: SourceDocument) -> str:
    provider = str(document.metadata.get("provider", "제공기관 미상"))
    dataset = str(document.metadata.get("dataset", document.title))
    license_name = str(document.metadata.get("license", "이용조건 미상"))
    return (
        f"본 사료는 {provider}가 제공한 ‘{dataset}’을 이용했으며, "
        f"{license_name} 조건에 따라 출처를 표시합니다. 공식 출처: {document.source_url}"
    )
