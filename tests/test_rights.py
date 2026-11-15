from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from history_rag.models import SourceDocument
from history_rag.rights import attribution_text, validate_external_use, validate_rights_registry


def licensed_document(**metadata_overrides: str) -> SourceDocument:
    metadata = {
        "dataset": "삼국사기 원문",
        "provider": "교육부 국사편찬위원회",
        "license": "공공누리 제1유형",
        "license_source_url": "https://www.kogl.or.kr/info/licenseType1.do",
        "rights_status": "conditional_allowed",
        "ai_training_marking": "explicit",
        "rights_verified_at": "2026-11-14",
        "rights_evidence_url": "https://www.data.go.kr/data/15053635/fileData.do",
    }
    metadata.update(metadata_overrides)
    return SourceDocument(
        id="sg-test",
        text="원문",
        title="기사",
        source_url="https://www.data.go.kr/data/15053635/fileData.do",
        metadata=metadata,
    )


class RightsTests(unittest.TestCase):
    def test_external_use_requires_explicit_ai_marking(self):
        document = licensed_document(ai_training_marking="not_marked")
        with self.assertRaisesRegex(ValueError, "AI 학습용"):
            validate_external_use([document])

    def test_external_use_accepts_verified_conditional_document(self):
        validate_external_use([licensed_document()])

    def test_attribution_contains_provider_dataset_license_and_url(self):
        document = licensed_document()
        text = attribution_text(document)
        self.assertIn("교육부 국사편찬위원회", text)
        self.assertIn("삼국사기 원문", text)
        self.assertIn("공공누리 제1유형", text)
        self.assertIn(document.source_url, text)

    def test_registry_cross_check_rejects_unapproved_dataset(self):
        registry = {
            "datasets": [
                {
                    "dataset": "삼국사기 원문",
                    "rights_status": "추가 확인 필요",
                    "ai_use": "미확인",
                    "official_page": "https://www.data.go.kr/data/15053635/fileData.do",
                }
            ]
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rights.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "허용하지 않습니다"):
                validate_rights_registry([licensed_document()], path)


if __name__ == "__main__":
    unittest.main()
