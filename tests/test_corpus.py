from __future__ import annotations

import unittest

from history_rag.corpus import stable_document_id, validate_corpus
from history_rag.models import SourceDocument


def sample(document_id: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        text="史料 원문",
        title="기사 제목",
        source_url="https://www.data.go.kr/data/123/fileData.do",
        metadata={
            "dataset": "자료 전체 이름",
            "dataset_key": "samguksagi",
            "literature": "삼국사기",
            "era": "고대",
            "article_id": "abc-1",
            "document_type": "원문",
            "provider": "국사편찬위원회",
        },
    )


class CorpusTests(unittest.TestCase):
    def test_stable_id_is_namespaced(self) -> None:
        self.assertEqual(stable_document_id("samguksagi", "abc-1"), "samguksagi:abc-1")

    def test_validate_rejects_duplicate_ids(self) -> None:
        document = sample("samguksagi:abc-1")
        with self.assertRaisesRegex(ValueError, "중복 문서 ID"):
            validate_corpus([document, document])

    def test_validate_accepts_shared_contract(self) -> None:
        validate_corpus([sample("samguksagi:abc-1")])

    def test_unverified_sillok_character_repair_is_blocked(self) -> None:
        document = SourceDocument(
            id="sillok:abc-1",
            text="복원 문자 𥘼 포함",
            title="기사",
            source_url="https://www.data.go.kr/data/15053647/fileData.do",
            metadata={
                **sample("samguksagi:abc-1").metadata,
                "dataset_key": "sillok",
            },
        )
        with self.assertRaisesRegex(ValueError, "공식 원문 대조 전"):
            validate_corpus([document])

    def test_officially_verified_sillok_character_is_accepted(self) -> None:
        metadata = {
            **sample("samguksagi:abc-1").metadata,
            "dataset_key": "sillok",
            "text_verification_status": "verified_official_match",
        }
        document = SourceDocument(
            id="sillok:abc-1",
            text="공식 대조 문자 李𥘼",
            title="기사",
            source_url="https://www.data.go.kr/data/15053647/fileData.do",
            metadata=metadata,
        )
        validate_corpus([document])


if __name__ == "__main__":
    unittest.main()
