from __future__ import annotations

import unittest

from history_rag.models import Citation
from history_rag.presentation import (
    citation_number_map,
    public_citation_text,
    public_issue_text,
    public_metadata,
    public_trace,
)


class PresentationTests(unittest.TestCase):
    def test_verified_internal_id_is_linked_to_same_number(self):
        source_id = "official-sillok-kda_12512030_002"
        citations = (Citation(source_id, "실제 확인된 인용문"),)
        numbers = citation_number_map(citations)

        self.assertEqual(numbers, {source_id: 1})
        self.assertEqual(
            public_citation_text(f"해설 문장 [{source_id}]", numbers, (source_id,)),
            "해설 문장 [1]",
        )

    def test_unverified_internal_id_never_receives_source_number(self):
        source_id = "unverified-internal-id"
        self.assertEqual(
            public_citation_text(f"해설 [{source_id}]", {}, (source_id,)),
            "해설 [출처 확인 필요]",
        )

    def test_verification_issue_uses_title_instead_of_internal_id(self):
        source_id = "official-sillok-kda_12512030_002"
        issue = f"원문과 일치하지 않는 인용문: {source_id}"

        displayed = public_issue_text(issue, {source_id: "훈민정음을 창제하다"})

        self.assertNotIn(source_id, displayed)
        self.assertEqual(displayed, "원문과 일치하지 않는 인용문: ‘훈민정음을 창제하다’")

    def test_public_metadata_hides_ids_without_mutating_original(self):
        metadata = {
            "article_id": "internal-article",
            "dataset_key": "joseon_sillok",
            "provider": "국사편찬위원회",
        }
        displayed = public_metadata(metadata)

        self.assertNotIn("article_id", displayed)
        self.assertEqual(displayed["dataset_key"], "joseon_sillok")
        self.assertEqual(metadata["article_id"], "internal-article")

    def test_public_trace_masks_raw_external_error(self):
        trace = {"executions": [{"error": "raw provider error"}], "retrieval": "completed"}
        displayed = public_trace(trace)

        self.assertEqual(displayed["executions"][0]["error"], "외부 서비스 오류 발생")
        self.assertEqual(displayed["retrieval"], "completed")


if __name__ == "__main__":
    unittest.main()

