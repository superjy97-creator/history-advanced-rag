from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.ingest_pinecone import load_resume_offset, write_checkpoint

from history_rag.models import SourceDocument
from history_rag.vectorstore import PineconeHistoryStore


class PineconePreparationTests(unittest.TestCase):
    def test_metadata_limit_accepts_final_shape(self) -> None:
        document = SourceDocument(
            id="sample:1",
            text="원문",
            title="제목",
            source_url="https://example.test",
            metadata={"dataset_key": "sample", "article_id": "1", "tags": ["고대"]},
        )
        self.assertLess(PineconeHistoryStore.validate_metadata(document), 40_960)

    def test_langchain_document_preserves_identity_and_source(self) -> None:
        source = SourceDocument(
            id="annals:1",
            text="원문",
            title="조선왕조실록",
            source_url="https://example.test/annals/1",
            metadata={"dataset_key": "annals", "article_id": "1"},
        )
        document = PineconeHistoryStore.langchain_document(source)
        self.assertEqual(document.page_content, source.text)
        self.assertEqual(document.metadata["document_id"], source.id)
        self.assertEqual(document.metadata["source_url"], source.source_url)

    def test_resume_checkpoint_rejects_other_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            write_checkpoint(
                path,
                {"input_sha256": "old", "namespace": "test", "total": 14, "next_offset": 7},
            )
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                load_resume_offset(path, "new", "test", 14)

    def test_remote_verification_checks_ids_dataset_counts_and_metadata(self) -> None:
        documents = [
            SourceDocument(
                id=f"dataset:{number}",
                text=f"원문 {number}",
                title="사료",
                source_url=f"https://example.test/{number}",
                metadata={"dataset_key": "dataset", "article_id": str(number)},
            )
            for number in range(3)
        ]
        records = {
            document.id: SimpleNamespace(
                metadata=PineconeHistoryStore.metadata_for(document)
            )
            for document in documents
        }

        class FakeIndex:
            def list(self, namespace):
                self.namespace = namespace
                return [[document.id for document in documents]]

            def fetch(self, ids, namespace):
                return SimpleNamespace(
                    vectors={document_id: records[document_id] for document_id in ids}
                )

        index = FakeIndex()
        store = object.__new__(PineconeHistoryStore)
        store.pc = SimpleNamespace(Index=lambda name: index)
        store.index_name = "index"
        store.namespace = "namespace"
        result = store.verify_remote_corpus(documents)
        self.assertEqual(result["remote_ids"], 3)
        self.assertEqual(result["remote_by_dataset"], {"dataset": 3})
        self.assertEqual(result["sample_records_verified"], 3)


if __name__ == "__main__":
    unittest.main()
