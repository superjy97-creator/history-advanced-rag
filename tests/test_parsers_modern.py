from pathlib import Path
from unittest import TestCase

from scripts.parsers_modern import DATASETS, find_dataset, parse_xml_file


class ModernParserTests(TestCase):
    def setUp(self) -> None:
        import tempfile

        self._temp = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_parse_modern_xml_preserves_text_and_traceability(self) -> None:
        xml = self.temp_path / "sample.xml"
        xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<item><level3 id="modern_001"><front><biblioData>
<title><mainTitle>3·1 독립선언 관련 문서</mainTitle></title>
<date><dateOccured>1919-03-01</dateOccured></date>
</biblioData></front><text><content>
<paragraph>오등은 자에 아 조선의 독립국임과 조선인의 자주민임을 선언하노라.</paragraph>
<paragraph>此로써 세계만방에 고하여 인류평등의 대의를 극명하며.</paragraph>
</content></text></level3></item>""",
            encoding="utf-8",
        )
        spec = DATASETS["삼일운동DB 데이터_격문"]

        rows = list(parse_xml_file(xml, spec))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], "data_15080861:modern_001")
        self.assertIn("조선의 독립국임", row["text"])
        self.assertIn("此로써", row["text"])
        self.assertEqual(row["metadata"]["date"], "1919-03-01")
        self.assertEqual(row["metadata"]["article_id"], "modern_001")
        self.assertEqual(row["metadata"]["xml_file"], "sample.xml")
        self.assertEqual(row["metadata"]["rights_status"], "conditional_allowed")
        self.assertEqual(row["metadata"]["ai_training_marking"], "explicit")
        self.assertEqual(row["metadata"]["rights_evidence_url"], row["source_url"])

    def test_short_or_metadata_only_nodes_are_excluded(self) -> None:
        xml = self.temp_path / "short.xml"
        xml.write_text(
            "<item><level3 id='index'><front><biblioData><title>"
            "<mainTitle>목록</mainTitle></title></biblioData></front>"
            "<text><content><paragraph>제목뿐</paragraph></content></text>"
            "</level3></item>",
            encoding="utf-8",
        )
        self.assertEqual(list(parse_xml_file(xml, DATASETS["자료대한민국사"])), [])

    def test_find_dataset_rejects_ambiguous_folder(self) -> None:
        (self.temp_path / "자료대한민국사_a").mkdir()
        (self.temp_path / "자료대한민국사_b").mkdir()
        with self.assertRaisesRegex(ValueError, "하나로 결정"):
            find_dataset(self.temp_path, "자료대한민국사")

    def test_only_explicit_ai_full_text_datasets_are_registered(self) -> None:
        datasets = {spec.dataset for spec in DATASETS.values()}
        self.assertFalse(any("한국독립운동사자료 원문" in name for name in datasets))
        self.assertFalse(any("근현대인물" in name for name in datasets))
        self.assertFalse(any("탄압기구" in name for name in datasets))
