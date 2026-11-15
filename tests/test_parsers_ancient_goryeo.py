from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.parsers_ancient_goryeo import (
    DATASETS,
    parse_dataset,
    parse_xml,
    write_jsonl,
)

XML = """<?xml version="1.0" encoding="UTF-8"?>
<level1 id="volume">
  <front><biblioData><title><mainTitle>문헌명</mainTitle></title></biblioData></front>
  <level2 id="section">
    <front><biblioData><title><mainTitle>편명</mainTitle></title></biblioData></front>
    <level3 id="article-1">
      <front><biblioData><title><mainTitle>기사 제목</mainTitle></title>
        <date><dateOccured date="1145">인종 23년</dateOccured></date>
      </biblioData></front>
      <text><content><paragraph>첫 문장 <index type="인명">김부식</index></paragraph>
      <paragraph>둘째 문장</paragraph></content></text>
      <level4 id="article-2">
        <front><biblioData><title><mainTitle>하위 기사</mainTitle></title></biblioData></front>
        <text><content><paragraph>별도 본문</paragraph></content></text></level4>
    </level3>
  </level2>
</level1>
"""


class AncientGoryeoParserTests(unittest.TestCase):
    def test_parses_variable_article_depth_without_parent_duplication(self) -> None:
        rows = parse_xml(XML.encode(), "sample.xml", DATASETS["goryeosa"])
        self.assertEqual(
            [row["id"] for row in rows],
            ["goryeosa:article-1", "goryeosa:article-2"],
        )
        self.assertEqual(rows[0]["text"], "첫 문장 김부식\n둘째 문장")
        self.assertNotIn("별도 본문", rows[0]["text"])
        self.assertEqual(rows[0]["metadata"]["date"], "인종 23년")
        self.assertEqual(rows[1]["metadata"]["date"], "인종 23년")
        self.assertEqual(rows[0]["metadata"]["title_hierarchy"], ["문헌명", "편명", "기사 제목"])

    def test_preserves_rights_and_source_metadata(self) -> None:
        row = parse_xml(XML.encode(), "sample.xml", DATASETS["samguk_yusa"])[0]
        self.assertEqual(row["source_url"], "https://www.data.go.kr/data/15053634/fileData.do")
        self.assertEqual(row["metadata"]["rights_status"], "conditional_allowed")
        self.assertEqual(row["metadata"]["ai_training_marking"], "explicit")
        self.assertTrue(row["metadata"]["attribution_required"])

    def test_dataset_rejects_duplicate_ids_across_files(self) -> None:
        spec = DATASETS["ancient_inscriptions"]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / spec.directory_name
            dataset.mkdir()
            (dataset / "one.xml").write_text(XML, encoding="utf-8")
            (dataset / "two.xml").write_text(XML, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "중복 기사 ID"):
                parse_dataset(Path(directory), spec)

    def test_writer_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            self.assertEqual(write_jsonl([], output), 0)
            with self.assertRaises(FileExistsError):
                write_jsonl([], output)


if __name__ == "__main__":
    unittest.main()
