from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.parsers_joseon import parse_file, parse_xml_preserving_hanja, write_jsonl


class JoseonParserTest(TestCase):
    def _xml(self, content: str) -> tuple[TemporaryDirectory[str], Path]:
        temp = TemporaryDirectory()
        path = Path(temp.name) / "sample.xml"
        path.write_text(content, encoding="utf-8")
        return temp, path

    def test_sillok_preserves_hanja_date_and_metadata(self) -> None:
        temp, path = self._xml(
            """<?xml version="1.0" encoding="UTF-8"?>
            <level1><level5 id="waa_10107017_001"><front><biblioData>
            <title><mainTitle>왕위에 오르다</mainTitle></title>
            <date><dateOccured type="서기" date="1392-07-17L0"/></date>
            </biblioData></front><text><content><paragraph>太祖가 壽昌宮에서
            즉위하다.</paragraph></content></text></level5></level1>"""
        )
        self.addCleanup(temp.cleanup)
        document = list(parse_file(path, "sillok"))[0]
        self.assertEqual(document["id"], "sillok:waa_10107017_001")
        self.assertEqual(document["text"], "太祖가 壽昌宮에서 즉위하다.")
        self.assertEqual(document["metadata"]["date"], "1392-07-17")
        self.assertEqual(document["metadata"]["rights_status"], "conditional_allowed")

    def test_invalid_utf16_numeric_pair_is_combined_without_editing_source(self) -> None:
        temp, path = self._xml("<root><name>李&#55381;&#56892;</name></root>")
        self.addCleanup(temp.cleanup)
        root = parse_xml_preserving_hanja(path)
        self.assertEqual(root.findtext("name"), "李𥘼")
        self.assertIn("&#55381;&#56892;", path.read_text(encoding="utf-8"))

    def test_law_parser_keeps_title_and_original_text(self) -> None:
        temp, path = self._xml(
            """<item><level3 id="jlawb_010_0000_0010"><front><biblioData>
            <title><mainTitle>大典續錄序</mainTitle></title></biblioData></front>
            <text><content><paragraph>夫創業之主, 立經陳紀。</paragraph></content></text>
            </level3></item>"""
        )
        self.addCleanup(temp.cleanup)
        document = list(parse_file(path, "laws"))[0]
        self.assertEqual(document["title"], "大典續錄序")
        self.assertEqual(document["metadata"]["document"], "조선시대 법령자료")

    def test_gojong_parser_keeps_reference_text_and_date(self) -> None:
        temp, path = self._xml(
            """<item><level5 id="gj_001_a00_010_010_010"><front><biblioData>
            <title><mainTitle>王, 昌德宮에서 昇遐</mainTitle></title>
            <date><dateOccured date="1863-12-08L0">12月 8日</dateOccured></date>
            </biblioData></front><text><content><paragraph>王이 昇遐하다.</paragraph>
            <reference><pTitle>哲宗實錄 卷15</pTitle></reference></content></text>
            </level5></item>"""
        )
        self.addCleanup(temp.cleanup)
        document = list(parse_file(path, "gojong"))[0]
        self.assertIn("哲宗實錄 卷15", document["text"])
        self.assertEqual(document["metadata"]["era"], "근대")
        self.assertEqual(document["metadata"]["date"], "1863-12-08")

    def test_writer_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "documents.jsonl"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_jsonl([], output)
