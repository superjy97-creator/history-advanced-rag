from __future__ import annotations

import unittest

from scripts.prepare_samguksagi import parse_xml


class PrepareTests(unittest.TestCase):
    def test_parse_level3_preserves_traceable_metadata(self):
        raw = b"""<level1><front><biblioData><title>
        <mainTitle>VOLUME</mainTitle></title></biblioData></front>
        <level3 id="doc-1"><front><biblioData><title><mainTitle>TITLE</mainTitle></title>
        <date><dateOccured>DATE</dateOccured></date></biblioData></front>
        <text><content><paragraph>ORIGINAL TEXT</paragraph></content></text></level3></level1>"""
        rows = parse_xml(raw, "sample.xml")
        self.assertEqual(rows[0]["id"], "samguksagi:doc-1")
        self.assertEqual(rows[0]["text"], "ORIGINAL TEXT")
        self.assertEqual(rows[0]["metadata"]["xml_file"], "sample.xml")
        self.assertTrue(rows[0]["source_url"].startswith("https://www.data.go.kr/"))


if __name__ == "__main__":
    unittest.main()
