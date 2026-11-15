from __future__ import annotations

import unittest

from scripts.verify_sillok_official import contexts, official_article_id, visible_text


class VerifySillokOfficialTests(unittest.TestCase):
    def test_maps_distribution_id_to_official_id(self) -> None:
        self.assertEqual(official_article_id("wqa_10602030_001"), "kqa_10602030_001")
        self.assertEqual(official_article_id("wra_10401015_001"), "kra_10401015_001")
        self.assertEqual(official_article_id("wrb_10401015_001"), "krb_10401015_001")

    def test_visible_text_decodes_official_character(self) -> None:
        self.assertEqual(visible_text("<p>李&#153148;</p>"), "李𥘼")

    def test_contexts_returns_every_occurrence(self) -> None:
        self.assertEqual(len(contexts("甲𥘼乙𥘼丙", radius=1)), 2)


if __name__ == "__main__":
    unittest.main()
