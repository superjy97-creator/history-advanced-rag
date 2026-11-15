from __future__ import annotations

import unittest

import tiktoken
from scripts.prepare_embedding_corpus import split_text, token_count


class EmbeddingCorpusTests(unittest.TestCase):
    def test_long_unicode_text_is_split_without_replacement_characters(self) -> None:
        encoding = tiktoken.get_encoding("cl100k_base")
        text = ("한글과 漢字가 섞인 역사 사료입니다.\n" * 200).strip()
        chunks = split_text(text, encoding, max_tokens=100, overlap_chars=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(token_count(chunk, encoding) <= 100 for chunk in chunks))
        self.assertTrue(all("�" not in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
