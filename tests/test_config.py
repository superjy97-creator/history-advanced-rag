from __future__ import annotations

import unittest

from history_rag.config import Settings


class ConfigTests(unittest.TestCase):
    def test_large_embedding_defaults_to_3072_dimensions(self) -> None:
        settings = Settings("openai", "pinecone", "index")
        self.assertEqual(settings.embedding_model, "text-embedding-3-large")
        self.assertEqual(settings.embedding_dimension, 3072)
        settings.validate_live()

    def test_model_dimension_mismatch_is_rejected(self) -> None:
        settings = Settings(
            "openai",
            "pinecone",
            "index",
            embedding_model="text-embedding-3-large",
            embedding_dimension=1536,
        )
        with self.assertRaisesRegex(RuntimeError, "기본 차원"):
            settings.validate_live()

    def test_rag_requires_explicit_gemini_configuration(self) -> None:
        settings = Settings("openai", "pinecone", "index")
        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY, GEMINI_MODEL"):
            settings.validate_rag()


if __name__ == "__main__":
    unittest.main()
