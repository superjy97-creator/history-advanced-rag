from __future__ import annotations

import unittest

from scripts.build_demo_corpus import evenly_spaced_indices


class DemoCorpusTests(unittest.TestCase):
    def test_even_selection_has_exact_size_and_covers_range(self) -> None:
        selected = evenly_spaced_indices(100, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(min(selected), 0)
        self.assertGreater(max(selected), 80)

    def test_limit_larger_than_data_keeps_all(self) -> None:
        self.assertEqual(evenly_spaced_indices(3, 10), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
