"""Checks for pair cleaning. Does not download the corpus."""

import unittest

from src.data import clean_rows

CFG = {"max_words": 40, "max_chars": 240, "length_ratio": 4.0}


class CleanTests(unittest.TestCase):
    def test_keeps_a_normal_pair(self):
        rows = clean_rows([("The bank is closed.", "बैंक बंद है।")], CFG)
        self.assertEqual(len(rows), 1)

    def test_drops_copies_empty_and_unbalanced_pairs(self):
        rows = clean_rows(
            [
                ("The bank is closed.", "बैंक बंद है।"),
                ("The bank is closed.", "बैंक बंद है।"),
                ("", "बैंक"),
                ("hello", "hello"),
                ("no hindi here", "still latin"),
                ("Hi", "यह वाक्य जानबूझकर बहुत लंबा है " * 30),
            ],
            CFG,
        )
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
