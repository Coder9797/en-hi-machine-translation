import unittest


class TransformerSmoke(unittest.TestCase):
    def test_one_step_and_beam(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        from src.train_transformer import smoke

        smoke()


if __name__ == "__main__":
    unittest.main()
