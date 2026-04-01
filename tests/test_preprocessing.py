import unittest

from hybrid_search.core.config import PreprocessingConfig
from hybrid_search.core.preprocessing import TextPreprocessor


class TextPreprocessorTest(unittest.TestCase):
    def test_normalize_cleans_markup_and_whitespace(self) -> None:
        preprocessor = TextPreprocessor(PreprocessingConfig())
        text = " <b>Hello</b>\n\tМИР  "
        self.assertEqual(preprocessor.normalize(text), "hello мир")

    def test_tokenize_is_deterministic(self) -> None:
        preprocessor = TextPreprocessor(PreprocessingConfig())
        self.assertEqual(preprocessor.tokenize("Semantic search, search!"), ["semantic", "search", "search"])


if __name__ == "__main__":
    unittest.main()
