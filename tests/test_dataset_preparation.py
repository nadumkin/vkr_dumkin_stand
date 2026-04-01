import tempfile
import unittest
from pathlib import Path

from hybrid_search.experiments.dataset_preparation import DatasetPreparer
from hybrid_search.io.data import load_jsonl


class DatasetPreparationTest(unittest.TestCase):
    def test_prepare_from_splits_builds_expected_files(self) -> None:
        preparer = DatasetPreparer(seed=11, holdout_fraction=0.5)
        splits = {
            "allnli": {
                "train": [
                    {"premise": "A man eats an apple", "hypothesis": "A person eats fruit", "label": 0},
                    {"premise": "A man eats an apple", "hypothesis": "Nobody is eating", "label": 2},
                ],
                "validation": [],
                "test": [],
            },
            "stsb": {
                "train": [
                    {"sentence1": "semantic retrieval", "sentence2": "semantic search", "label": 4.8},
                    {"sentence1": "semantic retrieval", "sentence2": "banana smoothie", "label": 1.0},
                ],
                "validation": [
                    {"sentence1": "semantic retrieval", "sentence2": "semantic search", "label": 4.9},
                ],
                "test": [
                    {"sentence1": "apple fruit", "sentence2": "fresh apple", "label": 4.7},
                ],
            },
            "qqp": {
                "train": [
                    {"question1": "How to learn Python?", "question2": "What is the best way to learn Python?", "label": 1},
                    {"question1": "How to learn Python?", "question2": "Why is the sky blue?", "label": 0},
                ],
                "validation": [
                    {"question1": "How to learn Python?", "question2": "What is the best way to learn Python?", "label": 1},
                ],
                "test": [],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = preparer.prepare_from_splits(tmp_dir, splits)
            self.assertGreater(summary["corpus_size"], 0)
            self.assertGreater(summary["train_triplets"], 0)
            self.assertGreater(summary["val_queries"], 0)
            self.assertGreater(summary["test_queries"], 0)
            self.assertEqual(summary["split_strategy"]["type"], "holdout_eval_pool")

            corpus_rows = load_jsonl(Path(tmp_dir) / "corpus.jsonl")
            triplets = load_jsonl(Path(tmp_dir) / "train_triplets.jsonl")
            val_queries = load_jsonl(Path(tmp_dir) / "val_queries.jsonl")
            test_queries = load_jsonl(Path(tmp_dir) / "test_queries.jsonl")

            self.assertTrue(any(row["text"] == "semantic search" for row in corpus_rows))
            self.assertTrue(any(row["query"] == "semantic retrieval" for row in triplets))
            self.assertFalse({row["query"] for row in val_queries} & {row["query"] for row in test_queries})
            self.assertEqual(sorted(row["query"] for row in val_queries + test_queries), sorted({row["query"] for row in val_queries + test_queries}))


if __name__ == "__main__":
    unittest.main()
