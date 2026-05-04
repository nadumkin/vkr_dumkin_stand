import tempfile
import unittest
from pathlib import Path

from hybrid_search.core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from hybrid_search.experiments.comparison import compare_with_baselines
from hybrid_search.io.storage import load_json


class BaselineComparisonTest(unittest.TestCase):
    def test_compare_with_baselines_writes_comparison_report(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        prepared_data_dir = project_root / "data" / "mini"

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = compare_with_baselines(
                prepared_data_dir=prepared_data_dir,
                run_dir=Path(tmp_dir) / "comparison",
                encoder_config=EncoderConfig(backend="hashing", embedding_dim=64, batch_size=8, device="cpu", seed=23),
                model_config=HashingModelConfig(code_bits=32, hidden_dims=(48, 32), dropout=0.0),
                training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=5e-3, device="cpu"),
                index_config=IndexConfig(code_bits=32, backend="numpy"),
                search_config=SearchConfig(top_k=3, oversample_factor=3),
            )

            self.assertTrue(artifacts.comparison_path.exists())
            stored = load_json(artifacts.comparison_path)
            self.assertIn("dense_exact", stored["models"])
            self.assertIn("hybrid_untrained", stored["models"])
            self.assertIn("hybrid_trained", stored["models"])
            self.assertGreater(stored["models"]["hybrid_trained"]["validation"]["queries"], 0)
            self.assertGreater(stored["models"]["dense_exact"]["test"]["queries"], 0)
            self.assertIn("trained_minus_untrained", stored["deltas"])


if __name__ == "__main__":
    unittest.main()
