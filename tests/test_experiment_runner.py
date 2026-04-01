import tempfile
import unittest
from pathlib import Path

from hybrid_search.core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from hybrid_search.experiments.experiment_runner import DatasetPreparationSettings, ExperimentRunner
from hybrid_search.io.storage import load_json


class ExperimentRunnerTest(unittest.TestCase):
    def test_runner_executes_full_pipeline_on_prepared_dataset(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        prepared_data_dir = project_root / "data" / "mini"

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            runner = ExperimentRunner()
            summary = runner.run(
                run_dir=run_dir,
                prepared_data_dir=prepared_data_dir,
                dataset_settings=DatasetPreparationSettings(seed=23),
                encoder_config=EncoderConfig(backend="hashing", embedding_dim=64, batch_size=8, device="cpu", seed=23),
                model_config=HashingModelConfig(code_bits=32, hidden_dims=(48, 32), dropout=0.0),
                training_config=TrainingConfig(epochs=2, batch_size=4, learning_rate=5e-3, device="cpu"),
                index_config=IndexConfig(code_bits=32, backend="numpy"),
                search_config=SearchConfig(top_k=3, oversample_factor=3),
            )

            self.assertEqual(summary["training"]["history"][-1]["epoch"], 2)
            self.assertGreater(summary["evaluation"]["validation"]["queries"], 0)
            self.assertGreater(summary["evaluation"]["test"]["queries"], 0)
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "training_history.json").exists())
            self.assertTrue((run_dir / "metrics" / "validation_metrics.json").exists())
            self.assertTrue((run_dir / "metrics" / "test_metrics.json").exists())
            self.assertTrue((run_dir / "index" / "encoder_config.json").exists())

            stored_summary = load_json(run_dir / "summary.json")
            self.assertEqual(stored_summary["evaluation"]["validation"]["queries"], summary["evaluation"]["validation"]["queries"])


if __name__ == "__main__":
    unittest.main()
