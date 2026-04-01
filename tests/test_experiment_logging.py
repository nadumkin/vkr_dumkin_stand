from __future__ import annotations

import unittest
from unittest.mock import patch

from hybrid_search.experiments.experiment_logging import create_experiment_logger


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.run_name = None
        self.logged_params = None
        self.logged_metrics = []
        self.tags = None
        self.artifacts = []
        self.ended = False

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, run_name: str | None = None) -> str:
        self.run_name = run_name
        return "fake-run"

    def log_params(self, payload: dict[str, str]) -> None:
        self.logged_params = payload

    def log_metrics(self, payload: dict[str, float], step: int | None = None) -> None:
        self.logged_metrics.append((payload, step))

    def set_tags(self, payload: dict[str, str]) -> None:
        self.tags = payload

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def end_run(self) -> None:
        self.ended = True


class ExperimentLoggingTest(unittest.TestCase):
    def test_mlflow_logger_uses_expected_client_calls(self) -> None:
        fake_mlflow = _FakeMlflow()
        with patch("importlib.import_module", return_value=fake_mlflow):
            logger = create_experiment_logger(
                "mlflow",
                experiment_name="hybrid-search-tests",
                run_name="runner-smoke",
                tracking_uri="file:///tmp/mlruns",
            )
            logger.start_run()
            logger.log_params({"encoder": {"backend": "hashing"}, "code_bits": 32})
            logger.log_metrics({"recall@10": 0.9}, step=2)
            logger.set_tags({"stage": "test"})
            logger.log_artifact("/tmp/example.json", artifact_path="metrics")
            logger.end_run()

        self.assertEqual(fake_mlflow.tracking_uri, "file:///tmp/mlruns")
        self.assertEqual(fake_mlflow.experiment_name, "hybrid-search-tests")
        self.assertEqual(fake_mlflow.run_name, "runner-smoke")
        self.assertEqual(fake_mlflow.logged_params["encoder.backend"], "hashing")
        self.assertEqual(fake_mlflow.logged_params["code_bits"], "32")
        self.assertEqual(fake_mlflow.logged_metrics, [({"recall@10": 0.9}, 2)])
        self.assertEqual(fake_mlflow.tags, {"stage": "test"})
        self.assertEqual(fake_mlflow.artifacts, [("/tmp/example.json", "metrics")])
        self.assertTrue(fake_mlflow.ended)


if __name__ == "__main__":
    unittest.main()
