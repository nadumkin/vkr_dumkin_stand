from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        compound_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=compound_key))
        elif isinstance(value, (list, tuple, set)):
            flattened[compound_key] = json.dumps(list(value), ensure_ascii=False)
        else:
            flattened[compound_key] = str(value)
    return flattened


class ExperimentLogger:
    def start_run(self) -> None:
        return None

    def log_params(self, payload: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        return None

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        return None

    def end_run(self) -> None:
        return None


class NullExperimentLogger(ExperimentLogger):
    pass


class MlflowExperimentLogger(ExperimentLogger):
    def __init__(
        self,
        experiment_name: str = "hybrid-search",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        try:
            self.mlflow = importlib.import_module("mlflow")
        except ImportError as exc:
            raise ImportError(
                "MLflow is not installed. Install the optional dependency set, for example: pip install -e .[mlflow]."
            ) from exc
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._run = None

    def start_run(self) -> None:
        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)
        self._run = self.mlflow.start_run(run_name=self.run_name)

    def log_params(self, payload: Mapping[str, Any]) -> None:
        flattened = _flatten_mapping(payload)
        if flattened:
            self.mlflow.log_params(flattened)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        numeric_metrics = {key: float(value) for key, value in metrics.items()}
        if numeric_metrics:
            self.mlflow.log_metrics(numeric_metrics, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if tags:
            self.mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end_run(self) -> None:
        if self._run is not None:
            self.mlflow.end_run()
            self._run = None


def create_experiment_logger(
    backend: str = "none",
    *,
    experiment_name: str = "hybrid-search",
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> ExperimentLogger:
    backend_key = backend.lower().strip()
    if backend_key in {"", "none"}:
        return NullExperimentLogger()
    if backend_key == "mlflow":
        return MlflowExperimentLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
        )
    raise ValueError(f"Unsupported experiment logger backend: {backend}")
