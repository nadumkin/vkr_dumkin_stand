from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from ..io.storage import ensure_directory, load_json, save_json
from .dataset_preparation import DatasetPreparer
from .experiment_logging import ExperimentLogger, create_experiment_logger
from .workflows import IndexArtifacts, TrainingArtifacts, build_index_from_corpus, evaluate_index, parse_hidden_dims, train_hash_module


@dataclass
class DatasetPreparationSettings:
    dataset_names: tuple[str, ...] = ("allnli", "stsb", "qqp")
    cache_dir: str | None = None
    limit_train_rows: int | None = None
    limit_eval_rows: int | None = None
    stsb_positive_threshold: float = 4.0
    stsb_negative_threshold: float = 2.0
    holdout_fraction: float = 0.5
    seed: int = 17


@dataclass
class ExperimentTrackingSettings:
    backend: str = "none"
    experiment_name: str = "hybrid-search"
    run_name: str | None = None
    tracking_uri: str | None = None


class ExperimentRunner:
    def __init__(self, logger: ExperimentLogger | None = None) -> None:
        self.logger = logger or create_experiment_logger("none")

    def run(
        self,
        run_dir: str | Path,
        *,
        dataset_settings: DatasetPreparationSettings,
        encoder_config: EncoderConfig,
        model_config: HashingModelConfig,
        training_config: TrainingConfig,
        index_config: IndexConfig,
        search_config: SearchConfig,
        prepared_data_dir: str | Path | None = None,
        datasets_by_name: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]] | None = None,
    ) -> dict[str, Any]:
        run_root = ensure_directory(run_dir)
        artifacts_dir = ensure_directory(run_root / "artifacts")
        metrics_dir = ensure_directory(run_root / "metrics")
        summary_path = run_root / "summary.json"
        training_history_path = metrics_dir / "training_history.json"
        validation_metrics_path = metrics_dir / "validation_metrics.json"
        test_metrics_path = metrics_dir / "test_metrics.json"
        checkpoint_path = artifacts_dir / "hash_model.pt"
        index_dir = run_root / "index"

        self.logger.start_run()
        self.logger.set_tags(
            {
                "component": "hybrid-search-experiment",
                "encoder_backend": encoder_config.backend,
                "index_backend": index_config.backend,
            }
        )
        self.logger.log_params(
            {
                "dataset": asdict(dataset_settings),
                "encoder": asdict(encoder_config),
                "hashing": model_config.to_dict(),
                "training": asdict(training_config),
                "index": asdict(index_config),
                "search": asdict(search_config),
                "run_dir": str(run_root),
            }
        )

        try:
            data_dir, dataset_summary = self._prepare_data(
                run_root=run_root,
                dataset_settings=dataset_settings,
                prepared_data_dir=prepared_data_dir,
                datasets_by_name=datasets_by_name,
            )

            training = train_hash_module(
                triplets_path=data_dir / "train_triplets.jsonl",
                checkpoint_path=checkpoint_path,
                encoder_config=encoder_config,
                model_config=model_config,
                training_config=training_config,
            )
            save_json(training_history_path, training.history)
            for epoch_metrics in training.history:
                step = int(epoch_metrics.get("epoch", 0))
                self.logger.log_metrics(
                    {
                        "train.loss": float(epoch_metrics["loss"]),
                        "train.triplet_loss": float(epoch_metrics["triplet_loss"]),
                        "train.quantization_loss": float(epoch_metrics["quantization_loss"]),
                    },
                    step=step,
                )

            index = build_index_from_corpus(
                corpus_path=data_dir / "corpus.jsonl",
                output_dir=index_dir,
                encoder_config=encoder_config,
                model_config=model_config,
                index_config=index_config,
                checkpoint_path=training.checkpoint,
                device=encoder_config.device,
            )

            validation_metrics = self._evaluate_split(
                index_dir=index.index_dir,
                queries_path=data_dir / "val_queries.jsonl",
                metrics_path=validation_metrics_path,
                search_config=search_config,
                device=encoder_config.device,
            )
            test_metrics = self._evaluate_split(
                index_dir=index.index_dir,
                queries_path=data_dir / "test_queries.jsonl",
                metrics_path=test_metrics_path,
                search_config=search_config,
                device=encoder_config.device,
            )

            self.logger.log_metrics({f"validation.{key}": float(value) for key, value in validation_metrics.items() if key != "queries"})
            self.logger.log_metrics({f"test.{key}": float(value) for key, value in test_metrics.items() if key != "queries"})

            summary = self._build_summary(
                run_root=run_root,
                data_dir=data_dir,
                dataset_summary=dataset_summary,
                training=training,
                index=index,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                dataset_settings=dataset_settings,
                encoder_config=encoder_config,
                model_config=model_config,
                training_config=training_config,
                index_config=index_config,
                search_config=search_config,
            )
            save_json(summary_path, summary)

            for artifact_path, artifact_group in (
                (summary_path, "summary"),
                (training_history_path, "metrics"),
                (validation_metrics_path, "metrics"),
                (test_metrics_path, "metrics"),
            ):
                self.logger.log_artifact(artifact_path, artifact_path=artifact_group)

            return summary
        finally:
            self.logger.end_run()

    def _prepare_data(
        self,
        *,
        run_root: Path,
        dataset_settings: DatasetPreparationSettings,
        prepared_data_dir: str | Path | None,
        datasets_by_name: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]] | None,
    ) -> tuple[Path, dict[str, Any]]:
        if prepared_data_dir is not None:
            data_dir = Path(prepared_data_dir)
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                dataset_summary = load_json(summary_path)
            else:
                dataset_summary = {
                    "files": {
                        "corpus": str(data_dir / "corpus.jsonl"),
                        "train_triplets": str(data_dir / "train_triplets.jsonl"),
                        "val_queries": str(data_dir / "val_queries.jsonl"),
                        "test_queries": str(data_dir / "test_queries.jsonl"),
                    }
                }
            return data_dir, dataset_summary

        data_dir = ensure_directory(run_root / "data")
        preparer = DatasetPreparer(
            stsb_positive_threshold=dataset_settings.stsb_positive_threshold,
            stsb_negative_threshold=dataset_settings.stsb_negative_threshold,
            holdout_fraction=dataset_settings.holdout_fraction,
            seed=dataset_settings.seed,
        )
        if datasets_by_name is not None:
            dataset_summary = preparer.prepare_from_splits(output_dir=data_dir, datasets_by_name=datasets_by_name)
        else:
            dataset_summary = preparer.prepare_from_hf(
                output_dir=data_dir,
                dataset_names=dataset_settings.dataset_names,
                cache_dir=dataset_settings.cache_dir,
                limit_train_rows=dataset_settings.limit_train_rows,
                limit_eval_rows=dataset_settings.limit_eval_rows,
            )
        self.logger.log_params({"prepared_data_dir": str(data_dir)})
        return data_dir, dataset_summary

    def _evaluate_split(
        self,
        *,
        index_dir: Path,
        queries_path: Path,
        metrics_path: Path,
        search_config: SearchConfig,
        device: str,
    ) -> dict[str, Any]:
        metrics = evaluate_index(
            index_dir=index_dir,
            queries_path=queries_path,
            search_config=search_config,
            device=device,
        )
        save_json(metrics_path, metrics)
        return metrics

    def _build_summary(
        self,
        *,
        run_root: Path,
        data_dir: Path,
        dataset_summary: dict[str, Any],
        training: TrainingArtifacts,
        index: IndexArtifacts,
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        dataset_settings: DatasetPreparationSettings,
        encoder_config: EncoderConfig,
        model_config: HashingModelConfig,
        training_config: TrainingConfig,
        index_config: IndexConfig,
        search_config: SearchConfig,
    ) -> dict[str, Any]:
        return {
            "run_dir": str(run_root),
            "data_dir": str(data_dir),
            "dataset_summary": dataset_summary,
            "training": training.to_dict(),
            "index": index.to_dict(),
            "evaluation": {
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "configs": {
                "dataset": asdict(dataset_settings),
                "encoder": asdict(encoder_config),
                "hashing_requested": model_config.to_dict(),
                "hashing_effective": training.model_config.to_dict(),
                "training": asdict(training_config),
                "index": asdict(index_config),
                "search": asdict(search_config),
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full hybrid-search experiment pipeline.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prepared-data-dir")
    parser.add_argument("--datasets", default="allnli,stsb,qqp")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-eval-rows", type=int)
    parser.add_argument("--stsb-positive-threshold", type=float, default=4.0)
    parser.add_argument("--stsb-negative-threshold", type=float, default=2.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--encoder-backend", default="hashing", choices=["hashing", "transformers"])
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--code-bits", type=int, default=128)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-layer-norm", action="store_true")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--quantization-weight", type=float, default=0.1)

    parser.add_argument("--index-backend", default="numpy", choices=["numpy", "hnswlib"])
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument("--strategy", default="coarse_rerank", choices=["coarse_rerank", "integrated_filtering"])
    parser.add_argument("--max-hamming-distance", type=int)

    parser.add_argument("--logger-backend", default="none", choices=["none", "mlflow"])
    parser.add_argument("--mlflow-experiment-name", default="hybrid-search")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--mlflow-tracking-uri")
    return parser


def _dataset_settings_from_args(args: argparse.Namespace) -> DatasetPreparationSettings:
    return DatasetPreparationSettings(
        dataset_names=tuple(item.strip() for item in args.datasets.split(",") if item.strip()),
        cache_dir=args.cache_dir,
        limit_train_rows=args.limit_train_rows,
        limit_eval_rows=args.limit_eval_rows,
        stsb_positive_threshold=args.stsb_positive_threshold,
        stsb_negative_threshold=args.stsb_negative_threshold,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )


def _encoder_config_from_args(args: argparse.Namespace) -> EncoderConfig:
    return EncoderConfig(
        backend=args.encoder_backend,
        model_name=args.transformer_model,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        device=args.device,
        seed=args.seed,
    )


def _hash_model_config_from_args(args: argparse.Namespace) -> HashingModelConfig:
    return HashingModelConfig(
        code_bits=args.code_bits,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        quantization_weight=args.quantization_weight,
        device=args.device,
    )


def _index_config_from_args(args: argparse.Namespace) -> IndexConfig:
    return IndexConfig(
        code_bits=args.code_bits,
        backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )


def _search_config_from_args(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        strategy=args.strategy,
        top_k=args.top_k,
        oversample_factor=args.oversample_factor,
        max_hamming_distance=args.max_hamming_distance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tracking_settings = ExperimentTrackingSettings(
        backend=args.logger_backend,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name,
        tracking_uri=args.mlflow_tracking_uri,
    )
    runner = ExperimentRunner(
        logger=create_experiment_logger(
            tracking_settings.backend,
            experiment_name=tracking_settings.experiment_name,
            run_name=tracking_settings.run_name,
            tracking_uri=tracking_settings.tracking_uri,
        )
    )
    summary = runner.run(
        run_dir=args.run_dir,
        prepared_data_dir=args.prepared_data_dir,
        dataset_settings=_dataset_settings_from_args(args),
        encoder_config=_encoder_config_from_args(args),
        model_config=_hash_model_config_from_args(args),
        training_config=_training_config_from_args(args),
        index_config=_index_config_from_args(args),
        search_config=_search_config_from_args(args),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
