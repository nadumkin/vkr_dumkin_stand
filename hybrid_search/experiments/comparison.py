from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from ..core.preprocessing import TextPreprocessor
from ..io.data import load_jsonl, load_text_records
from ..io.storage import ensure_directory, save_json
from ..models.encoders import build_encoder
from ..retrieval.dense_search import DenseSearchPipeline
from ..retrieval.evaluation import SearchEvaluator
from .workflows import build_index_from_corpus, evaluate_index, parse_hidden_dims, train_hash_module


@dataclass
class ComparisonArtifacts:
    run_dir: Path
    comparison_path: Path
    payload: dict[str, Any]


def compare_with_baselines(
    *,
    prepared_data_dir: str | Path,
    run_dir: str | Path,
    encoder_config: EncoderConfig,
    model_config: HashingModelConfig,
    training_config: TrainingConfig,
    index_config: IndexConfig,
    search_config: SearchConfig,
) -> ComparisonArtifacts:
    data_dir = Path(prepared_data_dir)
    run_root = ensure_directory(run_dir)
    current_dir = ensure_directory(run_root / "hybrid_trained")
    untrained_dir = ensure_directory(run_root / "hybrid_untrained")
    dense_dir = ensure_directory(run_root / "dense_exact")
    comparison_path = run_root / "comparison.json"

    train_artifacts = train_hash_module(
        triplets_path=data_dir / "train_triplets.jsonl",
        checkpoint_path=current_dir / "hash_model.pt",
        encoder_config=encoder_config,
        model_config=model_config,
        training_config=training_config,
    )
    current_index = build_index_from_corpus(
        corpus_path=data_dir / "corpus.jsonl",
        output_dir=current_dir / "index",
        encoder_config=encoder_config,
        model_config=model_config,
        index_config=index_config,
        checkpoint_path=train_artifacts.checkpoint,
        device=encoder_config.device,
    )
    current_val = evaluate_index(
        index_dir=current_index.index_dir,
        queries_path=data_dir / "val_queries.jsonl",
        search_config=search_config,
        device=encoder_config.device,
    )
    current_test = evaluate_index(
        index_dir=current_index.index_dir,
        queries_path=data_dir / "test_queries.jsonl",
        search_config=search_config,
        device=encoder_config.device,
    )

    untrained_index = build_index_from_corpus(
        corpus_path=data_dir / "corpus.jsonl",
        output_dir=untrained_dir / "index",
        encoder_config=encoder_config,
        model_config=model_config,
        index_config=index_config,
        checkpoint_path=None,
        device=encoder_config.device,
    )
    untrained_val = evaluate_index(
        index_dir=untrained_index.index_dir,
        queries_path=data_dir / "val_queries.jsonl",
        search_config=search_config,
        device=encoder_config.device,
    )
    untrained_test = evaluate_index(
        index_dir=untrained_index.index_dir,
        queries_path=data_dir / "test_queries.jsonl",
        search_config=search_config,
        device=encoder_config.device,
    )

    dense_preprocessor = TextPreprocessor()
    dense_encoder = build_encoder(encoder_config, preprocessor=dense_preprocessor)
    dense_pipeline = DenseSearchPipeline(preprocessor=dense_preprocessor, encoder=dense_encoder)
    dense_pipeline.index_documents(load_text_records(data_dir / "corpus.jsonl"))
    dense_evaluator = SearchEvaluator(dense_pipeline)
    dense_val = dense_evaluator.evaluate(load_jsonl(data_dir / "val_queries.jsonl"), search_config=search_config)
    dense_test = dense_evaluator.evaluate(load_jsonl(data_dir / "test_queries.jsonl"), search_config=search_config)

    save_json(dense_dir / "validation_metrics.json", dense_val)
    save_json(dense_dir / "test_metrics.json", dense_test)
    save_json(current_dir / "validation_metrics.json", current_val)
    save_json(current_dir / "test_metrics.json", current_test)
    save_json(untrained_dir / "validation_metrics.json", untrained_val)
    save_json(untrained_dir / "test_metrics.json", untrained_test)

    payload = {
        "prepared_data_dir": str(data_dir),
        "run_dir": str(run_root),
        "configs": {
            "encoder": asdict(encoder_config),
            "hashing_requested": model_config.to_dict(),
            "hashing_effective": train_artifacts.model_config.to_dict(),
            "training": asdict(training_config),
            "index": asdict(index_config),
            "search": asdict(search_config),
        },
        "models": {
            "dense_exact": {
                "validation": dense_val,
                "test": dense_test,
                "build": {
                    "encode_time_ms": dense_pipeline.last_encode_time_ms,
                    "hashing_time_ms": 0.0,
                    "build_time_ms": 0.0,
                    "memory": dense_pipeline.memory_breakdown(),
                },
            },
            "hybrid_untrained": {
                "validation": untrained_val,
                "test": untrained_test,
                "build": {
                    "encode_time_ms": untrained_index.encode_time_ms,
                    "hashing_time_ms": untrained_index.hashing_time_ms,
                    "build_time_ms": untrained_index.build_time_ms,
                    "memory": untrained_index.memory,
                    "index_backend_active": untrained_index.index_backend_active,
                },
            },
            "hybrid_trained": {
                "validation": current_val,
                "test": current_test,
                "training_history": train_artifacts.history,
                "build": {
                    "encode_time_ms": current_index.encode_time_ms,
                    "hashing_time_ms": current_index.hashing_time_ms,
                    "build_time_ms": current_index.build_time_ms,
                    "memory": current_index.memory,
                    "index_backend_active": current_index.index_backend_active,
                },
            },
        },
        "deltas": {
            "trained_minus_untrained": _metric_deltas(current_test, untrained_test),
            "trained_minus_dense_exact": _metric_deltas(current_test, dense_test),
        },
    }
    save_json(comparison_path, payload)
    return ComparisonArtifacts(run_dir=run_root, comparison_path=comparison_path, payload=payload)


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    shared_keys = sorted(set(left) & set(right))
    deltas: dict[str, float] = {}
    for key in shared_keys:
        if key == "queries":
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            deltas[key] = float(left_value) - float(right_value)
    return deltas


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the current hybrid implementation with baseline pipelines.")
    parser.add_argument("--prepared-data-dir", required=True)
    parser.add_argument("--run-dir", required=True)

    parser.add_argument("--encoder-backend", default="hashing", choices=["hashing", "transformers"])
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    encoder_config = EncoderConfig(
        backend=args.encoder_backend,
        model_name=args.transformer_model,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        device=args.device,
        seed=args.seed,
    )
    model_config = HashingModelConfig(
        code_bits=args.code_bits,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        quantization_weight=args.quantization_weight,
        device=args.device,
    )
    index_config = IndexConfig(
        code_bits=args.code_bits,
        backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    search_config = SearchConfig(
        strategy=args.strategy,
        top_k=args.top_k,
        oversample_factor=args.oversample_factor,
        max_hamming_distance=args.max_hamming_distance,
    )

    artifacts = compare_with_baselines(
        prepared_data_dir=args.prepared_data_dir,
        run_dir=args.run_dir,
        encoder_config=encoder_config,
        model_config=model_config,
        training_config=training_config,
        index_config=index_config,
        search_config=search_config,
    )
    print(json.dumps(artifacts.payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
