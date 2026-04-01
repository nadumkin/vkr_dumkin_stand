from __future__ import annotations

import argparse
import json

from .core.config import EncoderConfig, HashingModelConfig, IndexConfig, SearchConfig, TrainingConfig
from .experiments.dataset_preparation import DatasetPreparer
from .experiments.workflows import build_index_from_corpus, evaluate_index, load_pipeline_from_index, parse_hidden_dims, train_hash_module


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


def prepare_datasets_command(args: argparse.Namespace) -> int:
    preparer = DatasetPreparer(
        stsb_positive_threshold=args.stsb_positive_threshold,
        stsb_negative_threshold=args.stsb_negative_threshold,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    summary = preparer.prepare_from_hf(
        output_dir=args.output_dir,
        dataset_names=[item.strip() for item in args.datasets.split(",") if item.strip()],
        cache_dir=args.cache_dir,
        limit_train_rows=args.limit_train_rows,
        limit_eval_rows=args.limit_eval_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def train_hash_command(args: argparse.Namespace) -> int:
    encoder_config = _encoder_config_from_args(args)
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
    artifacts = train_hash_module(
        triplets_path=args.triplets,
        checkpoint_path=args.checkpoint,
        encoder_config=encoder_config,
        model_config=model_config,
        training_config=training_config,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(artifacts.checkpoint),
                "history": artifacts.history[-1] if artifacts.history else {},
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_index_command(args: argparse.Namespace) -> int:
    encoder_config = _encoder_config_from_args(args)
    index_config = IndexConfig(
        code_bits=args.code_bits,
        backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    model_config = HashingModelConfig(
        code_bits=args.code_bits,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )
    artifacts = build_index_from_corpus(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        encoder_config=encoder_config,
        model_config=model_config,
        index_config=index_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "index_dir": str(artifacts.index_dir),
                "documents": artifacts.documents,
                "index_backend_requested": artifacts.index_backend_requested,
                "index_backend_active": artifacts.index_backend_active,
            },
            ensure_ascii=False,
        )
    )
    return 0


def search_command(args: argparse.Namespace) -> int:
    pipeline = load_pipeline_from_index(args.index_dir, device=args.device)
    results = pipeline.search(
        args.query,
        config=SearchConfig(
            strategy=args.strategy,
            top_k=args.top_k,
            oversample_factor=args.oversample_factor,
            max_hamming_distance=args.max_hamming_distance,
        ),
    )
    print(
        json.dumps(
            [
                {
                    "id": result.record_id,
                    "dense_score": result.dense_score,
                    "binary_distance": result.binary_distance,
                    "rank": result.rank,
                    "text": result.text,
                    "metadata": result.metadata,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    metrics = evaluate_index(
        index_dir=args.index_dir,
        queries_path=args.queries,
        search_config=SearchConfig(
            strategy=args.strategy,
            top_k=args.top_k,
            oversample_factor=args.oversample_factor,
            max_hamming_distance=args.max_hamming_distance,
        ),
        device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid semantic search prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_encoder = {
        "encoder_backend": ("--encoder-backend", {"default": "hashing", "choices": ["hashing", "transformers"]}),
        "transformer_model": ("--transformer-model", {"default": "sentence-transformers/all-MiniLM-L6-v2"}),
        "embedding_dim": ("--embedding-dim", {"type": int, "default": 384}),
        "batch_size": ("--batch-size", {"type": int, "default": 64}),
        "device": ("--device", {"default": "cpu"}),
        "seed": ("--seed", {"type": int, "default": 17}),
    }

    common_hash = {
        "code_bits": ("--code-bits", {"type": int, "default": 128}),
        "hidden_dims": ("--hidden-dims", {"default": "256,128"}),
        "activation": ("--activation", {"default": "gelu"}),
        "dropout": ("--dropout", {"type": float, "default": 0.1}),
        "no_layer_norm": ("--no-layer-norm", {"action": "store_true"}),
    }
    common_index = {
        "index_backend": ("--index-backend", {"default": "numpy", "choices": ["numpy", "hnswlib"]}),
        "hnsw_m": ("--hnsw-m", {"type": int, "default": 16}),
        "hnsw_ef_construction": ("--hnsw-ef-construction", {"type": int, "default": 200}),
        "hnsw_ef_search": ("--hnsw-ef-search", {"type": int, "default": 64}),
    }

    train_parser = subparsers.add_parser("train-hash", help="Train the binary hashing module.")
    train_parser.add_argument("--triplets", required=True)
    train_parser.add_argument("--checkpoint", required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--margin", type=float, default=4.0)
    train_parser.add_argument("--quantization-weight", type=float, default=0.1)
    for _, (flag, kwargs) in {**common_encoder, **common_hash}.items():
        train_parser.add_argument(flag, **kwargs)
    train_parser.set_defaults(func=train_hash_command)

    prepare_parser = subparsers.add_parser("prepare-datasets", help="Download and prepare AllNLI/STS-B/QQP datasets.")
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--datasets", default="allnli,stsb,qqp")
    prepare_parser.add_argument("--cache-dir")
    prepare_parser.add_argument("--limit-train-rows", type=int)
    prepare_parser.add_argument("--limit-eval-rows", type=int)
    prepare_parser.add_argument("--stsb-positive-threshold", type=float, default=4.0)
    prepare_parser.add_argument("--stsb-negative-threshold", type=float, default=2.0)
    prepare_parser.add_argument("--holdout-fraction", type=float, default=0.5)
    prepare_parser.add_argument("--seed", type=int, default=17)
    prepare_parser.set_defaults(func=prepare_datasets_command)

    build_parser_cmd = subparsers.add_parser("build-index", help="Build an index over the corpus.")
    build_parser_cmd.add_argument("--corpus", required=True)
    build_parser_cmd.add_argument("--output-dir", required=True)
    build_parser_cmd.add_argument("--checkpoint")
    for _, (flag, kwargs) in {**common_encoder, **common_hash, **common_index}.items():
        build_parser_cmd.add_argument(flag, **kwargs)
    build_parser_cmd.set_defaults(func=build_index_command)

    search_parser = subparsers.add_parser("search", help="Run a search query.")
    search_parser.add_argument("--index-dir", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--oversample-factor", type=int, default=4)
    search_parser.add_argument("--strategy", default="coarse_rerank", choices=["coarse_rerank", "integrated_filtering"])
    search_parser.add_argument("--max-hamming-distance", type=int)
    search_parser.add_argument("--device", default="cpu")
    search_parser.set_defaults(func=search_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the search pipeline.")
    evaluate_parser.add_argument("--index-dir", required=True)
    evaluate_parser.add_argument("--queries", required=True)
    evaluate_parser.add_argument("--top-k", type=int, default=10)
    evaluate_parser.add_argument("--oversample-factor", type=int, default=4)
    evaluate_parser.add_argument("--strategy", default="coarse_rerank", choices=["coarse_rerank", "integrated_filtering"])
    evaluate_parser.add_argument("--max-hamming-distance", type=int)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.set_defaults(func=evaluate_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
