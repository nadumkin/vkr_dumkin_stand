"""Серия экспериментов для блоков 1, 2 и 3 плана §4.3 ВКР.

Блок 1: длина бинарного кода (64, 128, 256 бит) — полные сравнения.
Блок 2: чувствительность к efSearch HNSW (50, 100, 150, 200) при code_bits=128.
Блок 3: коэффициент oversampling (2, 5, 10, 20) при code_bits=128.

Блоки 2 и 3 переиспользуют один и тот же обученный хэш-модуль и индекс
(оба меняют только параметры поиска), что экономит около 40% времени.

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/run_experiment_sweep.py --blocks 1,2,3

Аргументы:
    --blocks       подмножество "1,2,3" (по умолчанию все)
    --data-dir     каталог prepared data (по умолчанию data/real_subset)
    --device       cpu | mps | cuda (auto-detect, если не задан)
    --epochs       число эпох обучения (по умолчанию 10)
    --output-root  корень для артефактов (по умолчанию artifacts/sweep)

Длительность: ~25–40 мин на M-чипе для всех трёх блоков.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Настройка кэшей и режима MPS до импорта torch
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from hybrid_search.core.config import (
    EncoderConfig,
    HashingModelConfig,
    IndexConfig,
    SearchConfig,
    TrainingConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.experiments.comparison import compare_with_baselines
from hybrid_search.experiments.workflows import (
    build_index_from_corpus,
    evaluate_index,
    train_hash_module,
)
from hybrid_search.io.storage import save_json


def auto_detect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encoder_cfg(device: str) -> EncoderConfig:
    return EncoderConfig(
        backend="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        batch_size=128,
        normalize_embeddings=True,
        device=device,
        seed=17,
    )


def training_cfg(device: str, epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        margin=4.0,
        quantization_weight=0.1,
        gradient_clip_norm=1.0,
        temperature=1.0,
        device=device,
    )


def hashing_cfg(code_bits: int) -> HashingModelConfig:
    return HashingModelConfig(
        input_dim=384,
        code_bits=code_bits,
        hidden_dims=(256, 128),
        activation="gelu",
        dropout=0.1,
        use_layer_norm=True,
    )


def index_cfg(code_bits: int, backend: str = "hnswlib", ef_search: int = 64) -> IndexConfig:
    return IndexConfig(
        code_bits=code_bits,
        oversample_factor=4,
        binary_keep=code_bits,
        backend=backend,
        dense_metric="cosine",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=ef_search,
    )


def search_cfg(top_k: int = 10, oversample: int = 20) -> SearchConfig:
    return SearchConfig(
        strategy="coarse_rerank",
        top_k=top_k,
        oversample_factor=oversample,
        max_hamming_distance=None,
    )


# ---------------------------------------------------------------------------
# Block 1 — длина бинарного кода
# ---------------------------------------------------------------------------
def run_block1(args: argparse.Namespace, output_root: Path) -> list[dict]:
    print("\n" + "=" * 78)
    print("Блок 1: длина бинарного кода (64, 128, 256 бит)")
    print("=" * 78)

    block_root = output_root / "block1_code_bits"
    block_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for code_bits in (64, 128, 256):
        run_dir = block_root / f"cb{code_bits}"
        print(f"\n[block 1] code_bits={code_bits}  →  {run_dir}")
        started = time.perf_counter()
        artifacts = compare_with_baselines(
            prepared_data_dir=args.data_dir,
            run_dir=run_dir,
            encoder_config=encoder_cfg(args.device),
            model_config=hashing_cfg(code_bits),
            training_config=training_cfg(args.device, args.epochs),
            index_config=index_cfg(code_bits, backend="hnswlib"),
            search_config=search_cfg(top_k=10, oversample=20),
        )
        elapsed = time.perf_counter() - started
        for name, model in artifacts.payload["models"].items():
            t = model["test"]
            build = model.get("build", {})
            row = {
                "block": 1,
                "code_bits": code_bits,
                "model": name,
                "recall@10": t["recall@k"],
                "ndcg@10": t["ndcg@k"],
                "map@10": t["map@k"],
                "latency_ms": t["latency_ms"],
                "build_time_ms": build.get("build_time_ms", 0.0),
                "encode_time_ms": build.get("encode_time_ms", 0.0),
                "memory_total_bytes": (build.get("memory") or {}).get("total_bytes", 0),
                "elapsed_s": round(elapsed, 1),
                "artifact": str(run_dir / "comparison.json"),
            }
            summaries.append(row)
            print(
                f"    {name:18s}  recall={row['recall@10']:.4f}  ndcg={row['ndcg@10']:.4f}  "
                f"lat={row['latency_ms']:.2f}ms  mem={row['memory_total_bytes']/1e6:.1f}MB"
            )
    return summaries


# ---------------------------------------------------------------------------
# Block 2 — efSearch sweep (HNSW)
# ---------------------------------------------------------------------------
def run_block2(args: argparse.Namespace, output_root: Path) -> list[dict]:
    print("\n" + "=" * 78)
    print("Блок 2: efSearch HNSW (50, 100, 150, 200) при code_bits=128, M=32")
    print("=" * 78)

    block_root = output_root / "block2_efsearch"
    block_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    code_bits = 128
    data_dir = Path(args.data_dir)

    # Тренируем хэш-модуль один раз
    train_dir = block_root / "_shared_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    print("[block 2] обучаю хэш-модуль (один раз)")
    train_artifacts = train_hash_module(
        triplets_path=data_dir / "train_triplets.jsonl",
        checkpoint_path=train_dir / "hash_model.pt",
        encoder_config=encoder_cfg(args.device),
        model_config=hashing_cfg(code_bits),
        training_config=training_cfg(args.device, args.epochs),
    )

    # И строим индекс один раз — efSearch меняется на этапе поиска
    index_dir = block_root / "_shared_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    print("[block 2] строю HNSW-индекс (один раз)")
    index_artifacts = build_index_from_corpus(
        corpus_path=data_dir / "corpus.jsonl",
        output_dir=index_dir,
        encoder_config=encoder_cfg(args.device),
        model_config=hashing_cfg(code_bits),
        index_config=index_cfg(code_bits, backend="hnswlib", ef_search=64),
        checkpoint_path=train_artifacts.checkpoint,
        device=args.device,
    )
    print(
        f"    build_time={index_artifacts.build_time_ms:.1f}ms  "
        f"memory={index_artifacts.memory.get('total_bytes', 0)/1e6:.1f}MB"
    )

    for ef_search in (50, 100, 150, 200):
        # На стороне поиска эффективен оверайд через index._hnsw_index.set_ef(...)
        # Перезагружаем pipeline и переопределяем ef в самом hnsw-индексе.
        print(f"\n[block 2] efSearch={ef_search}")
        from hybrid_search.experiments.workflows import load_pipeline_from_index

        pipeline = load_pipeline_from_index(index_dir=index_dir, device=args.device)
        if pipeline.index._hnsw_index is not None:  # type: ignore[attr-defined]
            pipeline.index._hnsw_index.set_ef(ef_search)  # type: ignore[attr-defined]
        pipeline.index.hnsw_ef_search = ef_search

        from hybrid_search.io.data import load_jsonl
        from hybrid_search.retrieval.evaluation import SearchEvaluator

        evaluator = SearchEvaluator(pipeline)
        cfg = search_cfg(top_k=10, oversample=20)
        test = evaluator.evaluate(load_jsonl(data_dir / "test_queries.jsonl"), search_config=cfg, warmup_queries=5)
        save_json(block_root / f"ef{ef_search}_test.json", {
            "code_bits": code_bits,
            "ef_search": ef_search,
            "config": asdict(cfg),
            "test": test,
        })
        row = {
            "block": 2,
            "code_bits": code_bits,
            "ef_search": ef_search,
            "model": "hybrid_trained",
            "recall@10": test["recall@k"],
            "ndcg@10": test["ndcg@k"],
            "map@10": test["map@k"],
            "latency_ms": test["latency_ms"],
            "candidate_selection_ms": test["latency_breakdown_ms"]["candidate_selection_ms"],
            "rerank_ms": test["latency_breakdown_ms"]["rerank_ms"],
            "query_encode_ms": test["latency_breakdown_ms"]["query_encode_ms"],
            "artifact": str(block_root / f"ef{ef_search}_test.json"),
        }
        summaries.append(row)
        print(
            f"    recall={row['recall@10']:.4f}  ndcg={row['ndcg@10']:.4f}  "
            f"lat={row['latency_ms']:.2f}ms  cand={row['candidate_selection_ms']:.2f}ms"
        )
    return summaries


# ---------------------------------------------------------------------------
# Block 3 — oversample sweep
# ---------------------------------------------------------------------------
def run_block3(args: argparse.Namespace, output_root: Path) -> list[dict]:
    print("\n" + "=" * 78)
    print("Блок 3: oversample factor (2k, 5k, 10k, 20k) при code_bits=128, efSearch=64")
    print("=" * 78)

    block_root = output_root / "block3_oversample"
    block_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    code_bits = 128
    data_dir = Path(args.data_dir)

    # Если блок 2 уже собрал shared-индекс — используем его, иначе строим заново.
    shared_index = output_root / "block2_efsearch" / "_shared_index"
    shared_train = output_root / "block2_efsearch" / "_shared_train"
    if shared_index.exists() and (shared_train / "hash_model.pt").exists():
        print("[block 3] переиспользую индекс и модель из блока 2")
        index_dir = shared_index
    else:
        train_dir = block_root / "_shared_train"
        train_dir.mkdir(parents=True, exist_ok=True)
        print("[block 3] обучаю хэш-модуль (блок 2 не запускался)")
        train_artifacts = train_hash_module(
            triplets_path=data_dir / "train_triplets.jsonl",
            checkpoint_path=train_dir / "hash_model.pt",
            encoder_config=encoder_cfg(args.device),
            model_config=hashing_cfg(code_bits),
            training_config=training_cfg(args.device, args.epochs),
        )
        index_dir = block_root / "_shared_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        build_index_from_corpus(
            corpus_path=data_dir / "corpus.jsonl",
            output_dir=index_dir,
            encoder_config=encoder_cfg(args.device),
            model_config=hashing_cfg(code_bits),
            index_config=index_cfg(code_bits, backend="hnswlib"),
            checkpoint_path=train_artifacts.checkpoint,
            device=args.device,
        )

    for oversample in (2, 5, 10, 20):
        print(f"\n[block 3] oversample={oversample}")
        cfg = search_cfg(top_k=10, oversample=oversample)
        test = evaluate_index(
            index_dir=index_dir,
            queries_path=data_dir / "test_queries.jsonl",
            search_config=cfg,
            device=args.device,
        )
        save_json(block_root / f"ob{oversample}_test.json", {
            "code_bits": code_bits,
            "oversample": oversample,
            "config": asdict(cfg),
            "test": test,
        })
        row = {
            "block": 3,
            "code_bits": code_bits,
            "oversample": oversample,
            "model": "hybrid_trained",
            "recall@10": test["recall@k"],
            "ndcg@10": test["ndcg@k"],
            "map@10": test["map@k"],
            "latency_ms": test["latency_ms"],
            "candidate_selection_ms": test["latency_breakdown_ms"]["candidate_selection_ms"],
            "rerank_ms": test["latency_breakdown_ms"]["rerank_ms"],
            "artifact": str(block_root / f"ob{oversample}_test.json"),
        }
        summaries.append(row)
        print(
            f"    recall={row['recall@10']:.4f}  ndcg={row['ndcg@10']:.4f}  "
            f"lat={row['latency_ms']:.2f}ms"
        )
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", default="1,2,3")
    parser.add_argument("--data-dir", default="data/real_subset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-root", default="artifacts/sweep")
    args = parser.parse_args()

    if args.device is None:
        args.device = auto_detect_device()
    print(f"[INFO] device={args.device}, data_dir={args.data_dir}, epochs={args.epochs}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    blocks = {b.strip() for b in args.blocks.split(",") if b.strip()}

    all_rows: list[dict] = []
    if "1" in blocks:
        all_rows.extend(run_block1(args, output_root))
    if "2" in blocks:
        all_rows.extend(run_block2(args, output_root))
    if "3" in blocks:
        all_rows.extend(run_block3(args, output_root))

    summary_path = output_root / "sweep_summary.json"
    save_json(summary_path, {"rows": all_rows})
    print(f"\n[OK] Готово. Сводка: {summary_path}")
    print(f"     Прогнано записей: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
