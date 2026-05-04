"""Подбор гиперпараметров обучения хэш-модуля.

Главная гипотеза: текущий ``margin=4.0`` фиксирован и не масштабируется с длиной
бинарного кода. На code_bits=128/256 это около 3-1.5% от диапазона, и лосс
тривиально выполняется — оптимизатор почти не двигает геометрию, а trained-модель
оказывается хуже untrained.

Скрипт перебирает значения margin (по умолчанию относительные к code_bits, чтобы
было удобно сравнивать). Корпус и триплеты кодируются один раз — каждая точка
сетки — это только тренировка MLP, перестроение бинарных кодов и оценка.

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/run_hash_training_grid.py --code-bits 128
    .venv/bin/python scripts/run_hash_training_grid.py --code-bits 256

Длительность: ~3–5 мин на M-чипе для одной длины кода (один transformer-проход
по корпусу + 4–6 точек тренировки по 10–20 сек).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.training import HashingTrainer, build_triplet_embedding_dataset
from hybrid_search.retrieval.evaluation import SearchEvaluator
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline


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


def hashing_cfg(code_bits: int) -> HashingModelConfig:
    return HashingModelConfig(
        input_dim=384,
        code_bits=code_bits,
        hidden_dims=(256, 128),
        activation="gelu",
        dropout=0.1,
        use_layer_norm=True,
    )


def training_cfg(device: str, epochs: int, margin: float, qw: float, lr: float) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=128,
        learning_rate=lr,
        weight_decay=1e-4,
        margin=margin,
        quantization_weight=qw,
        gradient_clip_norm=1.0,
        temperature=1.0,
        device=device,
    )


def search_cfg(top_k: int = 10, oversample: int = 20) -> SearchConfig:
    return SearchConfig(strategy="coarse_rerank", top_k=top_k, oversample_factor=oversample)


def index_cfg(code_bits: int) -> IndexConfig:
    return IndexConfig(
        code_bits=code_bits,
        oversample_factor=4,
        binary_keep=code_bits,
        backend="hnswlib",
        dense_metric="cosine",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
    )


def parse_grid(raw: str, code_bits: int) -> list[float]:
    """Разрешает запись типа '4,16,32' (абс) или 'r:0.03,0.1,0.25' (отн. к code_bits)."""
    if raw.startswith("r:"):
        factors = [float(x) for x in raw[2:].split(",") if x.strip()]
        return [round(f * code_bits, 2) for f in factors]
    return [float(x) for x in raw.split(",") if x.strip()]


def evaluate_with_model(
    *,
    model: HashingMLP,
    corpus_records,
    corpus_dense,
    encoder,
    preprocessor: TextPreprocessor,
    test_queries: list[dict],
    code_bits: int,
    device: str,
    cfg: SearchConfig,
) -> dict:
    """Перестраивает индекс с заданной моделью и оценивает на test."""
    index = BinaryCodeIndex(
        code_bits=code_bits,
        backend="hnswlib",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
    )
    binary_codes = model.encode_embeddings(corpus_dense, device=device)
    index.add(records=corpus_records, binary_codes=binary_codes, dense_embeddings=corpus_dense)
    pipeline = HybridSearchPipeline(
        preprocessor=preprocessor,
        encoder=encoder,
        hash_model=model,
        index=index,
        device=device,
    )
    evaluator = SearchEvaluator(pipeline)
    return evaluator.evaluate(test_queries, search_config=cfg, warmup_queries=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-bits", type=int, default=128)
    parser.add_argument(
        "--margins",
        default="r:0.03,0.1,0.2,0.4",
        help="Список margin: 'r:0.03,0.1' (относительно code_bits) или '4,12,32' (абсолютные)",
    )
    parser.add_argument("--quantization-weights", default="0.1")
    parser.add_argument("--learning-rates", default="1e-3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-dir", default="data/real_subset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="artifacts/hash_grid")
    args = parser.parse_args()

    if args.device is None:
        args.device = auto_detect_device()

    margins = parse_grid(args.margins, args.code_bits)
    qws = [float(x) for x in args.quantization_weights.split(",") if x.strip()]
    lrs = [float(x) for x in args.learning_rates.split(",") if x.strip()]

    print(f"[INFO] device={args.device}, code_bits={args.code_bits}")
    print(f"[INFO] grid: margins={margins}, qw={qws}, lr={lrs}")
    grid = [(m, q, lr) for m in margins for q in qws for lr in lrs]
    print(f"[INFO] {len(grid)} точек в сетке")

    output_root = Path(args.output_root) / f"cb{args.code_bits}"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Кодируем корпус и триплеты ОДИН раз ----------------------------------
    data_dir = Path(args.data_dir)
    preprocessor = TextPreprocessor()
    encoder = build_encoder(encoder_cfg(args.device), preprocessor=preprocessor)

    print("[INFO] кодирую корпус (один раз)...")
    started = time.perf_counter()
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    corpus_texts = [r.text for r in corpus_records]
    corpus_dense = encoder.encode(corpus_texts)
    print(f"    {len(corpus_records)} текстов за {time.perf_counter() - started:.1f}s")

    print("[INFO] кодирую триплеты (один раз)...")
    started = time.perf_counter()
    triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
    triplet_dataset = build_triplet_embedding_dataset(triplet_examples, encoder, preprocessor)
    print(f"    {len(triplet_examples)} триплетов за {time.perf_counter() - started:.1f}s")

    test_queries = load_jsonl(data_dir / "test_queries.jsonl")
    cfg = search_cfg(top_k=10, oversample=20)

    # 2. Untrained baseline ---------------------------------------------------
    print("\n[BASELINE] untrained (random init)...")
    untrained_model = HashingMLP(hashing_cfg(args.code_bits))
    untrained_model.to(args.device)
    untrained_test = evaluate_with_model(
        model=untrained_model,
        corpus_records=corpus_records,
        corpus_dense=corpus_dense,
        encoder=encoder,
        preprocessor=preprocessor,
        test_queries=test_queries,
        code_bits=args.code_bits,
        device=args.device,
        cfg=cfg,
    )
    print(
        f"    untrained: recall@10={untrained_test['recall@k']:.4f}  "
        f"ndcg@10={untrained_test['ndcg@k']:.4f}"
    )

    # 3. Перебираем grid ------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for margin, qw, lr in grid:
        tag = f"m{margin}_q{qw}_lr{lr}"
        print(f"\n[GRID] {tag}")
        model = HashingMLP(hashing_cfg(args.code_bits))
        trainer = HashingTrainer(model, training_cfg(args.device, args.epochs, margin, qw, lr))
        train_started = time.perf_counter()
        history = trainer.fit(triplet_dataset)
        train_elapsed = time.perf_counter() - train_started

        try:
            test = evaluate_with_model(
                model=model,
                corpus_records=corpus_records,
                corpus_dense=corpus_dense,
                encoder=encoder,
                preprocessor=preprocessor,
                test_queries=test_queries,
                code_bits=args.code_bits,
                device=args.device,
                cfg=cfg,
            )
        except RuntimeError as exc:
            # Типичный кейс: при слишком агрессивном margin модель схлопывается,
            # уникальных бинарных кодов остаётся мало и hnswlib не может вернуть
            # top-N. Записываем точку как fail и идём дальше.
            print(f"    [FAIL] eval упал: {exc}")
            row = {
                "code_bits": args.code_bits,
                "margin": margin,
                "margin_relative": round(margin / args.code_bits, 3),
                "quantization_weight": qw,
                "learning_rate": lr,
                "epochs": args.epochs,
                "train_time_s": round(train_elapsed, 1),
                "recall@10": None,
                "ndcg@10": None,
                "map@10": None,
                "latency_ms": None,
                "delta_recall_vs_untrained": None,
                "final_loss": history[-1]["loss"] if history else None,
                "final_triplet_loss": history[-1]["triplet_loss"] if history else None,
                "final_quantization_loss": history[-1]["quantization_loss"] if history else None,
                "error": str(exc),
            }
            save_json(output_root / f"{tag}.json", {"row": row, "history": history})
            rows.append(row)
            continue

        delta_vs_untrained = test["recall@k"] - untrained_test["recall@k"]
        row = {
            "code_bits": args.code_bits,
            "margin": margin,
            "margin_relative": round(margin / args.code_bits, 3),
            "quantization_weight": qw,
            "learning_rate": lr,
            "epochs": args.epochs,
            "train_time_s": round(train_elapsed, 1),
            "recall@10": test["recall@k"],
            "ndcg@10": test["ndcg@k"],
            "map@10": test["map@k"],
            "latency_ms": test["latency_ms"],
            "delta_recall_vs_untrained": delta_vs_untrained,
            "final_loss": history[-1]["loss"] if history else None,
            "final_triplet_loss": history[-1]["triplet_loss"] if history else None,
            "final_quantization_loss": history[-1]["quantization_loss"] if history else None,
        }
        save_json(output_root / f"{tag}.json", {"row": row, "history": history, "test": test})
        rows.append(row)
        print(
            f"    recall@10={row['recall@10']:.4f}  Δ vs untrained={delta_vs_untrained:+.4f}  "
            f"loss={row['final_loss']:.4f}  train={row['train_time_s']}s"
        )

    # 4. Итоговая таблица ----------------------------------------------------
    rows_sorted = sorted(rows, key=lambda r: r["recall@10"], reverse=True)
    summary = {
        "code_bits": args.code_bits,
        "untrained": {
            "recall@10": untrained_test["recall@k"],
            "ndcg@10": untrained_test["ndcg@k"],
            "map@10": untrained_test["map@k"],
            "latency_ms": untrained_test["latency_ms"],
        },
        "rows": rows_sorted,
    }
    save_json(output_root / "summary.json", summary)

    print("\n" + "=" * 78)
    print(f"Сводка (code_bits={args.code_bits}, untrained recall={untrained_test['recall@k']:.4f}):")
    print("=" * 78)
    print(f"{'margin':>10s}  {'rel':>6s}  {'qw':>5s}  {'lr':>8s}  "
          f"{'recall':>8s}  {'Δ untr':>8s}  {'loss':>7s}")
    for row in rows_sorted:
        marker = " ← BEST" if row is rows_sorted[0] else ""
        print(
            f"{row['margin']:>10.2f}  {row['margin_relative']:>6.2f}  "
            f"{row['quantization_weight']:>5.2f}  {row['learning_rate']:>8.0e}  "
            f"{row['recall@10']:>8.4f}  {row['delta_recall_vs_untrained']:>+8.4f}  "
            f"{row['final_loss'] or 0:>7.4f}{marker}"
        )
    print(f"\n[OK] {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
