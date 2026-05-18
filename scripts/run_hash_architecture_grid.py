"""Ablation study архитектуры хэш-головы.

Предыдущие эксперименты показали: наша обученная distillation-MLP проигрывает
классическим методам (ITQ ≈ dense_exact, наш MLP −8 п.п. в binary-only).
Гипотеза: проблема не в обучении, а в архитектуре — глубокая MLP с LayerNorm
и нелинейностями искажает геометрию dense-эмбеддингов сильнее, чем компенсирует
обучением на ограниченных триплетах.

Этот скрипт тестирует разные конфигурации хэш-головы при одинаковых:
    - данных (data/real_300k)
    - энкодере (frozen pretrained MiniLM-L6)
    - distillation loss и гиперпараметрах обучения
    - метрике (recall@10 в binary_only и rerank)

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    caffeinate -i .venv/bin/python scripts/run_hash_architecture_grid.py \\
        --data-dir data/real_300k \\
        --output-dir artifacts/hash_arch_grid \\
        --code-bits 256

Время: ~30 мин на M-чипе (5 мин encoder + 10 конфигураций × 2 мин).
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

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from hybrid_search.core.config import (
    EncoderConfig, HashingModelConfig, SearchConfig, TrainingConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.training import DistillationTrainer, build_triplet_embedding_dataset
from hybrid_search.retrieval.evaluation import (
    average_precision_at_k, ndcg_at_k, recall_at_k,
)
from hybrid_search.retrieval.indexing import BinaryCodeIndex


# ---------------------------------------------------------------------------
# Сетка архитектур
# ---------------------------------------------------------------------------
def architecture_grid() -> list[dict[str, Any]]:
    """10 конфигураций, покрывающих основные оси: глубина, LN, dropout, активация."""
    return [
        # Depth ablation: 0 / 1 / 2 / 3 hidden layers
        {"name": "linear",          "hidden_dims": (),               "use_layer_norm": False, "activation": "gelu", "dropout": 0.0},
        {"name": "1h_no_ln",        "hidden_dims": (256,),           "use_layer_norm": False, "activation": "gelu", "dropout": 0.0},
        {"name": "1h_with_ln",      "hidden_dims": (256,),           "use_layer_norm": True,  "activation": "gelu", "dropout": 0.0},
        {"name": "2h_no_ln_no_drop","hidden_dims": (256, 128),       "use_layer_norm": False, "activation": "gelu", "dropout": 0.0},
        {"name": "2h_no_drop",      "hidden_dims": (256, 128),       "use_layer_norm": True,  "activation": "gelu", "dropout": 0.0},
        {"name": "2h_default",      "hidden_dims": (256, 128),       "use_layer_norm": True,  "activation": "gelu", "dropout": 0.1},
        {"name": "3h_deep",         "hidden_dims": (512, 256, 128),  "use_layer_norm": True,  "activation": "gelu", "dropout": 0.1},
        # Activation ablation
        {"name": "2h_relu",         "hidden_dims": (256, 128),       "use_layer_norm": True,  "activation": "relu", "dropout": 0.0},
        {"name": "2h_tanh",         "hidden_dims": (256, 128),       "use_layer_norm": True,  "activation": "tanh", "dropout": 0.0},
        # Width ablation
        {"name": "1h_wide_768",     "hidden_dims": (768,),           "use_layer_norm": True,  "activation": "gelu", "dropout": 0.0},
    ]


def auto_detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Оценка (binary-only и rerank)
# ---------------------------------------------------------------------------
def evaluate_codes(
    *,
    encoder,
    hash_codes_corpus: np.ndarray,
    hash_codes_query_fn,
    corpus_records,
    corpus_dense: np.ndarray,
    test_queries: list[dict],
    code_bits: int,
    cfg: SearchConfig,
    preprocessor: TextPreprocessor,
    rerank: bool,
) -> dict:
    dense_for_index = corpus_dense if rerank else np.zeros((len(corpus_records), 1), dtype=np.float32)
    index = BinaryCodeIndex(
        code_bits=code_bits, backend="hnswlib",
        hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=64,
    )
    index.add(records=corpus_records, binary_codes=hash_codes_corpus, dense_embeddings=dense_for_index)

    rows = list(test_queries)
    metrics_records = []
    latencies = []
    for row in rows:
        query_text = str(row["query"])
        relevant = set(map(str, row["relevant_ids"]))
        normalized = preprocessor.normalize(query_text)

        started = time.perf_counter()
        dense_query = encoder.encode([normalized])[0].astype(np.float32, copy=False)
        binary_query = hash_codes_query_fn(dense_query.reshape(1, -1))[0]
        top_n = max(cfg.top_k * cfg.oversample_factor, cfg.top_k) if rerank else cfg.top_k
        candidate_indices, _ = index.candidate_indices(binary_query, top_n)
        if len(candidate_indices) == 0:
            ranked_ids = []
        elif rerank:
            scores = index.cosine_scores(dense_query, indices=candidate_indices)
            ranked_local = np.argsort(-scores, kind="stable")[: cfg.top_k]
            ranked_ids = [index.records[int(candidate_indices[i])].record_id for i in ranked_local]
        else:
            ranked_ids = [index.records[int(i)].record_id for i in candidate_indices[: cfg.top_k]]
        latencies.append((time.perf_counter() - started) * 1000.0)

        metrics_records.append({
            "recall": recall_at_k(ranked_ids, relevant, cfg.top_k),
            "map_k": average_precision_at_k(ranked_ids, relevant, cfg.top_k),
            "ndcg": ndcg_at_k(ranked_ids, relevant, cfg.top_k),
        })

    return {
        "recall@k": float(np.mean([m["recall"] for m in metrics_records])),
        "ndcg@k": float(np.mean([m["ndcg"] for m in metrics_records])),
        "map@k": float(np.mean([m["map_k"] for m in metrics_records])),
        "latency_ms": float(np.mean(latencies)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/real_300k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-bits", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantization-weight", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = args.device or auto_detect_device()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device}, output={output_root}", flush=True)

    # 1. Энкодер и данные ----------------------------------------------------
    preprocessor = TextPreprocessor()
    encoder_cfg = EncoderConfig(
        backend="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384, batch_size=128, normalize_embeddings=True,
        device=device, seed=args.seed,
    )
    encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)

    data_dir = Path(args.data_dir)
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")

    print(f"[INFO] corpus={len(corpus_records)}, triplets={len(triplet_examples)}, "
          f"test_queries={len(test_queries)}", flush=True)

    # 2. Кодирование один раз ------------------------------------------------
    print("[INFO] кодирую триплеты (один раз)...", flush=True)
    started = time.perf_counter()
    triplet_dataset = build_triplet_embedding_dataset(triplet_examples, encoder, preprocessor=preprocessor)
    print(f"    {time.perf_counter() - started:.1f}s", flush=True)

    print("[INFO] кодирую корпус (один раз)...", flush=True)
    started = time.perf_counter()
    corpus_dense = encoder.encode([r.text for r in corpus_records])
    print(f"    {time.perf_counter() - started:.1f}s", flush=True)

    cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k, oversample_factor=args.oversample)

    # 3. Прогон сетки --------------------------------------------------------
    grid = architecture_grid()
    print(f"\n[INFO] тестирую {len(grid)} архитектур; код {args.code_bits} бит", flush=True)
    results: list[dict[str, Any]] = []

    for i, arch in enumerate(grid, start=1):
        name = arch.pop("name")
        print(f"\n[{i}/{len(grid)}] {name}: hidden_dims={arch['hidden_dims']}, "
              f"LN={arch['use_layer_norm']}, dropout={arch['dropout']}, "
              f"act={arch['activation']}", flush=True)

        hash_cfg = HashingModelConfig(input_dim=384, code_bits=args.code_bits, **arch)
        model = HashingMLP(hash_cfg)
        param_count = sum(p.numel() for p in model.parameters())

        train_cfg = TrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=1e-4,
            margin=0.0, quantization_weight=args.quantization_weight,
            gradient_clip_norm=1.0, temperature=1.0, device=device,
        )
        trainer = DistillationTrainer(model, train_cfg, distillation_weight=1.0)

        train_started = time.perf_counter()
        history = trainer.fit(triplet_dataset)
        train_time = time.perf_counter() - train_started

        # Кодируем корпус и запросы текущим хэшем
        corpus_codes = model.encode_embeddings(corpus_dense, device=device)

        def query_fn(dense_q: np.ndarray) -> np.ndarray:
            return model.encode_embeddings(dense_q, device=device)

        binary_only_metrics = evaluate_codes(
            encoder=encoder, hash_codes_corpus=corpus_codes, hash_codes_query_fn=query_fn,
            corpus_records=corpus_records, corpus_dense=corpus_dense,
            test_queries=test_queries, code_bits=args.code_bits, cfg=cfg,
            preprocessor=preprocessor, rerank=False,
        )
        rerank_metrics = evaluate_codes(
            encoder=encoder, hash_codes_corpus=corpus_codes, hash_codes_query_fn=query_fn,
            corpus_records=corpus_records, corpus_dense=corpus_dense,
            test_queries=test_queries, code_bits=args.code_bits, cfg=cfg,
            preprocessor=preprocessor, rerank=True,
        )

        result = {
            "name": name,
            "hidden_dims": list(arch["hidden_dims"]),
            "use_layer_norm": arch["use_layer_norm"],
            "activation": arch["activation"],
            "dropout": arch["dropout"],
            "param_count": int(param_count),
            "train_time_s": round(train_time, 1),
            "final_loss": history[-1]["loss"] if history else None,
            "final_distill_loss": history[-1].get("distillation_loss") if history else None,
            "binary_only": binary_only_metrics,
            "rerank": rerank_metrics,
        }
        results.append(result)

        print(
            f"    binary_only: recall@10={binary_only_metrics['recall@k']:.4f}  "
            f"rerank: recall@10={rerank_metrics['recall@k']:.4f}  "
            f"params={param_count:,}  train={train_time:.1f}s  loss={result['final_loss']:.4f}",
            flush=True,
        )

    # 4. Сохраняем и печатаем ------------------------------------------------
    save_json(output_root / "grid_results.json", {"args": vars(args), "results": results})

    print("\n" + "=" * 110)
    print(f"Архитектурный grid (code_bits={args.code_bits}, distillation):")
    print("=" * 110)
    print(f"{'name':<22s}  {'params':>10s}  {'binary_only':>12s}  {'rerank':>10s}  "
          f"{'lat_bin':>8s}  {'lat_rer':>8s}  {'loss':>8s}")
    print("-" * 110)
    # Сортируем по binary_only recall (показатель «сырого» качества хэша)
    rows_sorted = sorted(results, key=lambda r: r["binary_only"]["recall@k"], reverse=True)
    for r in rows_sorted:
        marker = " ← BEST binary" if r is rows_sorted[0] else ""
        print(f"{r['name']:<22s}  {r['param_count']:>10,d}  "
              f"{r['binary_only']['recall@k']:>12.4f}  {r['rerank']['recall@k']:>10.4f}  "
              f"{r['binary_only']['latency_ms']:>8.2f}  {r['rerank']['latency_ms']:>8.2f}  "
              f"{(r['final_loss'] or 0):>8.4f}{marker}")
    print("=" * 110)
    print(f"\n[OK] {output_root / 'grid_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
