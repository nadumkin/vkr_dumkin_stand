"""Унифицированное сравнение бейслайнов компрессии векторов.

Сравнивает на одинаковых данных, кодах одинаковой длины, и одной test-выборке:
    - dense_exact         — точный плотный поиск (верхний потолок качества)
    - LSH                 — случайные гиперплоскости [Lu et al., 2018]
    - ITQ                 — Iterative Quantization [Gong et al., 2013]
    - PQ                  — Product Quantization [Jégou et al., 2011]
    - hybrid_distilled    — наш метод (загружается из артефакта если уже обучен)

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    caffeinate -i .venv/bin/python scripts/run_baselines_comparison.py \
        --data-dir data/real_300k \
        --output-dir artifacts/baselines/cb256 \
        --code-bits 256 \
        --pq-subspaces 32 \
        --pq-centroids 256

Время на M-чипе: ~10 минут (5 мин encoder forward по 224K корпусу + 1 мин ITQ
+ 2 мин PQ k-means + поиск). Distillation-точка либо обучается заново
(добавляет ~3 мин), либо загружается из существующего артефакта (мгновенно).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    EncoderConfig,
    HashingModelConfig,
    SearchConfig,
    TrainingConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.core.utils import l2_normalize
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.baselines import ITQHasher, LSHHasher, ProductQuantizer
from hybrid_search.models.encoders import build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.training import DistillationTrainer, build_triplet_embedding_dataset
from hybrid_search.retrieval.dense_search import DenseSearchPipeline
from hybrid_search.retrieval.evaluation import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline
from hybrid_search.retrieval.evaluation import SearchEvaluator


def auto_detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Утилиты для оценки
# ---------------------------------------------------------------------------
def evaluate_with_codes(
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
    device: str,
    method_name: str,
    rerank: bool = True,
) -> dict:
    """Оценка для бинарных методов (LSH/ITQ/distillation).

    Параметр ``rerank``:
        True  — двухэтапный pipeline: top_k×oversample кандидатов по hamming,
                затем cosine rerank на dense-векторах → top_k.
        False — binary-only: берём top_k кандидатов сразу по hamming-расстоянию
                без обращения к dense-векторам. Это «голое» качество хэша.
    """
    mode = "rerank" if rerank else "binary_only"
    print(f"[eval/{method_name}/{mode}] строю BinaryCodeIndex...", flush=True)
    # Для binary-only можно не хранить dense_embeddings — экономим память.
    dense_for_index = corpus_dense if rerank else np.zeros((len(corpus_records), 1), dtype=np.float32)
    index = BinaryCodeIndex(
        code_bits=code_bits,
        backend="hnswlib",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
    )
    index.add(records=corpus_records, binary_codes=hash_codes_corpus, dense_embeddings=dense_for_index)

    rows = list(test_queries)
    n = len(rows)
    metrics_records: list[dict] = []
    encode_times: list[float] = []
    cand_times: list[float] = []
    rerank_times: list[float] = []
    total_times: list[float] = []

    for row in rows:
        query_text = str(row["query"])
        relevant = set(map(str, row["relevant_ids"]))
        normalized = preprocessor.normalize(query_text)

        total_started = time.perf_counter()

        encode_started = time.perf_counter()
        dense_query = encoder.encode([normalized])[0].astype(np.float32, copy=False)
        binary_query = hash_codes_query_fn(dense_query.reshape(1, -1))[0]
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        cand_started = time.perf_counter()
        top_n = max(cfg.top_k * cfg.oversample_factor, cfg.top_k) if rerank else cfg.top_k
        candidate_indices, candidate_distances = index.candidate_indices(binary_query, top_n)
        cand_ms = (time.perf_counter() - cand_started) * 1000.0

        rerank_ms = 0.0
        if len(candidate_indices) == 0:
            ranked_ids: list[str] = []
        elif rerank:
            rerank_started = time.perf_counter()
            dense_scores = index.cosine_scores(dense_query, indices=candidate_indices)
            ranked_local = np.argsort(-dense_scores, kind="stable")[: cfg.top_k]
            ranked_ids = [
                index.records[int(candidate_indices[idx])].record_id
                for idx in ranked_local
            ]
            rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        else:
            # binary-only: hnswlib и numpy backend оба возвращают candidate_indices
            # отсортированными по hamming-расстоянию по возрастанию.
            ranked_ids = [
                index.records[int(idx)].record_id
                for idx in candidate_indices[: cfg.top_k]
            ]

        total_ms = (time.perf_counter() - total_started) * 1000.0

        metrics_records.append({
            "recall": recall_at_k(ranked_ids, relevant, cfg.top_k),
            "map_k": average_precision_at_k(ranked_ids, relevant, cfg.top_k),
            "ndcg": ndcg_at_k(ranked_ids, relevant, cfg.top_k),
        })
        encode_times.append(encode_ms)
        cand_times.append(cand_ms)
        rerank_times.append(rerank_ms)
        total_times.append(total_ms)

    return {
        "queries": n,
        "mode": mode,
        "recall@k": float(np.mean([m["recall"] for m in metrics_records])),
        "map@k": float(np.mean([m["map_k"] for m in metrics_records])),
        "ndcg@k": float(np.mean([m["ndcg"] for m in metrics_records])),
        "latency_ms": float(np.mean(total_times)),
        "latency_breakdown_ms": {
            "query_encode_ms": float(np.mean(encode_times)),
            "candidate_selection_ms": float(np.mean(cand_times)),
            "rerank_ms": float(np.mean(rerank_times)),
        },
        # Per-query метрики для bootstrap CI; порядок соответствует test_queries.
        "per_query": {
            "recall@k": [float(m["recall"]) for m in metrics_records],
            "map@k": [float(m["map_k"]) for m in metrics_records],
            "ndcg@k": [float(m["ndcg"]) for m in metrics_records],
        },
    }


def evaluate_pq(
    *,
    encoder,
    pq: ProductQuantizer,
    db_codes: np.ndarray,
    corpus_records,
    corpus_dense: np.ndarray | None,
    test_queries: list[dict],
    cfg: SearchConfig,
    preprocessor: TextPreprocessor,
    rerank: bool = False,
) -> dict:
    """Оценка PQ через ADC.

    Параметр ``rerank``:
        False — стандартный режим PQ: ADC → top_k. Не требует dense-векторов
                в памяти, минимальная RAM (типичное использование PQ).
        True  — гибридный: ADC → top_k×oversample кандидатов → cosine rerank
                на dense → top_k. Требует dense_embeddings, как у бинарных
                методов с rerank. Симметричный с LSH/ITQ/distill.
    """
    if rerank and corpus_dense is None:
        raise ValueError("PQ rerank требует corpus_dense")
    mode = "rerank" if rerank else "binary_only"
    rows = list(test_queries)
    metrics_records: list[dict] = []
    encode_times: list[float] = []
    search_times: list[float] = []
    rerank_times: list[float] = []
    total_times: list[float] = []

    for row in rows:
        query_text = str(row["query"])
        relevant = set(map(str, row["relevant_ids"]))
        normalized = preprocessor.normalize(query_text)

        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        dense_query = encoder.encode([normalized])[0].astype(np.float32, copy=False)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        n_candidates = max(cfg.top_k * cfg.oversample_factor, cfg.top_k) if rerank else cfg.top_k
        search_started = time.perf_counter()
        result = pq.search(dense_query, db_codes, n_candidates)
        search_ms = (time.perf_counter() - search_started) * 1000.0

        rerank_ms = 0.0
        if rerank and len(result.indices) > 0:
            rerank_started = time.perf_counter()
            cand_dense = corpus_dense[result.indices]
            # cosine — векторы уже L2-нормированы при энкодинге
            normalized_q = dense_query / max(float(np.linalg.norm(dense_query)), 1e-6)
            scores = cand_dense @ normalized_q
            ranked_local = np.argsort(-scores, kind="stable")[: cfg.top_k]
            ranked_ids = [corpus_records[int(result.indices[i])].record_id for i in ranked_local]
            rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        else:
            ranked_ids = [corpus_records[int(idx)].record_id for idx in result.indices[: cfg.top_k]]

        total_ms = (time.perf_counter() - total_started) * 1000.0

        metrics_records.append({
            "recall": recall_at_k(ranked_ids, relevant, cfg.top_k),
            "map_k": average_precision_at_k(ranked_ids, relevant, cfg.top_k),
            "ndcg": ndcg_at_k(ranked_ids, relevant, cfg.top_k),
        })
        encode_times.append(encode_ms)
        search_times.append(search_ms)
        rerank_times.append(rerank_ms)
        total_times.append(total_ms)

    return {
        "queries": len(rows),
        "mode": mode,
        "recall@k": float(np.mean([m["recall"] for m in metrics_records])),
        "map@k": float(np.mean([m["map_k"] for m in metrics_records])),
        "ndcg@k": float(np.mean([m["ndcg"] for m in metrics_records])),
        "latency_ms": float(np.mean(total_times)),
        "latency_breakdown_ms": {
            "query_encode_ms": float(np.mean(encode_times)),
            "candidate_selection_ms": float(np.mean(search_times)),
            "rerank_ms": float(np.mean(rerank_times)),
        },
        # Per-query метрики для bootstrap CI; порядок соответствует test_queries.
        "per_query": {
            "recall@k": [float(m["recall"]) for m in metrics_records],
            "map@k": [float(m["map_k"]) for m in metrics_records],
            "ndcg@k": [float(m["ndcg"]) for m in metrics_records],
        },
    }


def memory_bytes_binary(
    n_docs: int,
    code_bits: int,
    hnsw_m: int = 32,
    dense_dim: int | None = None,
) -> dict:
    """Оценка памяти для бинарных методов.

    Если ``dense_dim`` указан — учитываем хранение dense-векторов
    (нужно для rerank-режима). Если None — только бинарные коды + HNSW граф.
    """
    packed_bytes = n_docs * (code_bits // 8 if code_bits % 8 == 0 else code_bits // 8 + 1)
    hnsw_graph = n_docs * (2 * hnsw_m * 4 + 4) + n_docs * code_bits * 4
    dense_bytes = n_docs * dense_dim * 4 if dense_dim is not None else 0
    return {
        "packed_codes_bytes": int(packed_bytes),
        "hnsw_graph_bytes": int(hnsw_graph),
        "dense_for_rerank_bytes": int(dense_bytes),
        "total_bytes": int(packed_bytes + hnsw_graph + dense_bytes),
    }


def memory_bytes_pq(
    n_docs: int,
    n_subspaces: int,
    n_centroids: int,
    sub_dim: int,
    dense_dim: int | None = None,
) -> dict:
    """Оценка памяти для PQ: коды + кодовые книги (+ dense для rerank, если нужно)."""
    code_bytes = n_docs * n_subspaces * (1 if n_centroids <= 256 else 2)
    book_bytes = n_subspaces * n_centroids * sub_dim * 4
    dense_bytes = n_docs * dense_dim * 4 if dense_dim is not None else 0
    return {
        "codes_bytes": int(code_bytes),
        "codebooks_bytes": int(book_bytes),
        "dense_for_rerank_bytes": int(dense_bytes),
        "total_bytes": int(code_bytes + book_bytes + dense_bytes),
    }


def memory_bytes_dense(n_docs: int, dim: int) -> dict:
    bytes_total = n_docs * dim * 4
    return {"dense_embeddings_bytes": int(bytes_total), "total_bytes": int(bytes_total)}


# ---------------------------------------------------------------------------
# Главный поток
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/real_300k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)

    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--code-bits", type=int, default=256)

    # PQ параметры (M подпространств × K центроидов)
    parser.add_argument("--pq-subspaces", type=int, default=32)
    parser.add_argument("--pq-centroids", type=int, default=256)
    parser.add_argument("--pq-iter", type=int, default=25)

    # ITQ параметры
    parser.add_argument("--itq-iter", type=int, default=50)

    # Distillation: либо переобучить заново, либо взять из существующего артефакта
    parser.add_argument("--distill-from-artifact", default=None,
                        help="Путь к существующему артефакту для загрузки hash-MLP "
                             "(например, artifacts/distill/cb256). "
                             "Если не указан — обучаем дистилляцию здесь же.")
    parser.add_argument("--distill-epochs", type=int, default=8)
    parser.add_argument("--distill-batch-size", type=int, default=128)
    parser.add_argument("--distill-quant-weight", type=float, default=0.01)

    # Joint pair: добавляем чекпойнт finetuned-энкодера + хэша для отдельного метода
    parser.add_argument("--joint-checkpoint", default=None,
                        help="Опциональный путь к каталогу joint-чекпойнта "
                             "(должен содержать encoder.pt и hash_model.pt; "
                             "также читается meta.json для конфигурации хэша). "
                             "Если задан, добавляется метод 'joint_pair' с "
                             "finetuned-энкодером + обученным хэшем.")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = args.device or auto_detect_device()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"[INFO] device={device}, output={output_root}", flush=True)
    print(f"[INFO] code_bits={args.code_bits}, PQ M×K = {args.pq_subspaces}×{args.pq_centroids}",
          flush=True)

    # 1. Энкодер -------------------------------------------------------------
    preprocessor = TextPreprocessor()
    encoder_cfg = EncoderConfig(
        backend="transformers",
        model_name=args.transformer_model,
        embedding_dim=384,
        batch_size=128,
        normalize_embeddings=True,
        device=device,
        seed=args.seed,
    )
    encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
    print(f"[INFO] encoder loaded: {args.transformer_model}", flush=True)

    # 2. Данные --------------------------------------------------------------
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")
    print(f"[INFO] corpus={len(corpus_records)}, test_queries={len(test_queries)}", flush=True)

    cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k, oversample_factor=args.oversample)

    # 3. Кодируем корпус один раз -------------------------------------------
    print("[INFO] кодирую корпус (один раз для всех методов)...", flush=True)
    started = time.perf_counter()
    corpus_dense = encoder.encode([r.text for r in corpus_records])
    print(f"    {len(corpus_records)} текстов за {time.perf_counter() - started:.1f}s",
          flush=True)

    # 4. Прогон бейслайнов в обоих режимах ----------------------------------
    results: dict[str, Any] = {}
    timings: dict[str, dict[str, Any]] = {}
    n_docs = len(corpus_records)
    dense_dim = corpus_dense.shape[1]

    # 4a. dense_exact (один режим — это сам потолок)
    print("\n[run] dense_exact (точный плотный поиск)...", flush=True)
    dense_pipeline = DenseSearchPipeline(preprocessor=preprocessor, encoder=encoder)
    dense_pipeline.records = list(corpus_records)
    dense_pipeline.dense_embeddings = l2_normalize(corpus_dense)
    dense_pipeline.last_encode_time_ms = 0.0
    dense_eval = SearchEvaluator(dense_pipeline).evaluate(test_queries, search_config=cfg, warmup_queries=5)
    results["dense_exact"] = dense_eval
    results["dense_exact"]["memory"] = memory_bytes_dense(n_docs, dense_dim)
    print(f"    recall@10={dense_eval['recall@k']:.4f}  ndcg@10={dense_eval['ndcg@k']:.4f}  "
          f"latency={dense_eval['latency_ms']:.2f}ms", flush=True)

    # 4b. LSH (оба режима)
    print("\n[run] LSH (random hyperplanes) — оба режима...", flush=True)
    lsh = LSHHasher(input_dim=dense_dim, code_bits=args.code_bits, seed=args.seed)
    lsh.fit(corpus_dense)
    lsh_codes = lsh.encode(corpus_dense)
    timings["lsh_fit_ms"] = lsh.last_fit_time_ms
    for use_rerank, key in [(False, "lsh_binary_only"), (True, "lsh_rerank")]:
        results[key] = evaluate_with_codes(
            encoder=encoder, hash_codes_corpus=lsh_codes, hash_codes_query_fn=lsh.encode,
            corpus_records=corpus_records, corpus_dense=corpus_dense,
            test_queries=test_queries, code_bits=args.code_bits, cfg=cfg,
            preprocessor=preprocessor, device=device, method_name="lsh",
            rerank=use_rerank,
        )
        results[key]["memory"] = memory_bytes_binary(
            n_docs, args.code_bits, dense_dim=dense_dim if use_rerank else None
        )
        print(f"    [{key}] recall@10={results[key]['recall@k']:.4f}  "
              f"latency={results[key]['latency_ms']:.2f}ms", flush=True)

    # 4c. ITQ (оба режима)
    print("\n[run] ITQ (Iterative Quantization) — оба режима...", flush=True)
    itq = ITQHasher(input_dim=dense_dim, code_bits=args.code_bits,
                     n_iter=args.itq_iter, seed=args.seed)
    itq.fit(corpus_dense)
    itq_codes = itq.encode(corpus_dense)
    timings["itq_fit_ms"] = itq.last_fit_time_ms
    for use_rerank, key in [(False, "itq_binary_only"), (True, "itq_rerank")]:
        results[key] = evaluate_with_codes(
            encoder=encoder, hash_codes_corpus=itq_codes, hash_codes_query_fn=itq.encode,
            corpus_records=corpus_records, corpus_dense=corpus_dense,
            test_queries=test_queries, code_bits=args.code_bits, cfg=cfg,
            preprocessor=preprocessor, device=device, method_name="itq",
            rerank=use_rerank,
        )
        results[key]["memory"] = memory_bytes_binary(
            n_docs, args.code_bits, dense_dim=dense_dim if use_rerank else None
        )
        print(f"    [{key}] recall@10={results[key]['recall@k']:.4f}  "
              f"latency={results[key]['latency_ms']:.2f}ms", flush=True)

    # 4d. PQ (оба режима)
    print(f"\n[run] PQ (M={args.pq_subspaces}, K={args.pq_centroids}) — оба режима...", flush=True)
    pq = ProductQuantizer(
        input_dim=dense_dim, n_subspaces=args.pq_subspaces,
        n_centroids=args.pq_centroids, n_iter=args.pq_iter, seed=args.seed,
    )
    pq.fit(corpus_dense)
    pq_codes = pq.encode(corpus_dense)
    timings["pq_fit_ms"] = pq.last_fit_time_ms
    for use_rerank, key in [(False, "pq_binary_only"), (True, "pq_rerank")]:
        results[key] = evaluate_pq(
            encoder=encoder, pq=pq, db_codes=pq_codes,
            corpus_records=corpus_records,
            corpus_dense=l2_normalize(corpus_dense) if use_rerank else None,
            test_queries=test_queries, cfg=cfg, preprocessor=preprocessor,
            rerank=use_rerank,
        )
        results[key]["memory"] = memory_bytes_pq(
            n_docs, args.pq_subspaces, args.pq_centroids, pq.sub_dim,
            dense_dim=dense_dim if use_rerank else None,
        )
        results[key]["code_bits_total"] = pq.code_bits
        print(f"    [{key}] recall@10={results[key]['recall@k']:.4f}  "
              f"latency={results[key]['latency_ms']:.2f}ms", flush=True)

    # 4e. Distillation (оба режима)
    print(f"\n[run] hybrid_distilled (наш метод) — оба режима...", flush=True)
    hash_cfg = HashingModelConfig(
        input_dim=384, code_bits=args.code_bits,
        hidden_dims=(256, 128), activation="gelu", dropout=0.1, use_layer_norm=True,
    )
    hash_model = HashingMLP(hash_cfg)

    if args.distill_from_artifact:
        artifact = Path(args.distill_from_artifact)
        ckpt_candidates = [
            artifact / "hash_model.pt",
            artifact / "distillation_hash_model.pt",
            artifact / "checkpoints" / "checkpoint_epoch8" / "hash_model.pt",
            artifact / "checkpoint" / "hash_model.pt",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
        if ckpt_path is None:
            print(f"    [WARN] не нашёл hash_model.pt в {artifact}; обучаю заново", flush=True)
            args.distill_from_artifact = None
        else:
            print(f"    загружаю distillation hash из {ckpt_path}", flush=True)
            loaded = torch.load(ckpt_path, map_location=device)
            # Поддерживаем два формата:
            #   1) Полный checkpoint: {"model_config": {...}, "state_dict": {...}, ...}
            #   2) Голый state_dict (legacy)
            if isinstance(loaded, dict) and "model_config" in loaded and "state_dict" in loaded:
                cfg_payload = loaded["model_config"]
                saved_cfg = HashingModelConfig(
                    input_dim=int(cfg_payload["input_dim"]),
                    code_bits=int(cfg_payload["code_bits"]),
                    hidden_dims=tuple(cfg_payload["hidden_dims"]),
                    activation=str(cfg_payload["activation"]),
                    dropout=float(cfg_payload["dropout"]),
                    use_layer_norm=bool(cfg_payload["use_layer_norm"]),
                )
                if (saved_cfg.hidden_dims != hash_cfg.hidden_dims
                        or saved_cfg.use_layer_norm != hash_cfg.use_layer_norm
                        or saved_cfg.activation != hash_cfg.activation
                        or saved_cfg.dropout != hash_cfg.dropout):
                    print(f"    [INFO] архитектура из checkpoint: hidden_dims={saved_cfg.hidden_dims}, "
                          f"LN={saved_cfg.use_layer_norm}, act={saved_cfg.activation}, "
                          f"dropout={saved_cfg.dropout} — пересоздаю HashingMLP", flush=True)
                    hash_model = HashingMLP(saved_cfg)
                    hash_cfg = saved_cfg
                hash_model.load_state_dict(loaded["state_dict"])
            else:
                hash_model.load_state_dict(loaded)
            hash_model.to(device)
            hash_model.eval()

    if not args.distill_from_artifact:
        print("    обучаю distillation заново...", flush=True)
        triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
        triplet_dataset = build_triplet_embedding_dataset(triplet_examples, encoder, preprocessor=preprocessor)
        train_cfg = TrainingConfig(
            epochs=args.distill_epochs, batch_size=args.distill_batch_size,
            learning_rate=1e-3, weight_decay=1e-4,
            margin=0.0, quantization_weight=args.distill_quant_weight,
            gradient_clip_norm=1.0, temperature=1.0, device=device,
        )
        trainer = DistillationTrainer(hash_model, train_cfg, distillation_weight=1.0)
        trainer.fit(triplet_dataset)
        torch.save(hash_model.state_dict(), output_root / "distillation_hash_model.pt")

    distill_codes = hash_model.encode_embeddings(corpus_dense, device=device)

    def distill_query_fn(dense_q: np.ndarray) -> np.ndarray:
        return hash_model.encode_embeddings(dense_q, device=device)

    for use_rerank, key in [(False, "distill_binary_only"), (True, "distill_rerank")]:
        results[key] = evaluate_with_codes(
            encoder=encoder, hash_codes_corpus=distill_codes,
            hash_codes_query_fn=distill_query_fn,
            corpus_records=corpus_records, corpus_dense=corpus_dense,
            test_queries=test_queries, code_bits=args.code_bits, cfg=cfg,
            preprocessor=preprocessor, device=device, method_name="distill",
            rerank=use_rerank,
        )
        results[key]["memory"] = memory_bytes_binary(
            n_docs, args.code_bits, dense_dim=dense_dim if use_rerank else None
        )
        print(f"    [{key}] recall@10={results[key]['recall@k']:.4f}  "
              f"latency={results[key]['latency_ms']:.2f}ms", flush=True)

    # 4f. Joint pair (finetuned encoder + trained hash) — опционально --------
    joint_pair_added = False
    joint_code_bits = args.code_bits  # значение по умолчанию для таблицы
    if args.joint_checkpoint:
        ckpt = Path(args.joint_checkpoint)
        enc_path = ckpt / "encoder.pt"
        hash_path = ckpt / "hash_model.pt"
        meta_path = ckpt / "meta.json"
        if not (enc_path.exists() and hash_path.exists()):
            print(f"\n[WARN] {ckpt} не содержит encoder.pt и/или hash_model.pt — "
                  "пропускаю joint_pair", flush=True)
        else:
            print(f"\n[run] joint_pair: загружаю encoder + hash из {ckpt}", flush=True)
            joint_encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
            joint_encoder.model.load_state_dict(torch.load(enc_path, map_location=device))
            joint_encoder.model.eval()
            joint_encoder.set_trainable(False)

            # Читаем meta для точной конфигурации хэша
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                joint_hash_cfg = HashingModelConfig(
                    input_dim=int(meta["hash_config"]["input_dim"]),
                    code_bits=int(meta["hash_config"]["code_bits"]),
                    hidden_dims=tuple(meta["hash_config"]["hidden_dims"]),
                    activation=str(meta["hash_config"]["activation"]),
                    dropout=float(meta["hash_config"]["dropout"]),
                    use_layer_norm=bool(meta["hash_config"]["use_layer_norm"]),
                )
                joint_code_bits = joint_hash_cfg.code_bits
                if joint_code_bits != args.code_bits:
                    print(f"    [INFO] code_bits в чекпойнте = {joint_code_bits}, "
                          f"в --code-bits = {args.code_bits}; для joint_pair используем {joint_code_bits}",
                          flush=True)
            else:
                joint_hash_cfg = hash_cfg  # fallback

            joint_hash = HashingMLP(joint_hash_cfg)
            joint_hash.load_state_dict(torch.load(hash_path, map_location=device))
            joint_hash.to(device)
            joint_hash.eval()

            # Re-encode корпус finetuned-энкодером (его эмбеддинги отличаются от pretrained)
            print("    [INFO] кодирую корпус finetuned-энкодером...", flush=True)
            started = time.perf_counter()
            joint_corpus_dense = joint_encoder.encode([r.text for r in corpus_records])
            print(f"        {time.perf_counter() - started:.1f}s", flush=True)

            # Dense_exact для finetuned-энкодера — отдельная планка
            ft_dense_pipeline = DenseSearchPipeline(preprocessor=preprocessor, encoder=joint_encoder)
            ft_dense_pipeline.records = list(corpus_records)
            ft_dense_pipeline.dense_embeddings = l2_normalize(joint_corpus_dense)
            ft_dense_pipeline.last_encode_time_ms = 0.0
            ft_dense_eval = SearchEvaluator(ft_dense_pipeline).evaluate(
                test_queries, search_config=cfg, warmup_queries=5
            )
            results["dense_exact_finetuned"] = ft_dense_eval
            results["dense_exact_finetuned"]["memory"] = memory_bytes_dense(n_docs, dense_dim)
            print(f"    [dense_exact_finetuned] recall@10={ft_dense_eval['recall@k']:.4f}  "
                  f"latency={ft_dense_eval['latency_ms']:.2f}ms", flush=True)

            # Сами коды joint-пары
            joint_codes = joint_hash.encode_embeddings(joint_corpus_dense, device=device)

            def joint_query_fn(dense_q: np.ndarray) -> np.ndarray:
                return joint_hash.encode_embeddings(dense_q, device=device)

            for use_rerank, key in [(False, "joint_pair_binary_only"), (True, "joint_pair_rerank")]:
                results[key] = evaluate_with_codes(
                    encoder=joint_encoder,
                    hash_codes_corpus=joint_codes,
                    hash_codes_query_fn=joint_query_fn,
                    corpus_records=corpus_records,
                    corpus_dense=joint_corpus_dense,
                    test_queries=test_queries,
                    code_bits=joint_code_bits,
                    cfg=cfg,
                    preprocessor=preprocessor,
                    device=device,
                    method_name="joint_pair",
                    rerank=use_rerank,
                )
                results[key]["memory"] = memory_bytes_binary(
                    n_docs, joint_code_bits, dense_dim=dense_dim if use_rerank else None
                )
                print(f"    [{key}] recall@10={results[key]['recall@k']:.4f}  "
                      f"latency={results[key]['latency_ms']:.2f}ms", flush=True)
            joint_pair_added = True

    # 5. Сводная таблица -----------------------------------------------------
    summary = {
        "args": vars(args),
        "device": device,
        "corpus_size": n_docs,
        "test_queries": len(test_queries),
        "code_bits": args.code_bits,
        "pq": {"subspaces": args.pq_subspaces, "centroids": args.pq_centroids,
                "code_bits_total": pq.code_bits},
        "fit_times_ms": timings,
        "models": results,
    }
    save_json(output_root / "comparison.json", summary)

    print("\n" + "=" * 110)
    print("Финальная сводка (test) — каждый метод в двух режимах:")
    print("=" * 110)
    header = (f"{'method':<28s}  {'mode':<12s}  {'recall@10':>9s}  {'ndcg@10':>8s}  "
              f"{'map@10':>8s}  {'lat (ms)':>10s}  {'mem (MB)':>10s}")
    print(header)
    print("-" * 110)
    rows = [
        ("dense_exact (pretrained)", "dense_exact", "exact"),
        ("LSH", "lsh_binary_only", "binary_only"),
        ("LSH", "lsh_rerank", "rerank"),
        ("ITQ", "itq_binary_only", "binary_only"),
        ("ITQ", "itq_rerank", "rerank"),
        (f"PQ (M={args.pq_subspaces}×K={args.pq_centroids})", "pq_binary_only", "binary_only"),
        (f"PQ (M={args.pq_subspaces}×K={args.pq_centroids})", "pq_rerank", "rerank"),
        ("hybrid_distilled (frozen)", "distill_binary_only", "binary_only"),
        ("hybrid_distilled (frozen)", "distill_rerank", "rerank"),
    ]
    if joint_pair_added:
        rows.extend([
            ("dense_exact (finetuned)", "dense_exact_finetuned", "exact"),
            ("joint_pair (FT enc + hash)", "joint_pair_binary_only", "binary_only"),
            ("joint_pair (FT enc + hash)", "joint_pair_rerank", "rerank"),
        ])
    for label, key, mode_label in rows:
        if key not in results:
            continue
        m = results[key]
        mem_mb = m.get("memory", {}).get("total_bytes", 0) / 1e6
        print(f"{label:<28s}  {mode_label:<12s}  {m['recall@k']:>9.4f}  {m['ndcg@k']:>8.4f}  "
              f"{m['map@k']:>8.4f}  {m['latency_ms']:>10.2f}  {mem_mb:>10.1f}")
    print("=" * 110)
    print(f"\n[OK] {output_root / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
