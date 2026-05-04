"""Совместное дообучение энкодера и хэш-модуля + полная оценка.

Запуск (локально на M-чипе):
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/run_joint_training.py \
        --data-dir data/real_300k \
        --output-dir artifacts/joint_run01 \
        --code-bits 128 --epochs 3

Запуск (на HPC через SLURM): см. scripts/slurm_joint_training.sh.

Что делает:
    1. Загружает корпус, триплеты, val/test queries.
    2. Создаёт TransformerSentenceEncoder (требует grad) + HashingMLP.
    3. Совместно обучает в течение N эпох с разными LR и cosine-шедулером,
       сохраняя чекпойнты после каждой эпохи.
    4. Между эпохами строит индекс и считает recall@k на val_queries.
    5. После обучения — финальная оценка на test_queries и сравнение с
       тремя baseline'ами:
            - dense_exact_pretrained (исходный энкодер, без обучения)
            - dense_exact_finetuned  (наш дообученный энкодер, без хэша)
            - hybrid_untrained       (исходный энкодер + случайный хэш)
            - hybrid_joint           (наш дообученный энкодер + хэш) <- результат
       Это даёт полную картину: куда переехал dense baseline и что добавил
       binary stage сверху.
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

import torch

from hybrid_search.core.config import (
    EncoderConfig,
    HashingModelConfig,
    IndexConfig,
    SearchConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import TransformerSentenceEncoder, build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.joint_training import (
    JointTrainer,
    JointTrainingConfig,
    TextTripletDataset,
)
from hybrid_search.retrieval.dense_search import DenseSearchPipeline
from hybrid_search.retrieval.evaluation import SearchEvaluator
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def auto_detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_encoder_config(model_name: str, batch_size: int, device: str, seed: int) -> EncoderConfig:
    return EncoderConfig(
        backend="transformers",
        model_name=model_name,
        embedding_dim=384,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
        seed=seed,
    )


def make_hashing_config(code_bits: int) -> HashingModelConfig:
    return HashingModelConfig(
        input_dim=384,
        code_bits=code_bits,
        hidden_dims=(256, 128),
        activation="gelu",
        dropout=0.1,
        use_layer_norm=True,
    )


def make_index_config(code_bits: int, backend: str = "hnswlib") -> IndexConfig:
    return IndexConfig(
        code_bits=code_bits,
        oversample_factor=4,
        binary_keep=code_bits,
        backend=backend,
        dense_metric="cosine",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
    )


def make_search_config(top_k: int = 10, oversample: int = 20) -> SearchConfig:
    return SearchConfig(strategy="coarse_rerank", top_k=top_k, oversample_factor=oversample)


# ---------------------------------------------------------------------------
# Eval-хелперы
# ---------------------------------------------------------------------------
def evaluate_hybrid(
    *,
    encoder: TransformerSentenceEncoder,
    hash_model: HashingMLP,
    corpus_records,
    corpus_dense,
    queries: list[dict],
    code_bits: int,
    device: str,
    cfg: SearchConfig,
    preprocessor: TextPreprocessor,
) -> dict:
    index = BinaryCodeIndex(
        code_bits=code_bits,
        backend="hnswlib",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=64,
    )
    binary_codes = hash_model.encode_embeddings(corpus_dense, device=device)
    index.add(records=corpus_records, binary_codes=binary_codes, dense_embeddings=corpus_dense)
    pipeline = HybridSearchPipeline(
        preprocessor=preprocessor,
        encoder=encoder,
        hash_model=hash_model,
        index=index,
        device=device,
    )
    evaluator = SearchEvaluator(pipeline)
    return evaluator.evaluate(queries, search_config=cfg, warmup_queries=5)


def evaluate_dense(
    *,
    encoder: TransformerSentenceEncoder,
    corpus_records,
    queries: list[dict],
    cfg: SearchConfig,
    preprocessor: TextPreprocessor,
) -> dict:
    pipeline = DenseSearchPipeline(preprocessor=preprocessor, encoder=encoder)
    pipeline.index_documents(corpus_records)
    evaluator = SearchEvaluator(pipeline)
    return evaluator.evaluate(queries, search_config=cfg, warmup_queries=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/real_300k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)

    # Модели
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--code-bits", type=int, default=128)

    # Гиперпараметры обучения
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--hash-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--quantization-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    # Логирование/чекпойнты
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--eval-every-n-epochs", type=int, default=1)
    parser.add_argument("--save-every-n-epochs", type=int, default=1)
    parser.add_argument("--skip-final-baselines", action="store_true",
                        help="Пропустить dense_exact (pretrained/finetuned) и hybrid_untrained")

    args = parser.parse_args()

    device = args.device or auto_detect_device()
    print(f"[INFO] device={device}, output={args.output_dir}", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # 1. Загрузка данных ------------------------------------------------------
    preprocessor = TextPreprocessor()
    encoder_cfg = make_encoder_config(args.transformer_model, args.batch_size, device, args.seed)
    encoder = TransformerSentenceEncoder(encoder_cfg, preprocessor=preprocessor)
    hash_model = HashingMLP(make_hashing_config(args.code_bits))

    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
    val_queries = load_jsonl(data_dir / "val_queries.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")
    print(
        f"[INFO] corpus={len(corpus_records)}  triplets={len(triplet_examples)}  "
        f"val={len(val_queries)}  test={len(test_queries)}",
        flush=True,
    )

    # 2. Pre-tokenization -----------------------------------------------------
    print("[INFO] токенизация триплетов...", flush=True)
    started = time.perf_counter()
    triplet_dataset = TextTripletDataset(triplet_examples, encoder, max_length=args.max_seq_length)
    print(f"    {len(triplet_dataset)} триплетов за {time.perf_counter() - started:.1f}s", flush=True)

    # 3. Eval-callback (между эпохами): hybrid recall на val_queries ----------
    val_cfg = make_search_config(top_k=10, oversample=20)
    test_cfg = make_search_config(top_k=10, oversample=20)

    def eval_callback(epoch: int) -> dict[str, Any]:
        print(f"[eval] epoch={epoch}: encoding corpus + evaluating hybrid on val...", flush=True)
        eval_started = time.perf_counter()
        # переключаем energetic ←- в inference на время оценки
        encoder.set_trainable(False)
        hash_model.eval()
        with torch.no_grad():
            corpus_dense = encoder.encode([r.text for r in corpus_records])
            metrics = evaluate_hybrid(
                encoder=encoder,
                hash_model=hash_model,
                corpus_records=corpus_records,
                corpus_dense=corpus_dense,
                queries=val_queries,
                code_bits=args.code_bits,
                device=device,
                cfg=val_cfg,
                preprocessor=preprocessor,
            )
        encoder.set_trainable(True)
        hash_model.train()
        elapsed = time.perf_counter() - eval_started
        print(
            f"    val recall@10={metrics['recall@k']:.4f}  ndcg@10={metrics['ndcg@k']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        return metrics

    # 4. Обучение -------------------------------------------------------------
    train_cfg = JointTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        hash_lr=args.hash_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        margin=args.margin,
        quantization_weight=args.quantization_weight,
        temperature=args.temperature,
        gradient_clip_norm=args.gradient_clip_norm,
        max_seq_length=args.max_seq_length,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        num_workers=args.num_workers,
    )
    trainer = JointTrainer(encoder=encoder, hash_model=hash_model, config=train_cfg, device=device)
    print("[INFO] запускаю совместное обучение...", flush=True)
    train_started = time.perf_counter()
    history = trainer.fit(
        triplet_dataset,
        eval_callback=eval_callback,
        checkpoint_dir=output_root / "checkpoints",
    )
    train_elapsed = time.perf_counter() - train_started
    print(f"[OK] обучение завершено за {train_elapsed/60:.1f} мин", flush=True)

    save_json(output_root / "training_history.json", {
        "epochs": history.epochs,
        "eval": history.eval,
        "train_time_s": round(train_elapsed, 1),
    })

    # 5. Финальная оценка -----------------------------------------------------
    print("\n[INFO] финальная оценка...", flush=True)
    encoder.set_trainable(False)
    hash_model.eval()
    with torch.no_grad():
        corpus_dense_finetuned = encoder.encode([r.text for r in corpus_records])

    final = {}
    print("[final] hybrid_joint (наш метод) on test...", flush=True)
    final["hybrid_joint"] = evaluate_hybrid(
        encoder=encoder, hash_model=hash_model,
        corpus_records=corpus_records, corpus_dense=corpus_dense_finetuned,
        queries=test_queries, code_bits=args.code_bits, device=device,
        cfg=test_cfg, preprocessor=preprocessor,
    )
    print(f"    recall@10={final['hybrid_joint']['recall@k']:.4f}  ndcg@10={final['hybrid_joint']['ndcg@k']:.4f}", flush=True)

    if not args.skip_final_baselines:
        print("[final] dense_exact (finetuned encoder)...", flush=True)
        final["dense_exact_finetuned"] = evaluate_dense(
            encoder=encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=test_cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_finetuned']['recall@k']:.4f}", flush=True)

        print("[final] hybrid_untrained (random hash на finetuned encoder)...", flush=True)
        random_hash = HashingMLP(make_hashing_config(args.code_bits)).to(device)
        random_hash.eval()
        with torch.no_grad():
            final["hybrid_untrained"] = evaluate_hybrid(
                encoder=encoder, hash_model=random_hash,
                corpus_records=corpus_records, corpus_dense=corpus_dense_finetuned,
                queries=test_queries, code_bits=args.code_bits, device=device,
                cfg=test_cfg, preprocessor=preprocessor,
            )
        print(f"    recall@10={final['hybrid_untrained']['recall@k']:.4f}", flush=True)

        # Pretrained dense — нужен другой энкодер (исходные веса)
        print("[final] dense_exact (pretrained encoder, без обучения)...", flush=True)
        pretrained_encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
        final["dense_exact_pretrained"] = evaluate_dense(
            encoder=pretrained_encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=test_cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_pretrained']['recall@k']:.4f}", flush=True)

    # 6. Сводка ---------------------------------------------------------------
    summary = {
        "args": vars(args),
        "device": device,
        "training": {
            "config": train_cfg.to_dict(),
            "epochs_history": history.epochs,
            "eval_history": history.eval,
            "train_time_s": round(train_elapsed, 1),
        },
        "models": final,
    }
    save_json(output_root / "summary.json", summary)
    print("\n" + "=" * 78)
    print("Финальная сводка (test):")
    print("=" * 78)
    for name, m in final.items():
        print(
            f"  {name:30s}  recall@10={m['recall@k']:.4f}  ndcg@10={m['ndcg@k']:.4f}  "
            f"map@10={m['map@k']:.4f}  latency={m['latency_ms']:.2f}ms"
        )
    print(f"\n[OK] {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
