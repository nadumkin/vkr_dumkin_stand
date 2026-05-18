"""Hash-only обучение через self-distillation.

В отличие от triplet и joint, здесь:
    - Энкодер ЗАМОРОЖЕН (frozen). Используются только pretrained-эмбеддинги.
    - Хэш-голова обучается так, чтобы её бинарный код воспроизводил
      косинусное подобие dense-векторов (in-batch MSE по матрице 3B × 3B).
    - Это самый плотный сигнал из протестированных нами лоссов и наименее
      рискованный (нет drift'а энкодера).

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/run_distillation_training.py \\
        --data-dir data/real_300k \\
        --output-dir artifacts/distill/base \\
        --code-bits 128 --epochs 5 --batch-size 256

Время на M4 Pro Max: ~10-15 минут (включая один проход энкодера по корпусу
и финальные baseline'ы). Сам цикл distillation — секунды на эпоху, потому
что обучается только маленькая MLP, а тяжёлый трансформер уже не нужен.
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
    SearchConfig,
    TrainingConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import build_encoder
from hybrid_search.models.hashing import HashingMLP, save_hash_checkpoint
from hybrid_search.models.training import DistillationTrainer, build_triplet_embedding_dataset
from hybrid_search.retrieval.dense_search import DenseSearchPipeline
from hybrid_search.retrieval.evaluation import SearchEvaluator
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline


def auto_detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_encoder_cfg(model_name: str, device: str, batch_size: int = 128) -> EncoderConfig:
    return EncoderConfig(
        backend="transformers",
        model_name=model_name,
        embedding_dim=384,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
        seed=17,
    )


def make_hashing_cfg(
    code_bits: int,
    hidden_dims: tuple[int, ...] = (256, 128),
    activation: str = "gelu",
    dropout: float = 0.1,
    use_layer_norm: bool = True,
) -> HashingModelConfig:
    return HashingModelConfig(
        input_dim=384,
        code_bits=code_bits,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )


def parse_hidden_dims(raw: str) -> tuple[int, ...]:
    """Парсит '256,128' или '768' или '' (пустая строка для pure linear)."""
    raw = (raw or "").strip()
    if not raw:
        return ()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def evaluate_hybrid(*, encoder, hash_model, corpus_records, corpus_dense,
                    queries, code_bits, device, cfg, preprocessor):
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
        preprocessor=preprocessor, encoder=encoder,
        hash_model=hash_model, index=index, device=device,
    )
    return SearchEvaluator(pipeline).evaluate(queries, search_config=cfg, warmup_queries=5)


def evaluate_dense(*, encoder, corpus_records, queries, cfg, preprocessor):
    pipeline = DenseSearchPipeline(preprocessor=preprocessor, encoder=encoder)
    pipeline.index_documents(corpus_records)
    return SearchEvaluator(pipeline).evaluate(queries, search_config=cfg, warmup_queries=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/real_300k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)

    # Модель
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--code-bits", type=int, default=128)
    # Архитектура hash-головы
    parser.add_argument("--hidden-dims", default="256,128",
                        help="Comma-separated размеры скрытых слоёв; '' для чистого Linear.")
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-layer-norm", action="store_true",
                        help="Отключить LayerNorm в скрытых слоях.")

    # Гиперпараметры обучения
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Бóльшие батчи дают плотнее матрицу подобий — обычно лучше для distillation")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quantization-weight", type=float, default=0.1)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=None,
                        help="Если задан — линейное расписание α от --temperature до этого значения "
                             "(HashNet-style). Например, 1.0 → 10.0.")
    parser.add_argument("--itq-init", action="store_true",
                        help="Инициализировать output Linear ITQ-ротацией, посчитанной "
                             "на скрытых выходах backbone'а корпуса.")
    parser.add_argument("--itq-iter", type=int, default=50)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

    # Eval
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=20)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Только hybrid_distilled (без dense_exact и hybrid_untrained)")

    args = parser.parse_args()

    device = args.device or auto_detect_device()
    print(f"[INFO] device={device}", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    preprocessor = TextPreprocessor()

    # 1. Энкодер (frozen) -----------------------------------------------------
    encoder_cfg = make_encoder_cfg(args.transformer_model, device)
    encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
    print(f"[INFO] энкодер загружен: {args.transformer_model} на {device}", flush=True)

    # 2. Данные ---------------------------------------------------------------
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")
    print(
        f"[INFO] corpus={len(corpus_records)}  triplets={len(triplet_examples)}  "
        f"test={len(test_queries)}",
        flush=True,
    )

    # 3. Кодирование триплетов (один раз — энкодер заморожен) -----------------
    print("[INFO] кодирую триплеты pretrained-энкодером (один раз)...", flush=True)
    started = time.perf_counter()
    triplet_dataset = build_triplet_embedding_dataset(triplet_examples, encoder, preprocessor=preprocessor)
    print(f"    {time.perf_counter() - started:.1f}s", flush=True)

    # 4. Кодирование корпуса (один раз) ---------------------------------------
    print("[INFO] кодирую корпус pretrained-энкодером (один раз)...", flush=True)
    started = time.perf_counter()
    corpus_dense = encoder.encode([r.text for r in corpus_records])
    print(f"    {len(corpus_records)} текстов за {time.perf_counter() - started:.1f}s", flush=True)

    # 5. Distillation training ------------------------------------------------
    hash_cfg = make_hashing_cfg(
        code_bits=args.code_bits,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )
    print(f"[INFO] hash arch: hidden_dims={hash_cfg.hidden_dims}, "
          f"LN={hash_cfg.use_layer_norm}, act={hash_cfg.activation}, dropout={hash_cfg.dropout}", flush=True)
    hash_model = HashingMLP(hash_cfg)

    # Опциональная ITQ-инициализация выходного Linear --------------------------
    if args.itq_init:
        print("[INFO] ITQ-init выходного Linear на скрытых выходах backbone'а...", flush=True)
        itq_started = time.perf_counter()
        from hybrid_search.models.baselines import ITQHasher
        import numpy as _np

        hash_model.to(device)
        hash_model.eval()
        with torch.no_grad():
            BATCH = 1024
            hidden_chunks = []
            for i in range(0, len(corpus_dense), BATCH):
                chunk = torch.from_numpy(
                    corpus_dense[i : i + BATCH].astype(_np.float32, copy=False)
                ).to(device)
                hidden_chunks.append(hash_model.backbone(chunk).cpu().numpy())
            hidden_outputs = _np.vstack(hidden_chunks)

        itq = ITQHasher(
            input_dim=hidden_outputs.shape[1],
            code_bits=args.code_bits,
            n_iter=args.itq_iter,
            seed=args.seed,
        )
        itq.fit(hidden_outputs)

        # ITQ pipeline: y = (x - mean) @ pca @ rotation, sign(y) → bits
        # Output Linear делает: y = x @ W^T + b  ⇒  W = (pca @ rotation).T,  b = -mean @ pca @ rotation
        composed_W = (itq.pca @ itq.rotation).T.astype(_np.float32)              # (code_bits, hidden_dim)
        composed_b = (-(itq.mean.flatten() @ itq.pca @ itq.rotation)).astype(_np.float32)
        with torch.no_grad():
            hash_model.output.weight.data.copy_(torch.from_numpy(composed_W))
            hash_model.output.bias.data.copy_(torch.from_numpy(composed_b))
        hash_model.train()
        print(f"    ITQ-init готов за {time.perf_counter() - itq_started:.1f}s "
              f"(hidden_dim={hidden_outputs.shape[1]} → code_bits={args.code_bits})", flush=True)

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=0.0,                    # distillation не использует margin
        quantization_weight=args.quantization_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        device=device,
    )
    trainer = DistillationTrainer(hash_model, train_cfg, distillation_weight=args.distillation_weight)
    if args.temperature_end is not None:
        print(f"[INFO] temperature annealing: α(0) = {args.temperature} → α(end) = {args.temperature_end}",
              flush=True)
    print("[INFO] запускаю distillation training...", flush=True)
    started = time.perf_counter()
    history = trainer.fit(triplet_dataset)
    train_elapsed = time.perf_counter() - started
    print(f"[OK] обучение завершено за {train_elapsed:.1f}s", flush=True)
    for h in history:
        alpha_part = f"  α_end={h['alpha_end_of_epoch']:.2f}" if "alpha_end_of_epoch" in h else ""
        print(
            f"    epoch {h['epoch']}: loss={h['loss']:.4f}  "
            f"distill={h['distillation_loss']:.4f}  quant={h['quantization_loss']:.4f}{alpha_part}",
            flush=True,
        )

    save_json(output_root / "training_history.json", {"history": history,
                                                       "train_time_s": round(train_elapsed, 1)})

    # Сохраняем веса в checkpoint-формате (с конфигурацией) для последующей загрузки.
    save_hash_checkpoint(
        output_root / "hash_model.pt",
        hash_model,
        extra={
            "training_config": asdict(train_cfg),
            "distillation_weight": args.distillation_weight,
            "training_kind": "distillation_hash_only",
            "encoder_model_name": encoder_cfg.model_name,
        },
    )

    # 6. Финальная оценка -----------------------------------------------------
    cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k, oversample_factor=args.oversample)
    final = {}

    print("\n[final] hybrid_distilled (наш метод)...", flush=True)
    final["hybrid_distilled"] = evaluate_hybrid(
        encoder=encoder, hash_model=hash_model,
        corpus_records=corpus_records, corpus_dense=corpus_dense,
        queries=test_queries, code_bits=args.code_bits, device=device,
        cfg=cfg, preprocessor=preprocessor,
    )
    print(
        f"    recall@10={final['hybrid_distilled']['recall@k']:.4f}  "
        f"ndcg@10={final['hybrid_distilled']['ndcg@k']:.4f}",
        flush=True,
    )

    if not args.skip_baselines:
        print("[final] hybrid_untrained (random hash)...", flush=True)
        random_hash = HashingMLP(make_hashing_cfg(args.code_bits)).to(device)
        random_hash.eval()
        with torch.no_grad():
            final["hybrid_untrained"] = evaluate_hybrid(
                encoder=encoder, hash_model=random_hash,
                corpus_records=corpus_records, corpus_dense=corpus_dense,
                queries=test_queries, code_bits=args.code_bits, device=device,
                cfg=cfg, preprocessor=preprocessor,
            )
        print(
            f"    recall@10={final['hybrid_untrained']['recall@k']:.4f}  "
            f"ndcg@10={final['hybrid_untrained']['ndcg@k']:.4f}",
            flush=True,
        )

        print("[final] dense_exact_pretrained...", flush=True)
        final["dense_exact_pretrained"] = evaluate_dense(
            encoder=encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=cfg, preprocessor=preprocessor,
        )
        print(
            f"    recall@10={final['dense_exact_pretrained']['recall@k']:.4f}  "
            f"ndcg@10={final['dense_exact_pretrained']['ndcg@k']:.4f}",
            flush=True,
        )

    # 7. Сводка ---------------------------------------------------------------
    summary = {
        "args": vars(args),
        "device": device,
        "training": {
            "config": asdict(train_cfg),
            "history": history,
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
