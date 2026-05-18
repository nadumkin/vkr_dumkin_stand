"""Joint encoder + hash distillation training (teacher-student).

Архитектура:
    teacher = pretrained MiniLM (frozen, только для генерации target эмбеддингов
              на старте обучения; в самом цикле не нужен)
    student = копия того же энкодера, обучается
    hash    = MLP-голова поверх student, обучается

Loss = MSE между бинарным cosine ученика и dense cosine учителя
       + штраф квантизации.

Запуск (на M-чипе, ~30–45 мин):
    cd /Users/nikita/Documents/Docs/ВКР/src
    caffeinate -i .venv/bin/python scripts/run_joint_distillation.py \\
        --data-dir data/real_300k \\
        --output-dir artifacts/joint_distill/base \\
        --code-bits 256 --epochs 3 --batch-size 128 \\
        --quantization-weight 0.01 --encoder-lr 5e-6 --hash-lr 1e-3
"""
from __future__ import annotations

import argparse
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

from hybrid_search.core.config import EncoderConfig, HashingModelConfig, SearchConfig
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.io.data import load_jsonl, load_similarity_examples, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import TransformerSentenceEncoder, build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.joint_training import (
    JointDistillationTrainer,
    JointTrainingConfig,
    TextTripletDataset,
)
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


def make_encoder_cfg(model_name: str, device: str, batch_size: int) -> EncoderConfig:
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
    raw = (raw or "").strip()
    if not raw:
        return ()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def evaluate_hybrid(*, encoder, hash_model, corpus_records, corpus_dense,
                    queries, code_bits, device, cfg, preprocessor):
    index = BinaryCodeIndex(code_bits=code_bits, backend="hnswlib",
                            hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=64)
    binary_codes = hash_model.encode_embeddings(corpus_dense, device=device)
    index.add(records=corpus_records, binary_codes=binary_codes, dense_embeddings=corpus_dense)
    pipeline = HybridSearchPipeline(preprocessor=preprocessor, encoder=encoder,
                                     hash_model=hash_model, index=index, device=device)
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

    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--code-bits", type=int, default=256)

    # Гиперпараметры
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-lr", type=float, default=5e-6,
                        help="Очень мягкий fine-tune энкодера (учитель — анкер)")
    parser.add_argument("--hash-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--quantization-weight", type=float, default=0.01)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=None,
                        help="Если задан — линейный отжиг α в tanh(α·z) от --temperature к этому "
                             "значению. Например, 1.0 → 10.0 (HashNet-style).")
    # Архитектура hash-головы (по аналогии с run_distillation_training.py)
    parser.add_argument("--hidden-dims", default="256,128",
                        help="Comma-separated размеры скрытых слоёв; '' для pure linear.")
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-layer-norm", action="store_true",
                        help="Отключить LayerNorm в скрытых слоях.")
    # ITQ-инициализация выходного Linear
    parser.add_argument("--itq-init", action="store_true",
                        help="Перед обучением инициализировать выходной Linear ITQ-ротацией, "
                             "вычисленной на скрытых выходах backbone'а от teacher-эмбеддингов.")
    parser.add_argument("--itq-iter", type=int, default=50)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--eval-every-n-epochs", type=int, default=1)
    parser.add_argument("--save-every-n-epochs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=20)
    parser.add_argument("--skip-final-baselines", action="store_true")

    args = parser.parse_args()
    device = args.device or auto_detect_device()
    print(f"[INFO] device={device}, output={args.output_dir}", flush=True)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    preprocessor = TextPreprocessor()
    encoder_cfg = make_encoder_cfg(args.transformer_model, device, args.batch_size)

    # 1. Teacher: посчитать эмбеддинги для триплетов и корпуса --------------
    print("[INFO] Готовлю TEACHER (frozen pretrained encoder)...", flush=True)
    teacher_encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
    triplet_examples = load_similarity_examples(data_dir / "train_triplets.jsonl")
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")

    started = time.perf_counter()
    print(f"[INFO] teacher: кодирую {len(triplet_examples)} триплетов (q+p+n)...", flush=True)
    queries_text = [ex.query_text for ex in triplet_examples]
    positives_text = [ex.positive_text for ex in triplet_examples]
    negatives_text = [(ex.negative_text or ex.positive_text) for ex in triplet_examples]
    teacher_q = teacher_encoder.encode(queries_text)
    teacher_p = teacher_encoder.encode(positives_text)
    teacher_n = teacher_encoder.encode(negatives_text)
    print(f"    triplets encoded в {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    print(f"[INFO] teacher: кодирую корпус ({len(corpus_records)} текстов)...", flush=True)
    corpus_dense_teacher = teacher_encoder.encode([r.text for r in corpus_records])
    print(f"    corpus encoded в {time.perf_counter() - started:.1f}s", flush=True)

    # 2. Pre-tokenization ----------------------------------------------------
    print("[INFO] токенизация триплетов и подвешивание teacher embeddings...", flush=True)
    student_encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
    triplet_dataset = TextTripletDataset(
        triplet_examples,
        student_encoder,  # tokenizer один и тот же
        max_length=args.max_seq_length,
        teacher_embeddings=(teacher_q, teacher_p, teacher_n),
    )
    print(f"    {len(triplet_dataset)} триплетов готовы", flush=True)

    # 3. Hash-модель ---------------------------------------------------------
    hash_cfg = make_hashing_cfg(
        code_bits=args.code_bits,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    )
    print(f"[INFO] hash arch: hidden_dims={hash_cfg.hidden_dims}, "
          f"LN={hash_cfg.use_layer_norm}, act={hash_cfg.activation}, "
          f"dropout={hash_cfg.dropout}", flush=True)
    hash_model = HashingMLP(hash_cfg)

    # 3a. Опциональная ITQ-инициализация выходного Linear --------------------
    if args.itq_init:
        print("[INFO] ITQ-init выходного Linear на скрытых выходах backbone'а...", flush=True)
        itq_started = time.perf_counter()
        from hybrid_search.models.baselines import ITQHasher

        # Для ITQ-init нужны hidden-выходы backbone'а на корпусе. Используем
        # teacher embeddings (они ещё не записаны в переменную corpus_dense здесь —
        # вытащим из заранее посчитанной teacher-копии корпуса).
        hash_model.to(device)
        hash_model.eval()
        import numpy as _np
        with torch.no_grad():
            BATCH = 1024
            hidden_chunks = []
            for i in range(0, len(corpus_dense_teacher), BATCH):
                chunk = torch.from_numpy(
                    corpus_dense_teacher[i : i + BATCH].astype(_np.float32, copy=False)
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
        composed_W = (itq.pca @ itq.rotation).T.astype(_np.float32)
        composed_b = (-(itq.mean.flatten() @ itq.pca @ itq.rotation)).astype(_np.float32)
        with torch.no_grad():
            hash_model.output.weight.data.copy_(torch.from_numpy(composed_W))
            hash_model.output.bias.data.copy_(torch.from_numpy(composed_b))
        hash_model.train()
        print(f"    ITQ-init готов за {time.perf_counter() - itq_started:.1f}s "
              f"(hidden_dim={hidden_outputs.shape[1]} → code_bits={args.code_bits})", flush=True)

    # 4. Eval callback (val между эпохами) -----------------------------------
    val_queries = load_jsonl(data_dir / "val_queries.jsonl")
    val_cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k, oversample_factor=args.oversample)
    test_cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k, oversample_factor=args.oversample)

    def eval_callback(epoch: int) -> dict[str, Any]:
        eval_started = time.perf_counter()
        student_encoder.set_trainable(False)
        hash_model.eval()
        with torch.no_grad():
            corpus_dense_student = student_encoder.encode([r.text for r in corpus_records])
            metrics = evaluate_hybrid(
                encoder=student_encoder, hash_model=hash_model,
                corpus_records=corpus_records, corpus_dense=corpus_dense_student,
                queries=val_queries, code_bits=args.code_bits, device=device,
                cfg=val_cfg, preprocessor=preprocessor,
            )
        student_encoder.set_trainable(True)
        hash_model.train()
        elapsed = time.perf_counter() - eval_started
        print(
            f"[eval] epoch={epoch}: val recall@10={metrics['recall@k']:.4f}  "
            f"ndcg@10={metrics['ndcg@k']:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )
        return metrics

    # 5. Тренировка ----------------------------------------------------------
    train_cfg = JointTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        hash_lr=args.hash_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        margin=0.0,                       # не используется в distillation
        quantization_weight=args.quantization_weight,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        gradient_clip_norm=args.gradient_clip_norm,
        max_seq_length=args.max_seq_length,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_epochs=args.eval_every_n_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        num_workers=args.num_workers,
    )
    trainer = JointDistillationTrainer(
        encoder=student_encoder,
        hash_model=hash_model,
        config=train_cfg,
        device=device,
        distillation_weight=args.distillation_weight,
    )
    if args.temperature_end is not None:
        print(f"[INFO] temperature annealing: α(0)={args.temperature} → α(end)={args.temperature_end}",
              flush=True)
    print("[INFO] запускаю joint distillation training...", flush=True)
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

    # 6. Финальная оценка ----------------------------------------------------
    print("\n[INFO] финальная оценка...", flush=True)
    student_encoder.set_trainable(False)
    hash_model.eval()
    with torch.no_grad():
        corpus_dense_student = student_encoder.encode([r.text for r in corpus_records])

    final = {}
    print("[final] hybrid_joint_distill (наш метод)...", flush=True)
    final["hybrid_joint_distill"] = evaluate_hybrid(
        encoder=student_encoder, hash_model=hash_model,
        corpus_records=corpus_records, corpus_dense=corpus_dense_student,
        queries=test_queries, code_bits=args.code_bits, device=device,
        cfg=test_cfg, preprocessor=preprocessor,
    )
    print(f"    recall@10={final['hybrid_joint_distill']['recall@k']:.4f}  "
          f"ndcg@10={final['hybrid_joint_distill']['ndcg@k']:.4f}", flush=True)

    if not args.skip_final_baselines:
        print("[final] dense_exact_finetuned (только student-энкодер)...", flush=True)
        final["dense_exact_finetuned"] = evaluate_dense(
            encoder=student_encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=test_cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_finetuned']['recall@k']:.4f}", flush=True)

        print("[final] hybrid_untrained (random hash на student-энкодере)...", flush=True)
        random_hash = HashingMLP(hash_cfg).to(device)
        random_hash.eval()
        with torch.no_grad():
            final["hybrid_untrained"] = evaluate_hybrid(
                encoder=student_encoder, hash_model=random_hash,
                corpus_records=corpus_records, corpus_dense=corpus_dense_student,
                queries=test_queries, code_bits=args.code_bits, device=device,
                cfg=test_cfg, preprocessor=preprocessor,
            )
        print(f"    recall@10={final['hybrid_untrained']['recall@k']:.4f}", flush=True)

        print("[final] dense_exact_pretrained (исходный teacher)...", flush=True)
        final["dense_exact_pretrained"] = evaluate_dense(
            encoder=teacher_encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=test_cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_pretrained']['recall@k']:.4f}", flush=True)

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
        print(f"  {name:32s}  recall@10={m['recall@k']:.4f}  ndcg@10={m['ndcg@k']:.4f}  "
              f"map@10={m['map@k']:.4f}  latency={m['latency_ms']:.2f}ms")
    print(f"\n[OK] {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
