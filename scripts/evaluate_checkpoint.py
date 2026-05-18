"""Финальная оценка из сохранённого чекпойнта joint-training.

Используется когда обучение было прервано или нужно посчитать test-метрики
для конкретной эпохи, не повторяя весь прогон.

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/evaluate_checkpoint.py \
        --checkpoint artifacts/joint/base_3ep/checkpoints/checkpoint_epoch1 \
        --output artifacts/joint/base_3ep/summary_from_epoch1.json \
        --data-dir data/real_300k

Что считает:
    - dense_exact_pretrained — исходный энкодер (новая загрузка с HF), без обучения
    - dense_exact_finetuned  — энкодер из чекпойнта, без хэша
    - hybrid_untrained       — энкодер из чекпойнта + случайный хэш
    - hybrid_joint           — энкодер из чекпойнта + хэш из чекпойнта (итог)

Время: ~12–15 минут на M-чипе (4 раза кодирует 224K корпус).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from hybrid_search.io.data import load_jsonl, load_text_records
from hybrid_search.io.storage import save_json
from hybrid_search.models.encoders import TransformerSentenceEncoder, build_encoder
from hybrid_search.models.hashing import HashingMLP
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
    parser.add_argument("--checkpoint", required=True,
                        help="Папка с checkpoint_epochN (содержит encoder.pt, hash_model.pt, meta.json)")
    parser.add_argument("--output", required=True, help="Куда сохранить summary.json")
    parser.add_argument("--data-dir", default="data/real_300k")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=20)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Пропустить dense_exact_pretrained / hybrid_untrained / dense_exact_finetuned")
    args = parser.parse_args()

    device = args.device or auto_detect_device()
    checkpoint_dir = Path(args.checkpoint)
    if not (checkpoint_dir / "meta.json").exists():
        sys.stderr.write(f"[ERROR] не найден {checkpoint_dir / 'meta.json'}\n")
        return 1

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    code_bits = int(meta["hash_config"]["code_bits"])
    encoder_model_name = meta["encoder_model_name"]
    print(f"[INFO] device={device}, encoder={encoder_model_name}, code_bits={code_bits}", flush=True)

    # 1. Подготовка ----------------------------------------------------------
    preprocessor = TextPreprocessor()
    encoder_cfg = EncoderConfig(
        backend="transformers", model_name=encoder_model_name,
        embedding_dim=int(meta.get("embedding_dim", 384)),
        batch_size=128, normalize_embeddings=True, device=device, seed=17,
    )
    encoder = TransformerSentenceEncoder(encoder_cfg, preprocessor=preprocessor)
    print(f"[INFO] загружаю веса энкодера из {checkpoint_dir / 'encoder.pt'}", flush=True)
    encoder.model.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    encoder.model.eval()
    encoder.set_trainable(False)

    hash_cfg = HashingModelConfig(
        input_dim=int(meta["hash_config"]["input_dim"]),
        code_bits=code_bits,
        hidden_dims=tuple(meta["hash_config"]["hidden_dims"]),
        activation=str(meta["hash_config"]["activation"]),
        dropout=float(meta["hash_config"]["dropout"]),
        use_layer_norm=bool(meta["hash_config"]["use_layer_norm"]),
    )
    hash_model = HashingMLP(hash_cfg)
    print(f"[INFO] загружаю веса хэш-MLP из {checkpoint_dir / 'hash_model.pt'}", flush=True)
    hash_model.load_state_dict(torch.load(checkpoint_dir / "hash_model.pt", map_location=device))
    hash_model.to(device)
    hash_model.eval()

    # 2. Данные ---------------------------------------------------------------
    data_dir = Path(args.data_dir)
    corpus_records = load_text_records(data_dir / "corpus.jsonl")
    test_queries = load_jsonl(data_dir / "test_queries.jsonl")
    print(f"[INFO] corpus={len(corpus_records)}, test_queries={len(test_queries)}", flush=True)

    cfg = SearchConfig(strategy="coarse_rerank", top_k=args.top_k,
                       oversample_factor=args.oversample)

    # 3. Кодирование корпуса finetuned-энкодером (используется для 2 моделей)
    print("[INFO] кодирую корпус finetuned-энкодером...", flush=True)
    started = time.perf_counter()
    with torch.no_grad():
        corpus_dense_finetuned = encoder.encode([r.text for r in corpus_records])
    print(f"    {time.perf_counter() - started:.1f}s", flush=True)

    final = {}

    # 4. hybrid_joint (наш итог) ---------------------------------------------
    print("[final] hybrid_joint...", flush=True)
    final["hybrid_joint"] = evaluate_hybrid(
        encoder=encoder, hash_model=hash_model,
        corpus_records=corpus_records, corpus_dense=corpus_dense_finetuned,
        queries=test_queries, code_bits=code_bits, device=device,
        cfg=cfg, preprocessor=preprocessor,
    )
    print(f"    recall@10={final['hybrid_joint']['recall@k']:.4f}  "
          f"ndcg@10={final['hybrid_joint']['ndcg@k']:.4f}", flush=True)

    if not args.skip_baselines:
        # 5. dense_exact_finetuned --------------------------------------------
        print("[final] dense_exact_finetuned...", flush=True)
        final["dense_exact_finetuned"] = evaluate_dense(
            encoder=encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_finetuned']['recall@k']:.4f}", flush=True)

        # 6. hybrid_untrained -------------------------------------------------
        print("[final] hybrid_untrained...", flush=True)
        random_hash = HashingMLP(hash_cfg).to(device)
        random_hash.eval()
        with torch.no_grad():
            final["hybrid_untrained"] = evaluate_hybrid(
                encoder=encoder, hash_model=random_hash,
                corpus_records=corpus_records, corpus_dense=corpus_dense_finetuned,
                queries=test_queries, code_bits=code_bits, device=device,
                cfg=cfg, preprocessor=preprocessor,
            )
        print(f"    recall@10={final['hybrid_untrained']['recall@k']:.4f}", flush=True)

        # 7. dense_exact_pretrained -------------------------------------------
        print("[final] dense_exact_pretrained (новая загрузка с HF)...", flush=True)
        pretrained_encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
        final["dense_exact_pretrained"] = evaluate_dense(
            encoder=pretrained_encoder, corpus_records=corpus_records,
            queries=test_queries, cfg=cfg, preprocessor=preprocessor,
        )
        print(f"    recall@10={final['dense_exact_pretrained']['recall@k']:.4f}", flush=True)

    # 8. Сводка ---------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_dir),
        "device": device,
        "data_dir": str(data_dir),
        "models": final,
        "meta": meta,
    }
    save_json(output_path, summary)
    print("\n" + "=" * 78)
    print(f"Финальная сводка из {checkpoint_dir.name}:")
    print("=" * 78)
    for name, m in final.items():
        print(f"  {name:30s}  recall@10={m['recall@k']:.4f}  ndcg@10={m['ndcg@k']:.4f}  "
              f"map@10={m['map@k']:.4f}  latency={m['latency_ms']:.2f}ms")
    print(f"\n[OK] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
