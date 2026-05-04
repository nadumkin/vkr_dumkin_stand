"""Быстрая проверка новой инструментации (build_time, memory, latency_breakdown).

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/smoke_check_metrics.py

Без долгих прогонов: использует data/mini (несколько документов) и hashing-энкодер,
не требует transformers/torch на heavy-loop, но всё равно проверяет, что все новые
поля корректно появляются в результате.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_search.core.config import (
    EncoderConfig,
    HashingModelConfig,
    IndexConfig,
    SearchConfig,
    TrainingConfig,
)
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.core.schemas import TextRecord
from hybrid_search.experiments.comparison import compare_with_baselines
from hybrid_search.io.data import load_jsonl, load_text_records
from hybrid_search.models.encoders import build_encoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.retrieval.dense_search import DenseSearchPipeline
from hybrid_search.retrieval.evaluation import SearchEvaluator
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline


def stage_check_unit() -> None:
    """Проверяем, что HybridSearchPipeline и DenseSearchPipeline пишут last_search_timings."""
    print("[1/3] unit: per-search stage timings...")
    preprocessor = TextPreprocessor()
    encoder_cfg = EncoderConfig(backend="hashing", embedding_dim=64, batch_size=8, seed=23)
    encoder = build_encoder(encoder_cfg, preprocessor=preprocessor)
    hash_model = HashingMLP(HashingModelConfig(input_dim=64, code_bits=32, hidden_dims=(48, 32), dropout=0.0))
    records = [
        TextRecord(record_id=f"d{i}", text=f"document number {i} about cats and dogs")
        for i in range(20)
    ]
    index = BinaryCodeIndex(code_bits=32, backend="numpy")
    pipeline = HybridSearchPipeline(preprocessor, encoder, hash_model, index=index, device="cpu")
    pipeline.index_documents(records)
    print(
        f"    index.last_build_time_ms={index.last_build_time_ms:.4f}; "
        f"pipeline.last_encode_time_ms={pipeline.last_encode_time_ms:.4f}; "
        f"pipeline.last_hashing_time_ms={pipeline.last_hashing_time_ms:.4f}"
    )
    pipeline.search("cats", config=SearchConfig(top_k=3, oversample_factor=3))
    timings = pipeline.last_search_timings
    print(f"    hybrid stage: {timings}")
    for k in ("query_encode_ms", "candidate_selection_ms", "rerank_ms", "total_ms"):
        assert k in timings, f"missing key: {k}"

    dense_pipeline = DenseSearchPipeline(preprocessor=preprocessor, encoder=encoder)
    dense_pipeline.index_documents(records)
    dense_pipeline.search("cats", config=SearchConfig(top_k=3))
    print(f"    dense stage:  {dense_pipeline.last_search_timings}")
    print(f"    dense memory: {dense_pipeline.memory_breakdown()}")
    print(f"    hybrid memory: {index.memory_breakdown()}")
    print("    OK")


def stage_check_evaluator() -> None:
    """Проверяем, что SearchEvaluator агрегирует latency_breakdown_ms."""
    print("[2/3] evaluator: aggregated latency_breakdown_ms...")
    preprocessor = TextPreprocessor()
    encoder = build_encoder(EncoderConfig(backend="hashing", embedding_dim=64, seed=23), preprocessor=preprocessor)
    hash_model = HashingMLP(HashingModelConfig(input_dim=64, code_bits=32, hidden_dims=(48, 32), dropout=0.0))
    records = [TextRecord(record_id=f"d{i}", text=f"sample text item {i}") for i in range(15)]
    index = BinaryCodeIndex(code_bits=32, backend="numpy")
    pipeline = HybridSearchPipeline(preprocessor, encoder, hash_model, index=index, device="cpu")
    pipeline.index_documents(records)
    evaluator = SearchEvaluator(pipeline)
    queries = [{"query": f"sample {i}", "relevant_ids": [f"d{i}"]} for i in range(5)]
    result = evaluator.evaluate(queries, search_config=SearchConfig(top_k=3, oversample_factor=3), warmup_queries=1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert "latency_breakdown_ms" in result
    for k in ("query_encode_ms", "candidate_selection_ms", "rerank_ms"):
        assert k in result["latency_breakdown_ms"]
    print("    OK")


def stage_check_comparison() -> None:
    """Проверяем, что compare_with_baselines пишет build/memory в comparison.json."""
    print("[3/3] comparison: build + memory in comparison.json...")
    data_dir = PROJECT_ROOT / "data" / "mini"
    if not (data_dir / "corpus.jsonl").exists():
        print(f"    SKIP: нет {data_dir}; (это smoke без подготовки данных)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "smoke_run"
        artifacts = compare_with_baselines(
            prepared_data_dir=data_dir,
            run_dir=run_dir,
            encoder_config=EncoderConfig(backend="hashing", embedding_dim=64, seed=23, device="cpu"),
            model_config=HashingModelConfig(input_dim=64, code_bits=32, hidden_dims=(48, 32), dropout=0.0),
            training_config=TrainingConfig(epochs=1, batch_size=8, learning_rate=5e-3, device="cpu"),
            index_config=IndexConfig(code_bits=32, backend="numpy"),
            search_config=SearchConfig(top_k=3, oversample_factor=3),
        )
        for name, model in artifacts.payload["models"].items():
            build = model.get("build", {})
            test = model.get("test", {})
            print(
                f"    {name:18s}  build_time={build.get('build_time_ms', 0):.2f} ms  "
                f"memory_total={build.get('memory', {}).get('total_bytes', 0)} B  "
                f"latency_breakdown={test.get('latency_breakdown_ms')}"
            )
        for name in ("hybrid_trained", "hybrid_untrained", "dense_exact"):
            assert "build" in artifacts.payload["models"][name], f"no 'build' field for {name}"
            assert "memory" in artifacts.payload["models"][name]["build"]
            assert "latency_breakdown_ms" in artifacts.payload["models"][name]["test"]
        print("    OK")


if __name__ == "__main__":
    stage_check_unit()
    stage_check_evaluator()
    stage_check_comparison()
    print("\n[DONE] Все проверки пройдены.")
