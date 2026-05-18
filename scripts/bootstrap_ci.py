"""Bootstrap-доверительные интервалы для recall@10 из per-query метрик.

Читает comparison.json (один или несколько), для каждого метода извлекает
per_query.recall@k и считает:
  - mean (сверка с recall@k в JSON, должна совпадать),
  - 95% bootstrap CI (по умолчанию 2000 ресэмплов с возвращением),
  - стандартную ошибку среднего.

Выводит таблицу в формате Markdown с колонками:
  Метод | recall@10 | 95% CI [low, high] | half-width

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/bootstrap_ci.py \
        artifacts/baselines/length_sweep_300k/cb256/comparison.json

    # Можно указать несколько файлов одной командой — каждый выводится отдельно.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_recall_ci(
    per_query_recall: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> tuple[float, float, float, float]:
    """Возвращает (mean, ci_low, ci_high, half_width)."""
    arr = np.asarray(per_query_recall, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()

    mean = float(arr.mean())
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    half_width = (high - low) / 2.0
    return mean, low, high, half_width


METHOD_LABELS = [
    ("dense_exact (pretrained)",     "dense_exact"),
    ("LSH (binary_only)",            "lsh_binary_only"),
    ("LSH (rerank)",                 "lsh_rerank"),
    ("ITQ (binary_only)",            "itq_binary_only"),
    ("ITQ (rerank)",                 "itq_rerank"),
    ("PQ (binary_only)",             "pq_binary_only"),
    ("PQ (rerank)",                  "pq_rerank"),
    ("Distillation v3 (binary_only)", "distill_binary_only"),
    ("Distillation v3 (rerank)",     "distill_rerank"),
    ("dense_exact (finetuned)",      "dense_exact_finetuned"),
    ("Joint v3 (binary_only)",       "joint_pair_binary_only"),
    ("Joint v3 (rerank)",            "joint_pair_rerank"),
]


def process_file(path: Path, n_resamples: int = 2000, seed: int = 17) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    corpus_size = data.get("corpus_size", "?")
    code_bits = data.get("code_bits", "?")

    print(f"\n{'=' * 78}")
    print(f"  {path.parent.name}/{path.name}")
    print(f"  corpus_size={corpus_size}, code_bits={code_bits}, queries={data.get('test_queries', '?')}, "
          f"n_resamples={n_resamples}")
    print("=" * 78)
    print(f"{'Метод':<32s}  {'recall@10':>10s}  {'95% CI':>20s}  {'±':>8s}")
    print("-" * 78)

    for label, key in METHOD_LABELS:
        model = models.get(key)
        if not model:
            continue
        per_query = model.get("per_query", {}).get("recall@k")
        if not per_query:
            # Старый формат без per_query — выводим только агрегат.
            recall = model.get("recall@k", 0.0)
            print(f"{label:<32s}  {recall:>10.4f}  {'(нет per_query)':>20s}  {'—':>8s}")
            continue
        mean, low, high, half = bootstrap_recall_ci(
            per_query, n_resamples=n_resamples, seed=seed
        )
        ci_str = f"[{low:.4f}, {high:.4f}]"
        print(f"{label:<32s}  {mean:>10.4f}  {ci_str:>20s}  ±{half:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Один или несколько comparison.json")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    for f in args.files:
        process_file(Path(f), n_resamples=args.n_resamples, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
