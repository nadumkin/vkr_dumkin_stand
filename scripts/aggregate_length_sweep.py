"""Сводная таблица по результатам run_length_sweep_overnight.sh.

Читает comparison.json из обеих фаз серии:
  Phase 1: artifacts/baselines/length_sweep_300k/cb{64,128,256,512}/
  Phase 2: artifacts/baselines/length_sweep_1m/cb{64,128,512}/

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/aggregate_length_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_300K = PROJECT_ROOT / "artifacts" / "baselines" / "length_sweep_300k"
ROOT_1M = PROJECT_ROOT / "artifacts" / "baselines" / "length_sweep_1m"
# 256-битная точка на real_1m уже есть из прогона D
LEGACY_D_1M = PROJECT_ROOT / "artifacts" / "baselines" / "joint_D_1m_vs_v3"


def load_results(path: Path) -> dict:
    cmp = path / "comparison.json"
    if not cmp.exists():
        return {}
    return json.loads(cmp.read_text(encoding="utf-8")).get("models", {})


def load_corpus_results(root: Path, code_bits_list: list[int], extra_paths: dict[int, Path] | None = None) -> dict:
    extra = extra_paths or {}
    out: dict[int, dict] = {}
    for bits in code_bits_list:
        if bits in extra and extra[bits].exists() and (extra[bits] / "comparison.json").exists():
            out[bits] = load_results(extra[bits])
        else:
            out[bits] = load_results(root / f"cb{bits}")
    return out


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


METHOD_LABELS = [
    ("dense_exact (pretrained)",     "dense_exact"),
    ("LSH (binary_only)",            "lsh_binary_only"),
    ("LSH (rerank)",                 "lsh_rerank"),
    ("ITQ (binary_only)",            "itq_binary_only"),
    ("ITQ (rerank)",                 "itq_rerank"),
    ("PQ (binary_only)",             "pq_binary_only"),
    ("PQ (rerank)",                  "pq_rerank"),
    ("frozen v3 (binary_only)",      "distill_binary_only"),
    ("frozen v3 (rerank)",           "distill_rerank"),
    ("dense_exact (finetuned)",      "dense_exact_finetuned"),
    ("joint v3 (binary_only)",       "joint_pair_binary_only"),
    ("joint v3 (rerank)",            "joint_pair_rerank"),
]


def print_table(title: str, results: dict[int, dict], code_bits_list: list[int]) -> None:
    print(title)
    print("=" * (32 + 13 * len(code_bits_list)))
    print(f"{'Метод':<30s}  " + "  ".join(f"{f'{b} бит':>10s}" for b in code_bits_list))
    print("-" * (32 + 13 * len(code_bits_list)))
    for label, key in METHOD_LABELS:
        row = []
        for bits in code_bits_list:
            data = results.get(bits, {}).get(key)
            row.append(fmt(data["recall@k"]) if data else "—")
        print(f"{label:<30s}  " + "  ".join(f"{v:>10s}" for v in row))
    print()


def print_compact(title: str, results: dict[int, dict], code_bits_list: list[int],
                  methods: list[tuple[str, str]]) -> None:
    print(title)
    print("-" * (24 + 13 * len(code_bits_list)))
    print(f"{'Метод':<22s}  " + "  ".join(f"{f'{b} бит':>10s}" for b in code_bits_list))
    print("-" * (24 + 13 * len(code_bits_list)))
    for label, key in methods:
        row = []
        for bits in code_bits_list:
            data = results.get(bits, {}).get(key)
            row.append(fmt(data["recall@k"]) if data else "—")
        print(f"{label:<22s}  " + "  ".join(f"{v:>10s}" for v in row))
    print()


def main() -> int:
    # ----- Phase 1: real_300k -----
    bits_300k = [64, 128, 256, 512]
    results_300k = load_corpus_results(ROOT_300K, bits_300k)
    available_300k = [b for b in bits_300k if results_300k[b]]

    if available_300k:
        print()
        print("##############################################################")
        print("#  PHASE 1: real_300k (length sweep)                          #")
        print("##############################################################")
        print()
        print_table("Полная таблица:", results_300k, bits_300k)
        print_compact(
            "Сводка binary-only:", results_300k, bits_300k,
            [("LSH", "lsh_binary_only"),
             ("ITQ", "itq_binary_only"),
             ("PQ", "pq_binary_only"),
             ("frozen v3", "distill_binary_only"),
             ("joint v3", "joint_pair_binary_only"),
             ("dense (потолок)", "dense_exact")],
        )
        print_compact(
            "Сводка rerank:", results_300k, bits_300k,
            [("LSH+rerank", "lsh_rerank"),
             ("ITQ+rerank", "itq_rerank"),
             ("PQ+rerank", "pq_rerank"),
             ("frozen v3+rerank", "distill_rerank"),
             ("joint v3+rerank", "joint_pair_rerank"),
             ("dense (точно)", "dense_exact")],
        )
    else:
        print("[WARN] Phase 1 (real_300k) — нет данных")

    # ----- Phase 2: real_1m -----
    bits_1m = [64, 128, 256, 512]
    # 256 берётся из legacy D-прогона
    results_1m = load_corpus_results(ROOT_1M, bits_1m, extra_paths={256: LEGACY_D_1M})
    available_1m = [b for b in bits_1m if results_1m[b]]

    if available_1m:
        print()
        print("##############################################################")
        print("#  PHASE 2: real_1m (length sweep)                            #")
        print("##############################################################")
        print()
        print_table("Полная таблица:", results_1m, bits_1m)
        print_compact(
            "Сводка binary-only:", results_1m, bits_1m,
            [("LSH", "lsh_binary_only"),
             ("ITQ", "itq_binary_only"),
             ("PQ", "pq_binary_only"),
             ("frozen v3", "distill_binary_only"),
             ("joint v3", "joint_pair_binary_only"),
             ("dense (потолок)", "dense_exact")],
        )
        print_compact(
            "Сводка rerank:", results_1m, bits_1m,
            [("LSH+rerank", "lsh_rerank"),
             ("ITQ+rerank", "itq_rerank"),
             ("PQ+rerank", "pq_rerank"),
             ("frozen v3+rerank", "distill_rerank"),
             ("joint v3+rerank", "joint_pair_rerank"),
             ("dense (точно)", "dense_exact")],
        )
    else:
        print("[WARN] Phase 2 (real_1m) — нет данных")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
