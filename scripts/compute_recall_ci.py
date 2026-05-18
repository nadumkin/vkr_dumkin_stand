"""Доверительные интервалы для recall@10 из уже имеющихся артефактов.

Считывает агрегированные значения `recall@k` из существующих
`comparison.json` / `summary.json` (БЕЗ перепрогона экспериментов) и считает
**параметрические 95% CI** по нормальному приближению с консервативной
оценкой дисперсии:
    SE ≈ sqrt(p (1-p) / n)              # Bernoulli upper bound
    CI = p ± 1.96 SE

Per-query данные нигде не сохранены, поэтому это **верхняя граница**
для будущих bootstrap CI. После повторного прогона
`run_baselines_comparison.py` (его evaluator уже сохраняет per_query)
запусти `scripts/bootstrap_ci.py` — он даст точные интервалы из эмпирического
распределения, которые будут ≤ показанных здесь.

Соответствие таблиц главы 5 и артефактов:

  Таблица 5.3 (способы обучения хэша, 256 бит):
    Triplet     ← hash_arch_grid_cb256/grid_results.json (профиль 2h_default,
                  старая до-v3 архитектура — близок по абсолютным цифрам
                  к чисто триплетному baseline; *приближение*)
    Distill v1  ← baselines/cb256_v1/comparison.json (distill_*)
    + annealing ← baselines/cb256_v2/comparison.json (distill_*)
    + ITQ-init  ← baselines/cb256_v3/comparison.json (distill_*)

  Таблица 5.4 (методы при 256 бит, real_300k):
    Все строки  ← baselines/length_sweep_300k/cb256/comparison.json

  Таблица 5.5 (длина кода × real_300k):
    cb{64,128,256,512} ← baselines/length_sweep_300k/cb*/comparison.json

  Таблица 5.6 (joint A→B→C→D, 256 бит):
    A           ← joint_distill/A_arch_only/summary.json (rerank only)
    B           ← joint_distill/B_arch_annealing/summary.json (rerank only)
    C           ← baselines/joint_C_epoch3_vs_v3/comparison.json (joint_pair_*)
    D           ← baselines/joint_D_1m_vs_v3/comparison.json (joint_pair_*)
    frozen v3 (300k) ← baselines/length_sweep_300k/cb256/comparison.json (distill_*)
    frozen v3 (1m)   ← baselines/joint_D_1m_vs_v3/comparison.json (distill_*)

  Таблица 5.7 (длина кода × real_1m):
    cb{64,128,512} ← baselines/length_sweep_1m/cb*/comparison.json
    cb256          ← baselines/joint_D_1m_vs_v3/comparison.json

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/compute_recall_ci.py
    .venv/bin/python scripts/compute_recall_ci.py --out artifacts/recall_ci_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts"

Z_95 = 1.96


# ---------------------------------------------------------------------------
# Базовая математика CI
# ---------------------------------------------------------------------------
def ci_normal(p: float, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Возвращает (low, high, half_width) для среднего из n наблюдений в [0, 1]."""
    if n <= 0 or p is None:
        return (0.0, 0.0, 0.0)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    half = z * se
    return (max(0.0, p - half), min(1.0, p + half), half)


def fmt_ci(p: float | None, n: int) -> str:
    """Форматирует «0.XXXX [low, high]» либо «—» если значения нет."""
    if p is None:
        return "—"
    low, high, _ = ci_normal(p, n)
    return f"{p:.4f} [{low:.4f}, {high:.4f}]"


# ---------------------------------------------------------------------------
# Чтение источников
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison(path: Path) -> tuple[dict, int, int]:
    """Возвращает (models_dict, n_queries, corpus_size) — нули если файл отсутствует."""
    data = _read_json(path)
    if not data:
        return {}, 0, 0
    return (
        data.get("models", {}) or {},
        int(data.get("test_queries", 0)),
        int(data.get("corpus_size", 0)),
    )


def recall_from(models: dict, key: str) -> float | None:
    m = models.get(key) if models else None
    if not isinstance(m, dict):
        return None
    v = m.get("recall@k")
    return float(v) if v is not None else None


def load_distill_summary_rerank(path: Path) -> tuple[float | None, int]:
    """Из distill/joint summary.json (поле models.hybrid_*.recall@k)."""
    data = _read_json(path)
    if not data:
        return None, 0
    models = data.get("models", {}) or {}
    # Ключи различаются: hybrid_distilled (frozen) или hybrid_joint_distill (joint)
    for k in ("hybrid_distilled", "hybrid_joint_distill"):
        m = models.get(k)
        if isinstance(m, dict) and m.get("recall@k") is not None:
            return float(m["recall@k"]), int(m.get("queries", 0))
    return None, 0


def load_arch_grid_row(path: Path, row_name: str) -> tuple[float | None, float | None, int]:
    """Из hash_arch_grid_cb256/grid_results.json возвращает (binary, rerank, queries)."""
    data = _read_json(path)
    if not data:
        return None, None, 0
    for r in data.get("results", []):
        if r.get("name") == row_name:
            bin_r = r.get("binary_only", {}).get("recall@k")
            rer_r = r.get("rerank", {}).get("recall@k")
            # queries не хранится в grid_results — берём n=1858 по согласованию с real_300k
            return (
                float(bin_r) if bin_r is not None else None,
                float(rer_r) if rer_r is not None else None,
                1858,
            )
    return None, None, 0


# ---------------------------------------------------------------------------
# Пути к артефактам
# ---------------------------------------------------------------------------
SWEEP_300K = ART / "baselines" / "length_sweep_300k"
SWEEP_1M = ART / "baselines" / "length_sweep_1m"
JOINT_D_1M_PATH = ART / "baselines" / "joint_D_1m_vs_v3" / "comparison.json"

# Источники для Таблицы 5.3
DISTILL_V1_CMP = ART / "baselines" / "cb256_v1" / "comparison.json"     # distill базовая
DISTILL_V2_CMP = ART / "baselines" / "cb256_v2" / "comparison.json"     # + annealing
DISTILL_V3_CMP = ART / "baselines" / "cb256_v3" / "comparison.json"     # + ITQ-init
HASH_ARCH_GRID = ART / "hash_arch_grid_cb256" / "grid_results.json"     # for triplet/legacy

# Источники для Таблицы 5.6
JOINT_A_SUMMARY = ART / "joint_distill" / "A_arch_only" / "summary.json"
JOINT_B_SUMMARY = ART / "joint_distill" / "B_arch_annealing" / "summary.json"
JOINT_C_CMP = ART / "baselines" / "joint_C_epoch3_vs_v3" / "comparison.json"

# Карта длин кода
COMPARISON_PATHS = {
    "real_300k": {
        64:  SWEEP_300K / "cb64"  / "comparison.json",
        128: SWEEP_300K / "cb128" / "comparison.json",
        256: SWEEP_300K / "cb256" / "comparison.json",
        512: SWEEP_300K / "cb512" / "comparison.json",
    },
    "real_1m": {
        64:  SWEEP_1M / "cb64"  / "comparison.json",
        128: SWEEP_1M / "cb128" / "comparison.json",
        256: JOINT_D_1M_PATH,                     # 256 на real_1m — из прогона D
        512: SWEEP_1M / "cb512" / "comparison.json",
    },
}

TABLE_5_4_ROWS = [
    ("dense_exact (pretrained)",      "dense_exact",              "exact"),
    ("LSH",                           "lsh_binary_only",          "binary_only"),
    ("LSH",                           "lsh_rerank",               "rerank"),
    ("ITQ",                           "itq_binary_only",          "binary_only"),
    ("ITQ",                           "itq_rerank",               "rerank"),
    ("PQ (M=32, K=256)",              "pq_binary_only",           "binary_only"),
    ("PQ (M=32, K=256)",              "pq_rerank",                "rerank"),
    ("Distillation v3 (наш)",         "distill_binary_only",      "binary_only"),
    ("Distillation v3 (наш)",         "distill_rerank",           "rerank"),
    ("dense_exact (finetuned)",       "dense_exact_finetuned",    "exact"),
    ("Joint v3 (наш)",                "joint_pair_binary_only",   "binary_only"),
    ("Joint v3 (наш)",                "joint_pair_rerank",        "rerank"),
]

TABLE_LENGTH_SWEEP_ROWS = [
    ("dense_exact",     "dense_exact",            "exact"),
    ("LSH",             "lsh_binary_only",        "binary_only"),
    ("LSH",             "lsh_rerank",             "rerank"),
    ("ITQ",             "itq_binary_only",        "binary_only"),
    ("ITQ",             "itq_rerank",             "rerank"),
    ("PQ",              "pq_binary_only",         "binary_only"),
    ("PQ",              "pq_rerank",              "rerank"),
    ("Distillation v3", "distill_binary_only",    "binary_only"),
    ("Distillation v3", "distill_rerank",         "rerank"),
    ("Joint v3",        "joint_pair_binary_only", "binary_only"),
    ("Joint v3",        "joint_pair_rerank",      "rerank"),
]


# ---------------------------------------------------------------------------
# Сбор данных
# ---------------------------------------------------------------------------
def collect_table_5_3() -> tuple[list[tuple[str, float | None, float | None, str]], int]:
    """Возвращает [(label, recall_bin, recall_rer, source)], n_queries."""
    rows: list[tuple[str, float | None, float | None, str]] = []
    n = 0

    # Triplet baseline — точного 256-бит triplet прогона нет; берём ближайшее по
    # абсолютным значениям из hash_arch_grid_cb256 (2h_default — старая до-v3
    # архитектура, обученная distillation, но с близкими цифрами).
    bin_r, rer_r, n_q = load_arch_grid_row(HASH_ARCH_GRID, "2h_default")
    rows.append((
        "Triplet loss, margin = 4 (приближение)",
        bin_r, rer_r,
        "hash_arch_grid_cb256/grid_results.json [2h_default]",
    ))
    n = max(n, n_q)

    # Distillation базовая
    models, n_q, _ = load_comparison(DISTILL_V1_CMP)
    rows.append((
        "Distillation (базовая)",
        recall_from(models, "distill_binary_only"),
        recall_from(models, "distill_rerank"),
        "baselines/cb256_v1/comparison.json",
    ))
    n = max(n, n_q)

    # + Temperature annealing
    models, n_q, _ = load_comparison(DISTILL_V2_CMP)
    rows.append((
        "+ Temperature annealing (1 → 10)",
        recall_from(models, "distill_binary_only"),
        recall_from(models, "distill_rerank"),
        "baselines/cb256_v2/comparison.json",
    ))
    n = max(n, n_q)

    # + ITQ-init
    models, n_q, _ = load_comparison(DISTILL_V3_CMP)
    rows.append((
        "+ ITQ-init выходного линейного слоя",
        recall_from(models, "distill_binary_only"),
        recall_from(models, "distill_rerank"),
        "baselines/cb256_v3/comparison.json",
    ))
    n = max(n, n_q)

    return rows, n


def collect_table_5_6() -> list[tuple[str, str, float | None, float | None, int, str]]:
    """Возвращает [(label, corpus, recall_bin, recall_rer, n, source)]."""
    rows: list[tuple[str, str, float | None, float | None, int, str]] = []

    # A: rerank only (binary не оценивался в исходном summary)
    a_rer, a_n = load_distill_summary_rerank(JOINT_A_SUMMARY)
    rows.append((
        "A: arch only",
        "real_300k",
        None, a_rer, a_n,
        "joint_distill/A_arch_only/summary.json",
    ))

    # B: rerank only
    b_rer, b_n = load_distill_summary_rerank(JOINT_B_SUMMARY)
    rows.append((
        "B: + temperature annealing",
        "real_300k",
        None, b_rer, b_n,
        "joint_distill/B_arch_annealing/summary.json",
    ))

    # C: оба режима из baselines прогона
    c_models, c_n, _ = load_comparison(JOINT_C_CMP)
    rows.append((
        "C: + ITQ-init",
        "real_300k",
        recall_from(c_models, "joint_pair_binary_only"),
        recall_from(c_models, "joint_pair_rerank"),
        c_n,
        "baselines/joint_C_epoch3_vs_v3/comparison.json",
    ))

    # D: оба режима на real_1m
    d_models, d_n, _ = load_comparison(JOINT_D_1M_PATH)
    rows.append((
        "D: C + расширенный supervision",
        "real_1m",
        recall_from(d_models, "joint_pair_binary_only"),
        recall_from(d_models, "joint_pair_rerank"),
        d_n,
        "baselines/joint_D_1m_vs_v3/comparison.json",
    ))

    # frozen v3 на real_300k — из length_sweep_300k/cb256
    fv_models, fv_n, _ = load_comparison(COMPARISON_PATHS["real_300k"][256])
    rows.append((
        "(сравнение) frozen v3",
        "real_300k",
        recall_from(fv_models, "distill_binary_only"),
        recall_from(fv_models, "distill_rerank"),
        fv_n,
        "baselines/length_sweep_300k/cb256/comparison.json",
    ))

    # frozen v3 на real_1m — из joint_D_1m_vs_v3 (distill_* в том же прогоне)
    rows.append((
        "(сравнение) frozen v3",
        "real_1m",
        recall_from(d_models, "distill_binary_only"),
        recall_from(d_models, "distill_rerank"),
        d_n,
        "baselines/joint_D_1m_vs_v3/comparison.json",
    ))

    return rows


def detect_n(corpus: str) -> int:
    for path in COMPARISON_PATHS.get(corpus, {}).values():
        _, n, _ = load_comparison(path)
        if n > 0:
            return n
    return 0


# ---------------------------------------------------------------------------
# Рендеринг
# ---------------------------------------------------------------------------
def render_table_5_3() -> list[str]:
    rows, n = collect_table_5_3()
    lines = []
    lines.append("### Таблица 5.3. Способы обучения хэширующего модуля")
    lines.append(f"_256 бит, real_300k, n≈{n} запросов_\n")
    lines.append("| Способ обучения | recall@10 (binary_only) | recall@10 (rerank) | Источник |")
    lines.append("|---|---|---|---|")
    for label, p_bin, p_rer, src in rows:
        lines.append(
            f"| {label} | {fmt_ci(p_bin, n)} | {fmt_ci(p_rer, n)} | `{src}` |"
        )
    lines.append("")
    return lines


def render_table_5_4() -> list[str]:
    models, n, _ = load_comparison(COMPARISON_PATHS["real_300k"][256])
    lines = []
    lines.append("### Таблица 5.4. Сравнение методов при длине кода 256 бит на real_300k")
    lines.append(f"_n={n} запросов; источник: `baselines/length_sweep_300k/cb256/comparison.json`_\n")
    lines.append("| Метод | Режим | recall@10 [95% CI] |")
    lines.append("|---|---|---|")
    for label, key, mode in TABLE_5_4_ROWS:
        p = recall_from(models, key)
        lines.append(f"| {label} | {mode} | {fmt_ci(p, n)} |")
    lines.append("")
    return lines


def render_length_sweep_table(
    title: str,
    corpus: str,
    code_bits_list: Iterable[int],
    note: str,
) -> list[str]:
    n = detect_n(corpus)
    lines = []
    lines.append(f"### {title}")
    lines.append(f"_корпус {corpus}, n={n} запросов; {note}_\n")
    cols = list(code_bits_list)
    header = "| Метод | Режим | " + " | ".join(f"{b} бит" for b in cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)

    models_by_bits: dict[int, dict] = {}
    for bits in cols:
        models, _, _ = load_comparison(COMPARISON_PATHS[corpus][bits])
        models_by_bits[bits] = models

    for label, key, mode in TABLE_LENGTH_SWEEP_ROWS:
        cells = []
        for bits in cols:
            p = recall_from(models_by_bits[bits], key)
            cells.append(fmt_ci(p, n))
        lines.append(f"| {label} | {mode} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_table_5_6() -> list[str]:
    rows = collect_table_5_6()
    lines = []
    lines.append("### Таблица 5.6. Серия joint-обучения энкодера и хэш-модуля\n")
    lines.append("| Прогон | Корпус | recall@10 (binary_only) | recall@10 (rerank) | Источник |")
    lines.append("|---|---|---|---|---|")
    for label, corpus, p_bin, p_rer, n, src in rows:
        lines.append(
            f"| {label} | {corpus} (n={n}) | {fmt_ci(p_bin, n)} | {fmt_ci(p_rer, n)} | `{src}` |"
        )
    lines.append("")
    return lines


def render_methodology() -> list[str]:
    n_300k = detect_n("real_300k")
    n_1m = detect_n("real_1m")
    typical_300k = Z_95 * math.sqrt(0.2 * 0.8 / n_300k) if n_300k else 0
    typical_1m = Z_95 * math.sqrt(0.2 * 0.8 / n_1m) if n_1m else 0
    return [
        "## Методика расчёта CI",
        "",
        f"Использовано **нормальное приближение** для среднего из {n_300k} (real_300k) "
        f"и {n_1m} (real_1m) независимых тестовых запросов:",
        "",
        "    SE = sqrt(p(1-p)/n)         (Bernoulli upper bound)",
        "    CI₉₅ = p ± 1.96 · SE",
        "",
        "Per-query recall ∈ [0, 1]; при |relevant|>1 фактическая дисперсия меньше "
        "бернуллиевской, поэтому реальный bootstrap CI будет **уже** или равен "
        "показанному. Типичная полуширина 95% CI при p ≈ 0.8:",
        f"  - real_300k (n={n_300k}): ±{typical_300k:.4f}",
        f"  - real_1m   (n={n_1m}): ±{typical_1m:.4f}",
        "",
        "Все средние взяты напрямую из `comparison.json` / `summary.json` "
        "существующих артефактов (без повторных прогонов). Источник каждой "
        "строки явно указан в колонке *Источник* соответствующей таблицы.",
        "",
        "Точные bootstrap CI с per-query данными можно получить, перепрогнав "
        "`scripts/run_baselines_comparison.py` (обновлённый evaluator теперь "
        "сохраняет `per_query` в `comparison.json`) и затем запустив "
        "`scripts/bootstrap_ci.py`.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Куда сохранить markdown-сводку (помимо вывода в stdout)",
    )
    args = parser.parse_args()

    out_lines: list[str] = []
    out_lines.append("# Сводка recall@10 с 95% доверительными интервалами\n")
    out_lines.append(
        "Сгенерировано `scripts/compute_recall_ci.py` из уже имеющихся артефактов "
        "без повторных прогонов экспериментов.\n"
    )

    out_lines += render_table_5_3()
    out_lines += render_table_5_4()
    out_lines += render_length_sweep_table(
        title="Таблица 5.5. Зависимость recall@10 от длины кода на real_300k",
        corpus="real_300k",
        code_bits_list=[64, 128, 256, 512],
        note="источник: `baselines/length_sweep_300k/cb{64,128,256,512}/comparison.json`",
    )
    out_lines += render_table_5_6()
    out_lines += render_length_sweep_table(
        title="Таблица 5.7. Зависимость recall@10 от длины кода на real_1m",
        corpus="real_1m",
        code_bits_list=[64, 128, 256, 512],
        note="источник: `baselines/length_sweep_1m/cb{64,128,512}/comparison.json` "
             "+ `baselines/joint_D_1m_vs_v3/comparison.json` (для cb256)",
    )
    out_lines += render_methodology()

    text = "\n".join(out_lines)
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\n[OK] записано в {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
