"""Рис. 5.5. Совокупная память поисковых структур по компонентам.

Столбчатая диаграмма в логарифмическом масштабе по вертикали для пяти
конфигураций (LSH/ITQ/Distill, PQ — в обоих режимах + dense_exact).
Столбцы разделены на компоненты: бинарные коды, HNSW-граф, кодовые книги
PQ, плотные векторы для rerank.

Источник: src/artifacts/baselines/length_sweep_300k/cb256/comparison.json
          (поле memory.* в моделях)

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_memory_breakdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, apply_style, comparison_memory, read_json, save

import matplotlib.pyplot as plt
import numpy as np


CMP = ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json"
MB = 1024 ** 2  # байты в МБ


def get_mb(d: dict, *keys) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v / MB
    return 0.0


def main() -> int:
    apply_style()

    # ----- собрать данные по моделям -----
    # Колонки: бинарные коды | HNSW-граф | кодовые книги | плотные векторы
    configs = [
        ("dense_exact",                "dense_exact"),
        ("LSH/ITQ/Distill v3\n(binary_only)", "lsh_binary_only"),
        ("LSH/ITQ/Distill v3\n(rerank)",      "lsh_rerank"),
        ("PQ\n(binary_only)",          "pq_binary_only"),
        ("PQ\n(rerank)",               "pq_rerank"),
    ]

    labels = [c[0] for c in configs]
    codes_mb, hnsw_mb, codebooks_mb, dense_mb = [], [], [], []

    for _, key in configs:
        m = comparison_memory(CMP, key)
        # Бинарные коды: packed_codes_bytes (LSH/ITQ/Distill) или codes_bytes (PQ)
        codes_mb.append(get_mb(m, "packed_codes_bytes", "codes_bytes"))
        hnsw_mb.append(get_mb(m, "hnsw_graph_bytes"))
        codebooks_mb.append(get_mb(m, "codebooks_bytes"))
        dense_mb.append(get_mb(m, "dense_for_rerank_bytes", "dense_embeddings_bytes"))

    # ---------- фигура ----------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(configs))
    width = 0.62

    bottom = np.zeros(len(configs))
    components = [
        ("Бинарные коды",       codes_mb,     "#4d7ea8"),
        ("HNSW-граф",           hnsw_mb,      "#2e8b57"),
        ("Кодовые книги PQ",    codebooks_mb, "#e07b39"),
        ("Плотные векторы",     dense_mb,     "#9aa4ad"),
    ]
    for label, values, color in components:
        ax.bar(x, values, width, bottom=bottom, color=color, edgecolor="#202020",
               linewidth=0.4, label=label)
        bottom += np.array(values)

    # Подписи итогов над столбцами
    totals = bottom
    for xi, total in zip(x, totals):
        ax.text(xi, total * 1.15, f"{total:.1f} МБ", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Память поисковых структур, МБ (log-шкала)")
    ax.set_title(
        "Рис. 5.5. Совокупная память поисковых структур по компонентам\n"
        "(256 бит, real_300k = 224 017 документов)",
        loc="left",
    )
    ax.set_ylim(1, totals.max() * 3)
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # Аннотация контр-интуитивного результата
    ax.text(
        2, 1.5,
        "PQ binary_only — единственный метод\n"
        "с радикальной экономией памяти (≈8 МБ).\n"
        "HNSW-граф уничтожает выигрыш бинаризации\n"
        "у LSH/ITQ/Distill (≈295 МБ).",
        fontsize=9, color="#404040",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7e6", edgecolor="#d0a566", linewidth=0.6),
    )

    save(fig, "fig_5_5_memory_breakdown")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
