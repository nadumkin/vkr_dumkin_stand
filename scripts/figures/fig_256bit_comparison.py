"""Рис. 5.2. Сравнение методов сжатия при 256 битах на real_300k.

Сгруппированная столбчатая диаграмма по 5 методам (dense_exact, LSH, ITQ,
PQ, Distillation v3) с двумя столбцами на каждый метод: binary_only
(светлый) и rerank (тёмный). Сверху — пунктирная линия dense_exact.

Источник: src/artifacts/baselines/length_sweep_300k/cb256/comparison.json

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_256bit_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, COLOR, apply_style, ci95, comparison_recall, save

import matplotlib.pyplot as plt
import numpy as np


CMP = ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json"


def main() -> int:
    apply_style()

    methods = [
        ("dense_exact",      "dense_exact",          "dense_exact",          COLOR["dense"]),
        ("LSH",              "lsh_binary_only",      "lsh_rerank",           COLOR["lsh"]),
        ("ITQ",              "itq_binary_only",      "itq_rerank",           COLOR["itq"]),
        ("PQ (M=32, K=256)", "pq_binary_only",       "pq_rerank",            COLOR["pq"]),
        ("Distillation v3",  "distill_binary_only",  "distill_rerank",       COLOR["distill"]),
    ]

    bin_values, rer_values, base_colors = [], [], []
    n_q = 0
    for _, bk, rk, color in methods:
        bv, n = comparison_recall(CMP, bk)
        rv, _ = comparison_recall(CMP, rk)
        n_q = n_q or n
        bin_values.append(bv)
        rer_values.append(rv)
        base_colors.append(color)

    bin_errs = np.array([ci95(p, n_q) for p in bin_values]).T
    rer_errs = np.array([ci95(p, n_q) for p in rer_values]).T
    dense_recall = bin_values[0]  # dense_exact = bin_values[0] == rer_values[0]

    # ---------- фигура ----------
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(methods))
    width = 0.36

    # binary_only — светлый (с прозрачностью)
    bars_bin = ax.bar(
        x - width / 2, bin_values, width,
        yerr=bin_errs, color=base_colors, edgecolor="#202020", linewidth=0.5, alpha=0.55,
        error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capsize": 3},
        label="binary_only",
    )
    # rerank — тёмный
    bars_rer = ax.bar(
        x + width / 2, rer_values, width,
        yerr=rer_errs, color=base_colors, edgecolor="#202020", linewidth=0.5,
        error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capsize": 3},
        label="coarse_rerank",
    )

    # значения — поверх верхнего уса
    for bars, vals, errs in [(bars_bin, bin_values, bin_errs), (bars_rer, rer_values, rer_errs)]:
        upper = errs[1]
        for bar, v, up in zip(bars, vals, upper):
            ax.text(bar.get_x() + bar.get_width() / 2, v + up + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.1)

    # dense_exact reference
    ax.axhline(dense_recall, color="#222831", linestyle=":", linewidth=1.0,
               label=f"dense_exact = {dense_recall:.4f} (потолок)")

    method_labels = [m[0] for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel(f"recall@10 (256 бит, real_300k, n={n_q})")
    ax.set_title(
        "Рис. 5.2. Сравнение методов сжатия при длине кода 256 бит на real_300k\n"
        "binary_only (светлый, с прозрачностью) vs coarse_rerank (тёмный)",
        loc="left",
    )
    ax.set_ylim(0.77, 0.865)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)

    save(fig, "fig_5_2_256bit_comparison")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
