"""Рис. 4.1. Архитектурный ablation хэш-модуля.

Горизонтальная столбчатая диаграмма recall@10 binary_only для 10 конфигураций
архитектуры MLP, отсортированных по убыванию. Цвет — по наличию LayerNorm.
Сверху — горизонтальные линии ITQ и dense_exact. Справа — миниатюра rerank.

Источник: src/artifacts/hash_arch_grid_cb256/grid_results.json
         src/artifacts/baselines/length_sweep_300k/cb256/comparison.json
         (для горизонтальных линий ITQ / dense_exact)

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_arch_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ART, COLOR, FIG_DIR, apply_style, ci95, comparison_recall,
    read_json, save,
)

import matplotlib.pyplot as plt
import numpy as np


# Конфигурации, у которых LayerNorm: 1h_with_ln, 1h_wide_768,
# 2h_no_drop, 2h_default, 2h_relu, 2h_tanh, 3h_deep
# Без LayerNorm: linear, 1h_no_ln, 2h_no_ln_no_drop
LN_CONFIGS = {
    "1h_with_ln", "1h_wide_768", "2h_no_drop", "2h_default",
    "2h_relu", "2h_tanh", "3h_deep",
}


def main() -> int:
    apply_style()

    grid = read_json(ART / "hash_arch_grid_cb256" / "grid_results.json")
    rows = grid["results"]

    # Сортировка по recall@10 binary_only
    rows = sorted(rows, key=lambda r: r["binary_only"]["recall@k"], reverse=True)

    names = [r["name"] for r in rows]
    bin_recalls = [r["binary_only"]["recall@k"] for r in rows]
    rer_recalls = [r["rerank"]["recall@k"] for r in rows]
    n_q = 1858  # фиксированный размер тестовой выборки real_300k
    bin_errs = np.array([ci95(p, n_q) for p in bin_recalls]).T  # (2, N): (-, +)
    rer_errs = np.array([ci95(p, n_q) for p in rer_recalls]).T

    colors = [COLOR["with_ln"] if n in LN_CONFIGS else COLOR["no_ln"] for n in names]

    # Эталонные линии — ITQ и dense_exact из 256-битной таблицы
    itq_recall, _ = comparison_recall(
        ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json",
        "itq_binary_only",
    )
    dense_recall, _ = comparison_recall(
        ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json",
        "dense_exact",
    )

    # ---------- фигура ----------
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(1, 4, wspace=0.45)
    ax_main = fig.add_subplot(gs[0, 0:3])
    ax_inset = fig.add_subplot(gs[0, 3])

    # ----- основная панель: binary_only -----
    y_pos = np.arange(len(names))[::-1]  # лучший — сверху
    ax_main.barh(
        y_pos, bin_recalls,
        xerr=bin_errs, color=colors, edgecolor="#202020",
        linewidth=0.4, error_kw={"ecolor": "#303030", "elinewidth": 0.7, "capsize": 2.5},
    )
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(names, fontsize=9)
    ax_main.set_xlabel("recall@10 (binary_only)")
    ax_main.set_title("Рис. 4.1. Архитектурный ablation: recall@10 в режиме binary_only", loc="left")
    ax_main.set_xlim(0.55, 0.89)
    ax_main.grid(True, axis="x", alpha=0.25)

    # эталонные линии
    if itq_recall is not None:
        ax_main.axvline(itq_recall, color="#2e8b57", linestyle="--", linewidth=1.1,
                        label=f"ITQ (классический baseline) = {itq_recall:.4f}")
    if dense_recall is not None:
        ax_main.axvline(dense_recall, color="#222831", linestyle=":", linewidth=1.1,
                        label=f"dense_exact (верхняя граница) = {dense_recall:.4f}")

    # значения справа от верхнего конца CI-усов, чтобы не наезжать на капы
    for y, val, (lo, hi) in zip(y_pos, bin_recalls, zip(bin_errs[0], bin_errs[1])):
        ax_main.text(val + hi + 0.004, y, f"{val:.4f}",
                     va="center", ha="left", fontsize=8.5, color="#202020")

    # легенда по цвету: with_ln / no_ln + reference lines
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR["with_ln"], label="с LayerNorm"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR["no_ln"], label="без LayerNorm"),
    ]
    if itq_recall is not None:
        handles.append(plt.Line2D([0], [0], color="#2e8b57", linestyle="--",
                                  linewidth=1.1, label=f"ITQ baseline = {itq_recall:.4f}"))
    if dense_recall is not None:
        handles.append(plt.Line2D([0], [0], color="#222831", linestyle=":",
                                  linewidth=1.1, label=f"dense_exact = {dense_recall:.4f}"))
    ax_main.legend(handles=handles, loc="lower right", framealpha=0.92, fontsize=8.5)

    # ----- миниатюра: rerank -----
    ax_inset.barh(
        y_pos, rer_recalls,
        xerr=rer_errs, color=colors, edgecolor="#202020",
        linewidth=0.4, error_kw={"ecolor": "#303030", "elinewidth": 0.6, "capsize": 2.0},
    )
    ax_inset.set_yticks(y_pos)
    ax_inset.set_yticklabels([])  # не дублируем подписи
    ax_inset.set_xlabel("recall@10 (rerank)")
    ax_inset.set_title("rerank", loc="left", fontsize=10)
    ax_inset.set_xlim(0.76, 0.88)
    if dense_recall is not None:
        ax_inset.axvline(dense_recall, color="#222831", linestyle=":", linewidth=0.9)
    if itq_recall is not None:
        # ITQ rerank — тоже 0.8388 (потолок), линия совпадает с dense; не дублируем
        pass
    ax_inset.grid(True, axis="x", alpha=0.25)

    # Аннотация: диапазон 19.6 п.п. (binary) и 6.1 п.п. (rerank)
    range_bin = max(bin_recalls) - min(bin_recalls)
    range_rer = max(rer_recalls) - min(rer_recalls)
    ax_main.text(
        0.555, -0.5,
        f"Диапазон binary_only: {range_bin*100:.1f} п.п.   |   rerank: {range_rer*100:.1f} п.п.",
        fontsize=8.5, color="#404040",
    )

    fig.suptitle("", y=1.02)
    save(fig, "fig_4_1_arch_ablation")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
