"""Рис. 5.1. Способы обучения хэширующего модуля.

Столбчатая диаграмма recall@10 в режиме binary_only для четырёх способов
обучения: triplet, distillation, +annealing, +ITQ-init. Цвет — по виду
сигнала (triplet vs distillation) и параметрам (annealing, init).

Источники:
    - Triplet baseline: hash_arch_grid_cb256/grid_results.json [2h_default]
    - Distillation v1: baselines/cb256_v1/comparison.json (distill_binary_only)
    - + annealing v2: baselines/cb256_v2/comparison.json
    - + ITQ-init v3:  baselines/cb256_v3/comparison.json

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_training_schemes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, COLOR, apply_style, ci95, comparison_recall, read_json, save

import matplotlib.pyplot as plt
import numpy as np


def get_triplet_baseline() -> tuple[float, float]:
    """Возвращает (binary_only, rerank) для приближения triplet@m=4."""
    grid = read_json(ART / "hash_arch_grid_cb256" / "grid_results.json")
    for r in grid["results"]:
        if r["name"] == "2h_default":
            return r["binary_only"]["recall@k"], r["rerank"]["recall@k"]
    raise RuntimeError("2h_default не найден в grid_results.json")


def main() -> int:
    apply_style()

    # ----- собрать данные -----
    triplet_bin, triplet_rer = get_triplet_baseline()
    distill_v1_bin, n = comparison_recall(ART / "baselines" / "cb256_v1" / "comparison.json", "distill_binary_only")
    distill_v1_rer, _ = comparison_recall(ART / "baselines" / "cb256_v1" / "comparison.json", "distill_rerank")
    distill_v2_bin, _ = comparison_recall(ART / "baselines" / "cb256_v2" / "comparison.json", "distill_binary_only")
    distill_v2_rer, _ = comparison_recall(ART / "baselines" / "cb256_v2" / "comparison.json", "distill_rerank")
    distill_v3_bin, _ = comparison_recall(ART / "baselines" / "cb256_v3" / "comparison.json", "distill_binary_only")
    distill_v3_rer, _ = comparison_recall(ART / "baselines" / "cb256_v3" / "comparison.json", "distill_rerank")
    dense_recall, _ = comparison_recall(
        ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json", "dense_exact",
    )

    n_q = n or 1858

    labels = [
        "Triplet loss\nm = 4",
        "Distillation\n(базовая)",
        "+ Temperature\nannealing 1→10",
        "+ ITQ-init\nвыходного слоя",
    ]
    bin_values = [triplet_bin, distill_v1_bin, distill_v2_bin, distill_v3_bin]
    rer_values = [triplet_rer, distill_v1_rer, distill_v2_rer, distill_v3_rer]
    bin_errs = np.array([ci95(p, n_q) for p in bin_values]).T
    rer_errs = np.array([ci95(p, n_q) for p in rer_values]).T

    # Цвет: triplet — красный (отрицательный baseline), distillation/+annealing/+itq — синий-зелёный градиент
    colors = [
        "#b13e3e",                  # triplet (red — pre-distillation)
        "#a8c5e6",                  # distill base (light blue)
        "#5b8fbe",                  # + annealing (medium blue)
        "#2c5e9b",                  # + ITQ-init (dark blue — финал)
    ]

    # ---------- фигура ----------
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    x = np.arange(len(labels))
    width = 0.38

    # binary_only (основная — слева)
    bars_bin = ax.bar(
        x - width / 2, bin_values, width,
        yerr=bin_errs, color=colors, edgecolor="#202020", linewidth=0.5,
        error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capsize": 3},
        label="binary_only",
    )
    # rerank (вторая — справа)
    bars_rer = ax.bar(
        x + width / 2, rer_values, width,
        yerr=rer_errs, color=colors, edgecolor="#202020", linewidth=0.5, alpha=0.45,
        hatch="//", error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capsize": 3},
        label="rerank (coarse_rerank)",
    )

    # значения над столбцами — позиция считается от верхнего конца уса
    for bars, vals, errs in [(bars_bin, bin_values, bin_errs), (bars_rer, rer_values, rer_errs)]:
        upper = errs[1]  # массив верхних дельт
        for bar, v, up in zip(bars, vals, upper):
            ax.text(bar.get_x() + bar.get_width() / 2, v + up + 0.004,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.3)

    # dense_exact reference
    if dense_recall is not None:
        ax.axhline(dense_recall, color="#222831", linestyle=":", linewidth=1.2,
                   label=f"dense_exact = {dense_recall:.4f} (потолок)")

    # дельты между соседними столбцами (binary_only) — внизу под осью X в отдельной полосе
    deltas = [bin_values[i+1] - bin_values[i] for i in range(len(bin_values) - 1)]
    for i, d in enumerate(deltas):
        sign = "+" if d >= 0 else "−"
        x_mid = (x[i] + x[i+1]) / 2 - width / 2
        ax.text(x_mid, 0.715, f"{sign}{abs(d)*100:.2f} п.п.",
                ha="center", va="center", fontsize=8.5, color="#5a6772",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f3f4f6",
                          edgecolor="#cbd0d6", linewidth=0.4))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("recall@10 (256 бит, real_300k, n=1858)")
    ax.set_title(
        "Рис. 5.1. Способы обучения хэширующего модуля: последовательное "
        "включение техник\n"
        "(triplet → distillation → +temperature annealing → +ITQ-init)",
        loc="left",
    )
    ax.set_ylim(0.69, 0.88)
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)

    save(fig, "fig_5_1_training_schemes")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
