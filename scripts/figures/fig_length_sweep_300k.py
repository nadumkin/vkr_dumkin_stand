"""Рис. 5.3. Зависимость recall@10 от длины бинарного кода на real_300k.

Линейный график (логарифмическая шкала по горизонтали: 64, 128, 256, 512 бит).
Отдельная линия для каждого метода в режиме binary_only. Сверху —
пунктирная линия dense_exact. Показывает «диминишинг-возвраты» и
расхождение классических методов с Distillation v3.

Источник: src/artifacts/baselines/length_sweep_300k/cb{64,128,256,512}/comparison.json

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_length_sweep_300k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, COLOR, apply_style, ci95, comparison_recall, save

import matplotlib.pyplot as plt
import numpy as np


SWEEP_ROOT = ART / "baselines" / "length_sweep_300k"
CODE_BITS = [64, 128, 256, 512]


def collect(method_key: str) -> tuple[list[float], list[tuple[float, float]], int]:
    recalls, errs = [], []
    n_q = 0
    for b in CODE_BITS:
        p, n = comparison_recall(SWEEP_ROOT / f"cb{b}" / "comparison.json", method_key)
        n_q = n_q or n
        recalls.append(p if p is not None else float("nan"))
        errs.append(ci95(p, n) if p is not None else (0.0, 0.0))
    return recalls, errs, n_q


METHODS = [
    ("LSH",             "lsh_binary_only",        COLOR["lsh"],     "o"),
    ("ITQ",             "itq_binary_only",        COLOR["itq"],     "s"),
    ("PQ",              "pq_binary_only",         COLOR["pq"],      "^"),
    ("Distillation v3", "distill_binary_only",    COLOR["distill"], "D"),
    ("Joint v3",        "joint_pair_binary_only", COLOR["joint"],   "v"),
]


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 6))

    n_q = 0
    # dodge — небольшое горизонтальное смещение, чтобы маркеры/усы не наезжали
    n_methods = len(METHODS)
    for i, (label, key, color, marker) in enumerate(METHODS):
        recalls, errs, n = collect(key)
        n_q = n_q or n
        err_arr = np.array(errs).T
        # сдвиг каждого метода на ±2% по логарифмической оси
        dodge_factor = 1.0 + (i - (n_methods - 1) / 2.0) * 0.022
        xs = [b * dodge_factor for b in CODE_BITS]
        ax.errorbar(
            xs, recalls,
            yerr=err_arr, color=color, marker=marker, markersize=7,
            linewidth=1.6, elinewidth=0.9, capsize=3, label=label,
        )

    # dense_exact (одинаковый для всех длин)
    dense, _ = comparison_recall(SWEEP_ROOT / "cb256" / "comparison.json", "dense_exact")
    if dense is not None:
        ax.axhline(dense, color="#222831", linestyle=":", linewidth=1.2,
                   label=f"dense_exact = {dense:.4f} (потолок)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CODE_BITS)
    ax.set_xticklabels(CODE_BITS)
    ax.set_xlabel("Длина бинарного кода (бит), log₂-шкала")
    ax.set_ylabel(f"recall@10 (binary_only, real_300k, n={n_q})")
    ax.set_title(
        "Рис. 5.3. Зависимость recall@10 от длины кода в режиме binary_only на real_300k",
        loc="left",
    )
    ax.set_ylim(0.55, 0.86)
    ax.legend(loc="lower right", framealpha=0.92, ncol=2)
    ax.grid(True, alpha=0.25)

    save(fig, "fig_5_3_length_sweep_300k_binary")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
