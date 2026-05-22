"""Рис. 5.4. Зависимость recall@10 от длины кода на real_1m.

Тот же график, что 5.3, но для расширенного корпуса real_1m. Показывает
общее снижение абсолютных значений и перестановку лидеров (PQ выходит
вперёд уже на 256 битах).

Источники:
    cb{64,128,512}: baselines/length_sweep_1m/cb*/comparison.json
    cb256:          baselines/joint_D_1m_vs_v3/comparison.json

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_length_sweep_1m.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, COLOR, apply_style, ci95, comparison_recall, save

import matplotlib.pyplot as plt
import numpy as np


SWEEP_1M = ART / "baselines" / "length_sweep_1m"
JOINT_D_CMP = ART / "baselines" / "joint_D_1m_vs_v3" / "comparison.json"
CODE_BITS = [64, 128, 256, 512]


def path_for(bits: int) -> Path:
    if bits == 256:
        return JOINT_D_CMP
    return SWEEP_1M / f"cb{bits}" / "comparison.json"


def collect(method_key: str) -> tuple[list[float], list[tuple[float, float]], int]:
    recalls, errs = [], []
    n_q = 0
    for b in CODE_BITS:
        p, n = comparison_recall(path_for(b), method_key)
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
    n_methods = len(METHODS)
    for i, (label, key, color, marker) in enumerate(METHODS):
        recalls, errs, n = collect(key)
        n_q = n_q or n
        err_arr = np.array(errs).T
        dodge_factor = 1.0 + (i - (n_methods - 1) / 2.0) * 0.022
        xs = [b * dodge_factor for b in CODE_BITS]
        ax.errorbar(
            xs, recalls,
            yerr=err_arr, color=color, marker=marker, markersize=7,
            linewidth=1.6, elinewidth=0.9, capsize=3, label=label,
        )

    dense, _ = comparison_recall(JOINT_D_CMP, "dense_exact")
    if dense is not None:
        ax.axhline(dense, color="#222831", linestyle=":", linewidth=1.2,
                   label=f"dense_exact = {dense:.4f} (потолок)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CODE_BITS)
    ax.set_xticklabels(CODE_BITS)
    ax.set_xlabel("Длина бинарного кода (бит), log₂-шкала")
    ax.set_ylabel(f"recall@10 (binary_only, real_1m, n={n_q})")
    ax.set_title(
        "Рис. 5.4. Зависимость recall@10 от длины кода в режиме binary_only на real_1m",
        loc="left",
    )
    ax.set_ylim(0.48, 0.81)
    ax.legend(loc="lower right", framealpha=0.92, ncol=2)
    ax.grid(True, alpha=0.25)

    save(fig, "fig_5_4_length_sweep_1m_binary")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
