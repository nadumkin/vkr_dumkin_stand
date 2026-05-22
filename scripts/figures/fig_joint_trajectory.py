"""Рис. 5.X. Траектория прогонов joint-обучения A → B → C → D.

По горизонтали — индекс прогона (A, B, C, D); по вертикали — recall@10.
Две серии линий: binary_only (синяя) и rerank (оранжевая). Тонкие 95% CI.
Горизонтальные пунктирные линии — frozen v3 baseline.

Источники:
    A, B: joint_distill/{A_arch_only, B_arch_annealing}/summary.json
          (rerank only, binary_only не оценивался)
    C:    baselines/joint_C_epoch3_vs_v3/comparison.json (joint_pair_*)
    D:    baselines/joint_D_1m_vs_v3/comparison.json (joint_pair_* — на real_1m)
    frozen v3 (300k): baselines/length_sweep_300k/cb256/comparison.json (distill_*)
    frozen v3 (1m):   baselines/joint_D_1m_vs_v3/comparison.json (distill_*)

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/fig_joint_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ART, COLOR, apply_style, ci95, comparison_recall, read_json, save

import matplotlib.pyplot as plt
import numpy as np


def load_joint_summary_rerank(path: Path) -> tuple[float | None, int]:
    if not path.exists():
        return None, 0
    d = read_json(path)
    for k in ("hybrid_joint_distill", "hybrid_distilled"):
        m = d.get("models", {}).get(k)
        if isinstance(m, dict) and m.get("recall@k") is not None:
            return float(m["recall@k"]), int(m.get("queries", 0))
    return None, 0


def main() -> int:
    apply_style()

    # ----- собрать значения -----
    # A
    a_rer, na = load_joint_summary_rerank(ART / "joint_distill" / "A_arch_only" / "summary.json")
    # B
    b_rer, nb = load_joint_summary_rerank(ART / "joint_distill" / "B_arch_annealing" / "summary.json")
    # C
    c_cmp = ART / "baselines" / "joint_C_epoch3_vs_v3" / "comparison.json"
    c_bin, nc = comparison_recall(c_cmp, "joint_pair_binary_only")
    c_rer, _ = comparison_recall(c_cmp, "joint_pair_rerank")
    # D
    d_cmp = ART / "baselines" / "joint_D_1m_vs_v3" / "comparison.json"
    d_bin, nd = comparison_recall(d_cmp, "joint_pair_binary_only")
    d_rer, _ = comparison_recall(d_cmp, "joint_pair_rerank")

    # frozen v3 baselines
    frozen_300k_bin, _ = comparison_recall(
        ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json",
        "distill_binary_only",
    )
    frozen_300k_rer, _ = comparison_recall(
        ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json",
        "distill_rerank",
    )
    frozen_1m_bin, _ = comparison_recall(d_cmp, "distill_binary_only")
    frozen_1m_rer, _ = comparison_recall(d_cmp, "distill_rerank")

    runs = ["A\narch only", "B\n+annealing", "C\n+ITQ-init", "D\n+real_1m"]
    x = np.arange(len(runs))

    bin_vals = [None, None, c_bin, d_bin]
    rer_vals = [a_rer, b_rer, c_rer, d_rer]

    # n_q зависит от прогона: A/B/C на 1858, D на 3518
    ns = [na or 1858, nb or 1858, nc or 1858, nd or 3518]
    bin_errs = [ci95(p, n) if p is not None else (0.0, 0.0) for p, n in zip(bin_vals, ns)]
    rer_errs = [ci95(p, n) if p is not None else (0.0, 0.0) for p, n in zip(rer_vals, ns)]

    # ---------- фигура ----------
    fig, ax = plt.subplots(figsize=(10.5, 6))

    # binary_only — точки только там, где есть значения
    mask_bin = [p is not None for p in bin_vals]
    x_bin = x[np.array(mask_bin)]
    y_bin = [bin_vals[i] for i in range(len(bin_vals)) if mask_bin[i]]
    yerr_bin = np.array([bin_errs[i] for i in range(len(bin_errs)) if mask_bin[i]]).T

    ax.errorbar(
        x_bin, y_bin, yerr=yerr_bin,
        color=COLOR["lsh"], marker="o", markersize=8, linewidth=1.8,
        capsize=3.5, elinewidth=1.0,
        label="Joint v3, binary_only",
    )
    # пунктиром «без замера» между A↔B↔C для бинарного режима
    ax.plot([x[0], x[2]], [None, None], alpha=0)

    # rerank
    ax.errorbar(
        x, rer_vals, yerr=np.array(rer_errs).T,
        color=COLOR["pq"], marker="s", markersize=8, linewidth=1.8,
        capsize=3.5, elinewidth=1.0,
        label="Joint v3, rerank",
    )

    # frozen v3 baseline — отрезок A..C (real_300k)
    if frozen_300k_bin is not None:
        ax.hlines(frozen_300k_bin, xmin=-0.4, xmax=2.4,
                  colors=COLOR["lsh"], linestyles="--", linewidth=1.0, alpha=0.7,
                  label=f"frozen v3 binary (real_300k) = {frozen_300k_bin:.4f}")
    if frozen_300k_rer is not None:
        ax.hlines(frozen_300k_rer, xmin=-0.4, xmax=2.4,
                  colors=COLOR["pq"], linestyles="--", linewidth=1.0, alpha=0.7,
                  label=f"frozen v3 rerank (real_300k) = {frozen_300k_rer:.4f}")
    # frozen v3 на real_1m — отрезок D
    if frozen_1m_bin is not None:
        ax.hlines(frozen_1m_bin, xmin=2.6, xmax=3.4,
                  colors=COLOR["lsh"], linestyles=":", linewidth=1.4, alpha=0.85,
                  label=f"frozen v3 binary (real_1m) = {frozen_1m_bin:.4f}")
    if frozen_1m_rer is not None:
        ax.hlines(frozen_1m_rer, xmin=2.6, xmax=3.4,
                  colors=COLOR["pq"], linestyles=":", linewidth=1.4, alpha=0.85,
                  label=f"frozen v3 rerank (real_1m) = {frozen_1m_rer:.4f}")

    # вертикальная разделительная линия между C и D (смена корпуса)
    ax.axvline(2.5, color="#9aa4ad", linestyle="-.", linewidth=0.7)
    ax.text(2.5, 0.895, "Смена корпуса:\nreal_300k → real_1m", ha="center", va="bottom",
            fontsize=8.5, color="#5a6772",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f3f4f6",
                      edgecolor="#cbd0d6", linewidth=0.4))

    # значения у точек — позиционируем за концом верхнего/нижнего уса с боковым смещением
    for xi, p, err in zip(x, rer_vals, rer_errs):
        if p is not None:
            ax.text(xi + 0.10, p + err[1] + 0.002, f"{p:.4f}",
                    ha="left", va="bottom", fontsize=8.2, color=COLOR["pq"])
    for xi, p, err in zip(x_bin, y_bin, [bin_errs[i] for i in range(len(bin_errs)) if bin_vals[i] is not None]):
        ax.text(xi + 0.10, p - err[0] - 0.002, f"{p:.4f}",
                ha="left", va="top", fontsize=8.2, color=COLOR["lsh"])

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=10)
    ax.set_ylabel("recall@10")
    ax.set_title(
        "Рис. 5.X. Траектория joint-обучения A → B → C → D:\n"
        "ни одно улучшение не превзошло frozen v3 baseline (пунктиры)",
        loc="left",
    )
    ax.set_ylim(0.70, 0.93)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.25)

    save(fig, "fig_5_joint_trajectory")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
