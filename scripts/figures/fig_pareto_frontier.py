"""Рис. 5.7. Pareto-frontier «латентность × recall@10».

Главная задача рисунка — показать **три** Pareto-оптимальные точки и
доминирование остальных конфигураций. Чтобы читаемость не страдала,
точки идентифицируются цветом + формой маркера через табличную
**легенду справа** (а не подписями у каждой точки в плотном кластере).

Pareto-оптимальные точки помечены крупным золотым ободком и подписаны
крупными жирными метками с leader-стрелками, развёрнутыми в свободные
области.

Источник: src/artifacts/baselines/length_sweep_300k/cb256/comparison.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ART, COLOR, apply_style, ci95, comparison_latency, comparison_recall, save,
)

import matplotlib.pyplot as plt
import numpy as np


CMP = ART / "baselines" / "length_sweep_300k" / "cb256" / "comparison.json"


# (method, color, marker)
METHODS = [
    ("dense_exact",      COLOR["dense"],   "o"),
    ("LSH",              COLOR["lsh"],     "s"),
    ("ITQ",              COLOR["itq"],     "^"),
    ("PQ",               COLOR["pq"],      "D"),
    ("Distillation v3",  COLOR["distill"], "v"),
    ("Joint v3",         COLOR["joint"],   ">"),
]

# Какие конфигурации лежат на Pareto-frontier
PARETO = {
    ("PQ", "binary_only"),
    ("ITQ", "rerank"),
    ("dense_exact", "exact"),
}

# Конфигурации (метод, режим, ключ в comparison.json)
POINTS = [
    ("dense_exact",     "exact",       "dense_exact"),
    ("LSH",             "binary_only", "lsh_binary_only"),
    ("LSH",             "rerank",      "lsh_rerank"),
    ("ITQ",             "binary_only", "itq_binary_only"),
    ("ITQ",             "rerank",      "itq_rerank"),
    ("PQ",              "binary_only", "pq_binary_only"),
    ("PQ",              "rerank",      "pq_rerank"),
    ("Distillation v3", "binary_only", "distill_binary_only"),
    ("Distillation v3", "rerank",      "distill_rerank"),
    ("Joint v3",        "binary_only", "joint_pair_binary_only"),
    ("Joint v3",        "rerank",      "joint_pair_rerank"),
]


def main() -> int:
    apply_style()

    # ----- собрать данные -----
    color_for = {m[0]: m[1] for m in METHODS}
    marker_for = {m[0]: m[2] for m in METHODS}

    data = []  # (method, mode, key, lat, rec, half_lo, half_hi, is_pareto)
    n_q = 0
    for method, mode, key in POINTS:
        rec, n = comparison_recall(CMP, key)
        lat = comparison_latency(CMP, key)
        if rec is None or lat is None:
            continue
        n_q = n_q or n
        hi_lo, hi_up = ci95(rec, n)
        data.append((method, mode, key, lat, rec, hi_lo, hi_up,
                     (method, mode) in PARETO))

    # ---------- фигура с двумя панелями: основной график и легенда ----------
    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    # ----- соединяющие линии binary_only ↔ rerank внутри метода -----
    for method in [m[0] for m in METHODS if m[0] != "dense_exact"]:
        pts = [d for d in data if d[0] == method]
        if len(pts) >= 2:
            xs = [p[3] for p in pts]
            ys = [p[4] for p in pts]
            ax.plot(xs, ys, "-", color=color_for[method], linewidth=1.2,
                    alpha=0.45, zorder=1)

    # ----- точки -----
    for method, mode, key, lat, rec, hi_lo, hi_up, is_pareto in data:
        alpha = 0.55 if mode == "binary_only" else 1.0
        size = 180 if is_pareto else 90
        edge_color = "#d4a017" if is_pareto else "#303030"  # золотой ободок для Pareto
        edge_lw = 2.5 if is_pareto else 0.6
        z = 5 if is_pareto else 2

        # errorbar отдельно (чтобы можно было использовать scatter с edge)
        ax.errorbar([lat], [rec], yerr=np.array([[hi_lo], [hi_up]]),
                    fmt="none", ecolor="#404040", elinewidth=0.8,
                    capsize=3, alpha=alpha, zorder=z)
        ax.scatter([lat], [rec], s=size,
                   marker=marker_for[method], c=color_for[method],
                   alpha=alpha, edgecolors=edge_color, linewidths=edge_lw,
                   zorder=z + 1)

    # ----- Pareto-frontier как ступенчатая граница -----
    pareto_pts = sorted(
        [(p[3], p[4]) for p in data if p[7]],  # (lat, rec)
    )
    if len(pareto_pts) >= 2:
        # step-функция вдоль frontier
        px = [pareto_pts[0][0]]
        py = [pareto_pts[0][1]]
        for x, y in pareto_pts[1:]:
            px.append(x)
            py.append(py[-1])  # горизонтальный участок
            px.append(x)
            py.append(y)       # вертикальный скачок
        ax.plot(px, py, "--", color="#202020", linewidth=2.0, alpha=0.75,
                zorder=0.5, label="Pareto-frontier")

    # ----- подписи только для Pareto-точек, с leader-стрелками -----
    pareto_offsets = {
        ("PQ", "binary_only"):     ((20, -2), (35, 14)),
        ("ITQ", "rerank"):         ((-10, 30), (-115, 50)),
        ("dense_exact", "exact"):  ((-30, 20), (-105, 35)),
    }
    for method, mode, key, lat, rec, *_ in data:
        if (method, mode) not in PARETO:
            continue
        label_text, _ = pareto_offsets.get((method, mode), ((10, 10), (10, 10)))
        # подпись для Pareto-точки
        text = {
            ("PQ", "binary_only"):    "★ PQ binary_only\n(минимум памяти)",
            ("ITQ", "rerank"):        "★ ITQ + rerank\n(максимум качества)",
            ("dense_exact", "exact"): "★ dense_exact\n(без сжатия)",
        }[(method, mode)]
        off = pareto_offsets[(method, mode)][1]
        ax.annotate(
            text, xy=(lat, rec), xytext=off, textcoords="offset points",
            fontsize=10, fontweight="bold", color="#202020",
            ha="left" if off[0] >= 0 else "right",
            arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.2,
                            shrinkA=2, shrinkB=6),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6",
                      edgecolor="#d4a017", linewidth=1.0),
            zorder=10,
        )

    ax.set_xscale("log")
    ax.set_xlabel(f"Латентность, мс (log-шкала; n={n_q} запросов)")
    ax.set_ylabel("recall@10 (256 бит, real_300k)")
    ax.set_title(
        "Рис. 5.7. Pareto-frontier «латентность × recall@10» при 256 бит на real_300k",
        loc="left",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(0.788, 0.860)
    ax.set_xlim(4.5, 35.0)

    # ----- легенда справа (методы) -----
    ax_leg.set_title("Метод и режим", loc="left", fontsize=10, pad=4)
    y_cursor = 0.96

    def legend_row(y: float, color: str, marker: str, label: str,
                   mode_label: str, alpha: float, pareto: bool,
                   recall_val: float, latency_val: float) -> None:
        # маркер
        ax_leg.scatter(
            [0.06], [y], s=120, marker=marker, c=color, alpha=alpha,
            edgecolors=("#d4a017" if pareto else "#303030"),
            linewidths=(2.0 if pareto else 0.6),
            transform=ax_leg.transAxes,
            clip_on=False,
        )
        star = "★ " if pareto else "  "
        fw = "bold" if pareto else "normal"
        # текст: метод (режим)
        ax_leg.text(0.14, y, f"{star}{label}", transform=ax_leg.transAxes,
                    fontsize=8.8, va="center", fontweight=fw)
        ax_leg.text(0.14, y - 0.024,
                    f"   {mode_label}: recall={recall_val:.4f}, lat={latency_val:.1f}мс",
                    transform=ax_leg.transAxes, fontsize=7.6, va="center",
                    color="#5a6772")

    # одна запись на каждую точку
    for method, mode, key, lat, rec, _, _, is_pareto in data:
        legend_row(
            y_cursor, color_for[method], marker_for[method],
            method, mode,
            alpha=(0.55 if mode == "binary_only" else 1.0),
            pareto=is_pareto, recall_val=rec, latency_val=lat,
        )
        y_cursor -= 0.075

    # подсказка-легенда внизу
    ax_leg.text(0.04, 0.04,
                "★ — Pareto-оптимальные точки\nполупрозрачные маркеры — binary_only\n"
                "плотные — coarse_rerank",
                transform=ax_leg.transAxes, fontsize=7.8, color="#5a6772",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6",
                          edgecolor="#cbd0d6", linewidth=0.4))

    save(fig, "fig_5_6_pareto_frontier")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
