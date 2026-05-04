from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_epoch_dynamics_figure(
    output_path: str | Path,
    *,
    comparison_2_epochs: str | Path,
    comparison_5_epochs: str | Path,
    comparison_10_epochs: str | Path,
) -> Path:
    runs = [
        (2, load_json(comparison_2_epochs)),
        (5, load_json(comparison_5_epochs)),
        (10, load_json(comparison_10_epochs)),
    ]
    epochs = [epoch for epoch, _ in runs]
    metric_specs = [
        ("Recall@10", "recall@k", "#1f77b4"),
        ("MAP@10", "map@k", "#ff7f0e"),
        ("nDCG@10", "ndcg@k", "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for label, key, color in metric_specs:
        values = [payload["models"]["hybrid_trained"]["test"][key] for _, payload in runs]
        ax.plot(epochs, values, marker="o", linewidth=2.4, markersize=6, label=label, color=color)
        for epoch, value in zip(epochs, values):
            ax.annotate(f"{value:.3f}", (epoch, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    ax.set_title("Влияние числа эпох на качество поиска\n32 бита, oversample_factor = 3, тестовая выборка")
    ax.set_xlabel("Число эпох обучения")
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(epochs)
    ax.set_ylim(0.0, 0.14)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()

    target = Path(output_path)
    fig.savefig(target, format=target.suffix.lstrip("."))
    plt.close(fig)
    return target


def build_best_run_comparison_figure(
    output_path: str | Path,
    *,
    best_run_comparison: str | Path,
) -> Path:
    payload = load_json(best_run_comparison)
    model_specs = [
        ("dense_exact", "Dense exact"),
        ("hybrid_untrained", "Гибрид\nбез обучения"),
        ("hybrid_trained", "Гибрид\nпосле обучения"),
    ]
    metric_specs = [
        ("Recall@10", "recall@k", "#1f77b4"),
        ("MAP@10", "map@k", "#ff7f0e"),
        ("nDCG@10", "ndcg@k", "#2ca02c"),
    ]

    labels = [label for _, label in model_specs]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    for index, (metric_label, metric_key, color) in enumerate(metric_specs):
        offsets = x + (index - 1) * width
        values = [payload["models"][model_name]["test"][metric_key] for model_name, _ in model_specs]
        bars = ax.bar(offsets, values, width=width, label=metric_label, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Сравнение базового и гибридного поиска\n128 бит, oversample_factor = 20, тестовая выборка")
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.82)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()

    target = Path(output_path)
    fig.savefig(target, format=target.suffix.lstrip("."))
    plt.close(fig)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build figures for the experimental results section.")
    parser.add_argument("--output-dir", default="docs/figures")
    parser.add_argument(
        "--comparison-2-epochs",
        default="artifacts/baseline_comparison_real_subset/comparison.json",
    )
    parser.add_argument(
        "--comparison-5-epochs",
        default="artifacts/baseline_comparison_real_subset_e5/comparison.json",
    )
    parser.add_argument(
        "--comparison-10-epochs",
        default="artifacts/baseline_comparison_real_subset_e10/comparison.json",
    )
    parser.add_argument(
        "--comparison-best-run",
        default="artifacts/baseline_comparison_real_subset_e10_ob20_cb128/comparison.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = ensure_directory(args.output_dir)
    build_epoch_dynamics_figure(
        output_dir / "epoch_dynamics_real_subset.svg",
        comparison_2_epochs=args.comparison_2_epochs,
        comparison_5_epochs=args.comparison_5_epochs,
        comparison_10_epochs=args.comparison_10_epochs,
    )
    build_best_run_comparison_figure(
        output_dir / "best_run_metrics_real_subset.svg",
        best_run_comparison=args.comparison_best_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
