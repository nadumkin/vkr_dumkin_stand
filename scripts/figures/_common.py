"""Общие утилиты для скриптов построения рисунков глав 4–5.

Стиль matplotlib, расчёт 95% CI, цветовая палитра, чтение артефактов.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART = PROJECT_ROOT / "artifacts"
FIG_DIR = ART / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

Z_95 = 1.96


# ---------------------------------------------------------------------------
# Стиль
# ---------------------------------------------------------------------------
def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


# Палитра — методичная (не радужная)
COLOR = {
    "dense":     "#222831",
    "lsh":       "#4d7ea8",
    "itq":       "#2e8b57",
    "pq":        "#e07b39",
    "distill":   "#a23e48",
    "joint":     "#7b3fa0",
    "binary":    "#9ab7d8",
    "rerank":    "#395e8c",
    "with_ln":   "#2c5e9b",
    "no_ln":     "#b13e3e",
}


# ---------------------------------------------------------------------------
# CI: нормальное приближение
# ---------------------------------------------------------------------------
def ci95(p: float | None, n: int) -> tuple[float, float]:
    """Возвращает (half_low, half_high) — асимметричные «усы» относительно p."""
    if p is None or n <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    half = Z_95 * se
    low = max(0.0, p - half)
    high = min(1.0, p + half)
    return (p - low, high - p)


# ---------------------------------------------------------------------------
# Чтение артефактов
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_recall(path: Path, key: str) -> tuple[float | None, int]:
    """Из comparison.json: (recall@k, n_test_queries) для модели key."""
    if not path.exists():
        return None, 0
    data = read_json(path)
    n = int(data.get("test_queries", 0))
    m = (data.get("models") or {}).get(key, {})
    if not isinstance(m, dict):
        return None, n
    r = m.get("recall@k")
    return (float(r) if r is not None else None, n)


def comparison_memory(path: Path, key: str) -> dict:
    """Из comparison.json: словарь memory.* для модели key."""
    if not path.exists():
        return {}
    data = read_json(path)
    m = (data.get("models") or {}).get(key, {})
    return m.get("memory", {}) if isinstance(m, dict) else {}


def comparison_latency(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    data = read_json(path)
    m = (data.get("models") or {}).get(key, {})
    if not isinstance(m, dict):
        return None
    v = m.get("latency_ms")
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, name: str) -> Path:
    """Сохраняет в artifacts/figures/{name}.png и .pdf."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")
    return png
