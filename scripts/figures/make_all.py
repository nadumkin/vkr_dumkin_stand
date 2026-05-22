"""Сгенерировать все рисунки глав 4-5 одной командой.

Запускает по очереди 8 скриптов в scripts/figures/ и складывает
PNG + PDF в artifacts/figures/.

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/figures/make_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "fig_arch_ablation.py",
    "fig_training_schemes.py",
    "fig_256bit_comparison.py",
    "fig_length_sweep_300k.py",
    "fig_joint_trajectory.py",
    "fig_length_sweep_1m.py",
    "fig_memory_breakdown.py",
    "fig_pareto_frontier.py",
]


def main() -> int:
    py = sys.executable
    for name in SCRIPTS:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        ret = subprocess.run([py, str(THIS_DIR / name)])
        if ret.returncode != 0:
            print(f"[ERR] {name} вернул код {ret.returncode}")
            return ret.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
