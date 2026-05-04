"""Сборка всех артефактов sweep'а в единые таблицы CSV и Markdown.

Запуск:
    cd /Users/nikita/Documents/Docs/ВКР/src
    .venv/bin/python scripts/aggregate_sweep_results.py

Выход:
    artifacts/sweep/sweep_table.csv      — машинная таблица
    artifacts/sweep/sweep_table.md       — для встраивания в главу 4
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    by_block: dict[int, list[dict]] = {}
    for row in rows:
        by_block.setdefault(int(row.get("block", 0)), []).append(row)

    lines: list[str] = []
    block_titles = {
        1: "Блок 1. Длина бинарного кода",
        2: "Блок 2. efSearch HNSW",
        3: "Блок 3. Коэффициент oversampling",
    }
    block_columns = {
        1: ["code_bits", "model", "recall@10", "ndcg@10", "map@10", "latency_ms",
            "build_time_ms", "memory_total_bytes"],
        2: ["ef_search", "recall@10", "ndcg@10", "map@10", "latency_ms",
            "candidate_selection_ms", "rerank_ms", "query_encode_ms"],
        3: ["oversample", "recall@10", "ndcg@10", "map@10", "latency_ms",
            "candidate_selection_ms", "rerank_ms"],
    }
    for block_id in sorted(by_block.keys()):
        title = block_titles.get(block_id, f"Блок {block_id}")
        cols = block_columns.get(block_id, sorted(by_block[block_id][0].keys()))
        lines.append(f"## {title}\n")
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines.append(header)
        lines.append(sep)
        for row in by_block[block_id]:
            cells = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    if "memory" in col:
                        cells.append(f"{value/1e6:.2f} MB")
                    elif "_ms" in col:
                        cells.append(f"{value:.2f}")
                    else:
                        cells.append(f"{value:.4f}")
                elif isinstance(value, int) and "memory" in col:
                    cells.append(f"{value/1e6:.2f} MB")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    sweep_root = PROJECT_ROOT / "artifacts" / "sweep"
    summary_path = sweep_root / "sweep_summary.json"
    rows = load_summary(summary_path)
    if not rows:
        print(f"[WARN] нет записей в {summary_path}. Сначала запустите run_experiment_sweep.py")
        return 1
    csv_path = sweep_root / "sweep_table.csv"
    md_path = sweep_root / "sweep_table.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"[OK] {csv_path}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
