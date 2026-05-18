#!/usr/bin/env bash
# Подготовка увеличенной выборки (~700–900K текстов в корпусе и ~100–150K
# обучающих триплетов) — для проверки гипотезы о том, что качество
# joint encoder + hash distillation ограничено объёмом supervisory сигнала.
#
# Берёт по 200 000 train-строк и 20 000 eval-строк из каждого датасета.
# STS-B упрётся в свой потолок (~5.7K train, ~1.5K val, ~1.4K test),
# AllNLI и QQP заполнят лимит. После дедупликации в корпус попадает
# примерно 700–900K уникальных текстов; обучающие триплеты — ~100–150K.
#
# Запуск:
#   cd /Users/nikita/Documents/Docs/ВКР/src
#   bash scripts/prepare_dataset_1m.sh
#
# Время: ~15–25 мин (загрузка датасетов из кэша + дедупликация на бóльшем
# объёме данных). Диск: ~250–400 МБ под data/real_1m/.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] .venv/bin/python не найден."
  exit 1
fi

PY=".venv/bin/python"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache"

OUTPUT_DIR="data/real_1m"

if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/corpus.jsonl" ]; then
  echo "[INFO] $OUTPUT_DIR уже существует. Удалить и подготовить заново? (Ctrl+C — отмена)"
  read -r -p "  yes/no: " ans
  if [ "$ans" = "yes" ]; then
    rm -rf "$OUTPUT_DIR"
  else
    echo "[INFO] оставляю существующий каталог."
    exit 0
  fi
fi

echo "[INFO] подготовка $OUTPUT_DIR с лимитами 200K train / 20K eval на каждом датасете..."
echo "[INFO] это бóльшая выборка, чем real_300k (50K/10K) — ожидаемый рост триплетов: 3–4×."
started=$(date +%s)

"$PY" -m hybrid_search.cli prepare-datasets \
  --output-dir "$OUTPUT_DIR" \
  --datasets allnli,stsb,qqp \
  --holdout-fraction 0.5 \
  --limit-train-rows 200000 \
  --limit-eval-rows 20000 \
  --cache-dir "$HF_HOME" \
  > "$OUTPUT_DIR.prep.log" 2>&1 || {
    echo "[ERROR] prepare-datasets упал. Лог:"
    tail -50 "$OUTPUT_DIR.prep.log"
    exit 1
  }

elapsed=$(( $(date +%s) - started ))
echo "[OK] подготовка заняла ${elapsed}s"

echo
echo "[SUMMARY] dataset_summary.json:"
"$PY" - <<PY
import json
from pathlib import Path
s = json.loads(Path("$OUTPUT_DIR/dataset_summary.json").read_text())
print(f"  corpus_size       = {s['corpus_size']}")
print(f"  train_triplets    = {s['train_triplets']}")
print(f"  val_queries       = {s['val_queries']}")
print(f"  test_queries      = {s['test_queries']}")
for ds, splits in s['datasets'].items():
    parts = ', '.join(f"{name}={n}" for name, n in splits.items())
    print(f"  {ds:8s} {parts}")
PY

echo
echo "[NEXT] серия joint+distillation прогонов на этой выборке (см. инструкции в чате)"
