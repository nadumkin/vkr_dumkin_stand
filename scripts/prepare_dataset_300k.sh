#!/usr/bin/env bash
# Подготовка увеличенного набора данных (~300K текстов в корпусе).
#
# Берёт по 50 000 train-строк и 10 000 eval-строк из каждого датасета
# (allnli, stsb, qqp). После дедупликации в корпус попадает ~250–300K
# уникальных текстов; обучающая выборка триплетов вырастает до ~25–40K,
# валидация и тест — до ~3–5K запросов в каждом.
#
# Запуск:
#   cd /Users/nikita/Documents/Docs/ВКР/src
#   bash scripts/prepare_dataset_300k.sh
#
# Время: ~10–15 мин (загрузка датасетов из кэша + дедупликация).
# Диск: ~70–100 МБ под data/real_300k/.

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

OUTPUT_DIR="data/real_300k"

if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/corpus.jsonl" ]; then
  echo "[INFO] $OUTPUT_DIR уже существует. Удалить и подготовить заново? (Ctrl+C — отмена)"
  read -r -p "  yes/no: " ans
  if [ "$ans" = "yes" ]; then
    rm -rf "$OUTPUT_DIR"
  else
    echo "[INFO] оставляю существующий каталог. Если нужно перезаписать — удалите вручную."
    exit 0
  fi
fi

echo "[INFO] подготовка $OUTPUT_DIR с лимитами 50K train / 10K eval на каждом датасете..."
started=$(date +%s)

"$PY" -m hybrid_search.cli prepare-datasets \
  --output-dir "$OUTPUT_DIR" \
  --datasets allnli,stsb,qqp \
  --holdout-fraction 0.5 \
  --limit-train-rows 50000 \
  --limit-eval-rows 10000 \
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
echo "[NEXT] для запуска sweep'а на новых данных передайте --data-dir $OUTPUT_DIR:"
echo "  $PY scripts/run_hash_training_grid.py --code-bits 128 --data-dir $OUTPUT_DIR"
echo "  $PY scripts/run_experiment_sweep.py --blocks 1,2,3 --data-dir $OUTPUT_DIR"
