#!/usr/bin/env bash
# Прогон baseline-сравнения dense_exact / hybrid_untrained / hybrid_trained
# с transformer-энкодером (sentence-transformers/all-MiniLM-L6-v2).
#
# Конфигурация повторяет лучший существующий запуск
# (artifacts/baseline_comparison_real_subset_e10_ob20_cb128),
# но с реальным семантическим энкодером вместо детерминированного hashing.
#
# Запуск:
#   cd /Users/nikita/Documents/Docs/ВКР/src
#   bash scripts/run_baseline_transformer.sh
#
# Использует Apple Metal (MPS) на M-чипе — кодирование 34 992 текстов
# проходит за пару минут. Если MPS не доступен, скрипт автоматически
# откатится на CPU.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Окружение -----------------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] .venv/bin/python не найден. Создайте окружение:"
  echo "        python3 -m venv .venv && source .venv/bin/activate && pip install -e .[mlflow]"
  exit 1
fi

PY=".venv/bin/python"

# Кэш HF держим внутри проекта, чтобы модель скачалась один раз и осталась рядом
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false

# 2. Проверяем зависимости и выбираем устройство -------------------------------
echo "[INFO] Проверяю установленные пакеты и устройство..."
DEVICE=$("$PY" - <<'PY'
import sys
import importlib.util
required = ["torch", "transformers", "numpy"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(f"[ERROR] Не установлены пакеты: {missing}\n")
    sys.stderr.write("        pip install -e '.[mlflow]'\n")
    sys.exit(1)
import torch, transformers
sys.stderr.write(f"[OK] torch={torch.__version__}, transformers={transformers.__version__}\n")
if torch.backends.mps.is_available():
    sys.stderr.write("[OK] Apple Metal (MPS) доступен — используем его.\n")
    print("mps")
elif torch.cuda.is_available():
    sys.stderr.write("[OK] CUDA доступна — используем её.\n")
    print("cuda")
else:
    sys.stderr.write("[WARN] GPU недоступен, откатываюсь на CPU.\n")
    print("cpu")
PY
)
echo "[INFO] device=$DEVICE"

# Workaround: некоторые операции на MPS требуют отката на CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 3. Запуск сравнения ----------------------------------------------------------
RUN_DIR="artifacts/baseline_comparison_real_subset_transformer_e10_ob20_cb128"
DATA_DIR="data/real_subset"

if [ ! -f "$DATA_DIR/corpus.jsonl" ]; then
  echo "[ERROR] Нет подготовленных данных в $DATA_DIR/. Сначала запустите prepare-datasets."
  exit 1
fi

echo "[INFO] Запускаю baseline-сравнение с transformer-энкодером..."
echo "[INFO] Артефакты будут в $RUN_DIR/"
echo "[INFO] Первый прогон скачает модель all-MiniLM-L6-v2 (~90 МБ) в $HF_HOME"

"$PY" scripts/compare_with_baseline.py \
  --prepared-data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --encoder-backend transformers \
  --transformer-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-dim 384 \
  --batch-size 128 \
  --device "$DEVICE" \
  --code-bits 128 \
  --hidden-dims 256,128 \
  --activation gelu \
  --dropout 0.1 \
  --epochs 10 \
  --learning-rate 1e-3 \
  --margin 4.0 \
  --quantization-weight 0.1 \
  --index-backend numpy \
  --top-k 10 \
  --oversample-factor 20 \
  --strategy coarse_rerank \
  > "$RUN_DIR.log" 2>&1 || {
    echo "[ERROR] Прогон упал. Лог в $RUN_DIR.log:"
    tail -30 "$RUN_DIR.log"
    exit 1
  }

echo "[OK] Готово. Сводка:"
"$PY" - <<PY
import json
p = "$RUN_DIR/comparison.json"
d = json.load(open(p))
print(f"  artifact: {p}")
for name, m in d["models"].items():
    t = m["test"]
    print(f"  {name:20s}  recall@10={t['recall@k']:.4f}  ndcg@10={t['ndcg@k']:.4f}  latency={t['latency_ms']:.2f} ms")
PY
