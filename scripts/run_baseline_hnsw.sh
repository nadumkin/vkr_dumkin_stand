#!/usr/bin/env bash
# Прогон baseline-сравнения с двумя backend-ами индекса (numpy + hnswlib)
# при одинаковых остальных параметрах. Цель — оценить, что даёт переход
# на hnswlib по латентности и сохраняется ли качество поиска.
#
# Запуск:
#   cd /Users/nikita/Documents/Docs/ВКР/src
#   bash scripts/run_baseline_hnsw.sh
#
# Время: ~5–10 минут на M-чипе (один прогон) × 2 ≈ 10–20 минут.
# Большую часть времени занимает кодирование 34 992 текстов корпуса.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] .venv/bin/python не найден."
  echo "        python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[mlflow]'"
  exit 1
fi

PY=".venv/bin/python"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Проверка зависимостей и устройства ------------------------------------------
echo "[INFO] Проверяю зависимости и устройство..."
DEVICE=$("$PY" - <<'PY'
import sys
import importlib.util
required = ["torch", "transformers", "numpy", "hnswlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(f"[ERROR] Не установлены пакеты: {missing}\n")
    sys.stderr.write("        pip install -e '.[mlflow]'\n")
    sys.exit(1)
import torch, transformers, hnswlib
hnsw_ver = getattr(hnswlib, "__version__", "unknown")
sys.stderr.write(
    f"[OK] torch={torch.__version__}, transformers={transformers.__version__}, "
    f"hnswlib={hnsw_ver}\n"
)
if torch.backends.mps.is_available():
    sys.stderr.write("[OK] MPS доступен.\n"); print("mps")
elif torch.cuda.is_available():
    sys.stderr.write("[OK] CUDA доступна.\n"); print("cuda")
else:
    sys.stderr.write("[WARN] GPU недоступен, использую CPU.\n"); print("cpu")
PY
)
echo "[INFO] device=$DEVICE"

DATA_DIR="data/real_subset"
if [ ! -f "$DATA_DIR/corpus.jsonl" ]; then
  echo "[ERROR] Нет подготовленных данных в $DATA_DIR/."
  exit 1
fi

# Общие параметры повторяют best run с transformer-энкодером
COMMON_ARGS=(
  --prepared-data-dir "$DATA_DIR"
  --encoder-backend transformers
  --transformer-model sentence-transformers/all-MiniLM-L6-v2
  --embedding-dim 384
  --batch-size 128
  --device "$DEVICE"
  --code-bits 128
  --hidden-dims 256,128
  --activation gelu
  --dropout 0.1
  --epochs 10
  --learning-rate 1e-3
  --margin 4.0
  --quantization-weight 0.1
  --top-k 10
  --oversample-factor 20
  --strategy coarse_rerank
  --hnsw-m 32
  --hnsw-ef-construction 200
  --hnsw-ef-search 64
)

# 1. Прогон с numpy backend (контрольная точка) -------------------------------
NUMPY_RUN="artifacts/baseline_comparison_real_subset_transformer_e10_ob20_cb128_numpy"
echo "[INFO] [1/2] Прогон с index-backend=numpy → $NUMPY_RUN/"
"$PY" scripts/compare_with_baseline.py \
  --run-dir "$NUMPY_RUN" \
  --index-backend numpy \
  "${COMMON_ARGS[@]}" \
  > "$NUMPY_RUN.log" 2>&1 || { echo "[ERROR] numpy-прогон упал. Лог:"; tail -30 "$NUMPY_RUN.log"; exit 1; }

# 2. Прогон с hnswlib backend -------------------------------------------------
HNSW_RUN="artifacts/baseline_comparison_real_subset_transformer_e10_ob20_cb128_hnsw"
echo "[INFO] [2/2] Прогон с index-backend=hnswlib → $HNSW_RUN/"
"$PY" scripts/compare_with_baseline.py \
  --run-dir "$HNSW_RUN" \
  --index-backend hnswlib \
  "${COMMON_ARGS[@]}" \
  > "$HNSW_RUN.log" 2>&1 || { echo "[ERROR] hnswlib-прогон упал. Лог:"; tail -30 "$HNSW_RUN.log"; exit 1; }

# 3. Сводное сравнение --------------------------------------------------------
echo "[OK] Готово. Сводка:"
"$PY" - <<PY
import json
runs = [
    ("numpy",   "$NUMPY_RUN/comparison.json"),
    ("hnswlib", "$HNSW_RUN/comparison.json"),
]
for label, path in runs:
    d = json.load(open(path))
    cfg = d["configs"]["index"]
    print(f"\n--- backend={label} (M={cfg['hnsw_m']}, efC={cfg['hnsw_ef_construction']}, efS={cfg['hnsw_ef_search']}) ---")
    for name, m in d["models"].items():
        t = m["test"]
        print(f"  {name:20s}  recall@10={t['recall@k']:.4f}  ndcg@10={t['ndcg@k']:.4f}  latency={t['latency_ms']:.2f} ms")
PY
