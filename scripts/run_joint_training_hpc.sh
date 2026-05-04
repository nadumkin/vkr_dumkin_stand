#!/usr/bin/env bash
# Прямой запуск совместного дообучения на HPC (без SLURM, через SSH/tmux).
#
# ----------------------------------------------------------------------------
# ВНИМАНИЕ: обучение длится 1–4 часа на одном GPU. SSH-сессия в браузере
# закрывается при бездействии — обязательно используйте tmux/screen, чтобы
# процесс не убивался при разрыве соединения.
# ----------------------------------------------------------------------------
#
# 1) Подготовка (один раз):
#       ssh user@hpc
#       cd /path/to/ВКР/src
#       python3 -m venv .venv
#       source .venv/bin/activate
#       pip install -e '.[mlflow]'
#       # скопируйте локальные data/real_300k на HPC, если ещё не скопировано
#
# 2) Запуск под tmux:
#       tmux new -s hash      # новая сессия
#       bash scripts/run_joint_training_hpc.sh
#       # Ctrl+B, потом D — отстегнуться от сессии (скрипт продолжит работать)
#       # tmux attach -t hash — вернуться к выводу
#
#    Альтернатива через nohup (без tmux):
#       nohup bash scripts/run_joint_training_hpc.sh > logs/joint.log 2>&1 &
#       echo $! > logs/joint.pid    # запомнить PID
#       tail -f logs/joint.log      # смотреть прогресс
#       # завершить: kill $(cat logs/joint.pid)
#
# 3) Параметры можно менять прямо переменными окружения:
#       RUN_NAME=enc5e5 ENCODER_LR=5e-5 bash scripts/run_joint_training_hpc.sh
#       RUN_NAME=cb256 CODE_BITS=256 EPOCHS=2 bash scripts/run_joint_training_hpc.sh
# ----------------------------------------------------------------------------

set -euo pipefail

# --- 1. Каталог проекта ------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# --- 2. Параметры прогона ----------------------------------------------------
RUN_NAME="${RUN_NAME:-joint_default}"
DATA_DIR="${DATA_DIR:-data/real_300k}"
OUTPUT_DIR="artifacts/joint/${RUN_NAME}"
LOG_FILE="logs/joint_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
HASH_LR="${HASH_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MARGIN="${MARGIN:-4.0}"
QUANT_WEIGHT="${QUANT_WEIGHT:-0.1}"
CODE_BITS="${CODE_BITS:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"     # 1 — включить AMP, 0 — выключить

# --- 3. Окружение ------------------------------------------------------------
# venv (если есть)
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
# при использовании DataLoader workers стоит ограничить потоки BLAS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# --- 4. Печать сводки --------------------------------------------------------
{
  echo "=================================================================="
  echo "  joint training run"
  echo "=================================================================="
  echo "  date         $(date)"
  echo "  hostname     $(hostname)"
  echo "  pwd          $(pwd)"
  echo "  RUN_NAME     $RUN_NAME"
  echo "  DATA_DIR     $DATA_DIR"
  echo "  OUTPUT_DIR   $OUTPUT_DIR"
  echo "  LOG_FILE     $LOG_FILE"
  echo "  CODE_BITS    $CODE_BITS"
  echo "  EPOCHS       $EPOCHS"
  echo "  BATCH_SIZE   $BATCH_SIZE"
  echo "  ENCODER_LR   $ENCODER_LR"
  echo "  HASH_LR      $HASH_LR"
  echo "  MARGIN       $MARGIN"
  echo "  QUANT_WEIGHT $QUANT_WEIGHT"
  echo "  AMP          ${MIXED_PRECISION}"
  echo "------------------------------------------------------------------"
  python -V
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
  else
    echo "[WARN] nvidia-smi не найден — будет проверена доступность через torch"
  fi
  echo "=================================================================="
} | tee -a "$LOG_FILE"

# --- 5. Проверка зависимостей и устройства -----------------------------------
DEVICE=$(python - <<'PY' 2>>"$LOG_FILE"
import importlib.util, sys
required = ["torch", "transformers", "numpy", "hnswlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(f"[ERROR] missing packages: {missing}\n")
    sys.exit(1)
import torch
if torch.cuda.is_available():
    sys.stderr.write(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}\n")
    print("cuda")
elif torch.backends.mps.is_available():
    sys.stderr.write("[OK] MPS available\n")
    print("mps")
else:
    sys.stderr.write("[WARN] GPU не найден, использую CPU (будет очень медленно!)\n")
    print("cpu")
PY
)
echo "[INFO] device=$DEVICE" | tee -a "$LOG_FILE"

# --- 6. Сборка флагов --------------------------------------------------------
EXTRA_FLAGS=()
if [ "$MIXED_PRECISION" = "1" ] && [ "$DEVICE" = "cuda" ]; then
  EXTRA_FLAGS+=(--mixed-precision)
  echo "[INFO] mixed precision: ON" | tee -a "$LOG_FILE"
else
  echo "[INFO] mixed precision: OFF (требует CUDA)" | tee -a "$LOG_FILE"
fi

# --- 7. Запуск ---------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

CMD=(
  python scripts/run_joint_training.py
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --code-bits "$CODE_BITS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --encoder-lr "$ENCODER_LR"
    --hash-lr "$HASH_LR"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --margin "$MARGIN"
    --quantization-weight "$QUANT_WEIGHT"
    --num-workers "$NUM_WORKERS"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --log-every-n-steps 50
)
CMD+=("${EXTRA_FLAGS[@]}")

echo "[CMD] ${CMD[*]}" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------------" | tee -a "$LOG_FILE"

# Запускаем с дублированием вывода в лог (stdout и stderr)
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "------------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[OK] joint training завершено $(date)" | tee -a "$LOG_FILE"
echo "[OK] артефакты: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[OK] лог:        $LOG_FILE"
