#!/usr/bin/env bash
#SBATCH --job-name=hash-joint
#SBATCH --output=logs/joint_%j.out
#SBATCH --error=logs/joint_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# ----------------------------------------------------------------------------
# Совместное дообучение энкодера + хэш-модуля на HPC.
#
# Перед первым запуском:
#   1. поставьте окружение в проектной папке:
#        python3 -m venv .venv
#        source .venv/bin/activate
#        pip install -e '.[mlflow]'
#   2. подготовьте данные (или скопируйте data/real_300k с локальной машины).
#
# Запуск:
#   sbatch scripts/slurm_joint_training.sh
#
# Параметры экспериментов задаются ниже в RUN_NAME / TRAINING_ARGS.
# Чтобы запустить серию (например по encoder_lr): создать N SBATCH-скриптов
# или передать аргументы через --export. Для чистоты артефактов каждая
# конфигурация пишет в свой каталог под artifacts/joint/.
# ----------------------------------------------------------------------------

set -euo pipefail

# --- 1. Конфигурация ---------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

# имя прогона можно переопределить через переменные окружения при sbatch:
#   sbatch --export=ALL,RUN_NAME=enc_lr5e5 scripts/slurm_joint_training.sh
RUN_NAME="${RUN_NAME:-joint_default}"
DATA_DIR="${DATA_DIR:-data/real_300k}"
OUTPUT_DIR="artifacts/joint/${RUN_NAME}"

# --- 2. Окружение ------------------------------------------------------------
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Cluster-friendly defaults
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Чтобы torch видел только выделенный GPU
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
fi

echo "[INFO] Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[INFO] RUN_NAME=$RUN_NAME"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi || true
python -V

# --- 3. Параметры обучения ---------------------------------------------------
# Базовая конфигурация: 3 эпохи, batch=128, encoder_lr=2e-5, hash_lr=1e-3.
# При желании можно перебить аргументы через переменные окружения.
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
MIXED_PRECISION_FLAG="${MIXED_PRECISION_FLAG:---mixed-precision}"   # пусто = выкл

# --- 4. Запуск ---------------------------------------------------------------
TRAIN_CMD=(
  python scripts/run_joint_training.py
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --device cuda
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
    --log-every-n-steps 50
)
if [ -n "$MIXED_PRECISION_FLAG" ]; then
  TRAIN_CMD+=("$MIXED_PRECISION_FLAG")
fi

echo "[CMD] ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "[OK] joint training done. Artifacts in $OUTPUT_DIR"
