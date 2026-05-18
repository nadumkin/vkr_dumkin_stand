#!/usr/bin/env bash
# Ночная двухфазная серия:
#
#   PHASE 1 (real_300k):
#     для каждой длины кода (64, 128, 256, 512):
#       1. Frozen distillation v3 (1h_wide_768 + annealing + ITQ-init).
#       2. Joint distillation с теми же параметрами (--epochs 2 для скорости —
#          val плато наблюдалось после 1-2 эпохи).
#       3. Baselines comparison на real_300k с обеими подложками.
#
#   PHASE 2 (real_1m):
#     для каждой длины кода (64, 128, 512):  -- 256 уже есть из прогона D
#       4. Baselines comparison на real_1m с теми же подложками из Phase 1.
#
# 256-битная конфигурация в Phase 1 переиспользует уже готовые артефакты
# (frozen v3 и C) — для неё только baseline comparison.
#
# Запуск под фоном:
#   cd /Users/nikita/Documents/Docs/ВКР/src
#   nohup bash scripts/run_length_sweep_overnight.sh \
#       > logs/length_sweep_$(date +%Y%m%d_%H%M).log 2>&1 &
#   echo $! > logs/length_sweep.pid
#   tail -f logs/length_sweep_*.log
#
# Или интерактивно:
#   caffeinate -i bash scripts/run_length_sweep_overnight.sh
#
# Оценочное время:
#   Phase 1: ~5.5 часов (тренинги 64/128/512 + baselines 4 длины).
#   Phase 2: ~3.75 часов (baselines на real_1m для 64/128/512).
#   Итого: ~9 часов.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs \
    artifacts/distill \
    artifacts/joint_distill/length_sweep \
    artifacts/baselines/length_sweep_300k \
    artifacts/baselines/length_sweep_1m

PY=".venv/bin/python"

COOLDOWN_SECONDS=300   # 5 мин на остывание M-чипа после тяжёлых GPU-прогонов

# Параметр M для PQ как функция длины кода (bash 3.2 не поддерживает -A массивы).
pq_subspaces_for_code_bits() {
    case "$1" in
        64)  echo 8  ;;
        128) echo 16 ;;
        256) echo 32 ;;
        512) echo 64 ;;
        *)   echo 32 ;;
    esac
}

DATA_300K="data/real_300k"
DATA_1M="data/real_1m"

EPOCHS_FROZEN=8
EPOCHS_JOINT=2   # val плато достигается уже на эпохе 1-2 — экономим время
BATCH_SIZE=128
QW=0.01
HIDDEN_DIMS=768
TEMP_START=1.0
TEMP_END=10.0
ENCODER_LR=5e-6
HASH_LR=1e-3

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

cooldown() {
    log "  [cooldown] остываем ${COOLDOWN_SECONDS}s..."
    sleep "$COOLDOWN_SECONDS"
}

# Проверка наличия больших данных
if [ ! -f "$DATA_1M/corpus.jsonl" ]; then
    log "[ERROR] $DATA_1M не подготовлен. Сначала запустите: bash scripts/prepare_dataset_1m.sh"
    exit 1
fi

log "============================================================"
log "PHASE 1: real_300k — тренинги + baselines на всех длинах"
log "============================================================"
echo

for CODE_BITS in 64 128 256 512; do
    PQ_SUBS=$(pq_subspaces_for_code_bits "$CODE_BITS")

    log "------------------------------------------------------------"
    log "[Phase 1] Длина кода: $CODE_BITS бит (PQ: M=$PQ_SUBS, K=256)"
    log "------------------------------------------------------------"

    FROZEN_DIR="artifacts/distill/cb${CODE_BITS}_v3_sweep"
    JOINT_DIR="artifacts/joint_distill/length_sweep/cb${CODE_BITS}"
    BASELINES_DIR_300K="artifacts/baselines/length_sweep_300k/cb${CODE_BITS}"

    # ----- 256-бит: переиспользуем существующие артефакты -----
    if [ "$CODE_BITS" -eq 256 ]; then
        FROZEN_DIR="artifacts/distill/cb256_v3_anneal_itqinit"
        JOINT_CKPT="artifacts/joint_distill/C_arch_annealing_itqinit/checkpoints/checkpoint_epoch3"
        log "[Phase 1/256] переиспользую готовые артефакты:"
        log "  frozen: $FROZEN_DIR"
        log "  joint:  $JOINT_CKPT"
    else
        # ----- 1) Frozen v3 distillation -----
        if [ -d "$FROZEN_DIR" ] && [ -f "$FROZEN_DIR/hash_model.pt" ]; then
            log "[Phase 1/$CODE_BITS] Frozen v3 уже обучен — пропускаю"
        else
            log "[Phase 1/$CODE_BITS] обучаю frozen v3 → $FROZEN_DIR"
            "$PY" scripts/run_distillation_training.py \
                --data-dir "$DATA_300K" \
                --output-dir "$FROZEN_DIR" \
                --code-bits "$CODE_BITS" \
                --epochs "$EPOCHS_FROZEN" \
                --batch-size "$BATCH_SIZE" \
                --quantization-weight "$QW" \
                --hidden-dims "$HIDDEN_DIMS" \
                --activation gelu \
                --dropout 0.0 \
                --temperature "$TEMP_START" \
                --temperature-end "$TEMP_END" \
                --itq-init \
                --skip-baselines
            log "  Frozen v3 готов"
            cooldown
        fi

        # ----- 2) Joint distillation -----
        if [ -d "$JOINT_DIR/checkpoints/checkpoint_epoch${EPOCHS_JOINT}" ]; then
            log "[Phase 1/$CODE_BITS] Joint уже обучен — пропускаю"
        else
            log "[Phase 1/$CODE_BITS] обучаю joint distillation → $JOINT_DIR"
            "$PY" scripts/run_joint_distillation.py \
                --data-dir "$DATA_300K" \
                --output-dir "$JOINT_DIR" \
                --code-bits "$CODE_BITS" \
                --epochs "$EPOCHS_JOINT" \
                --batch-size "$BATCH_SIZE" \
                --encoder-lr "$ENCODER_LR" \
                --hash-lr "$HASH_LR" \
                --quantization-weight "$QW" \
                --hidden-dims "$HIDDEN_DIMS" \
                --activation gelu \
                --dropout 0.0 \
                --temperature "$TEMP_START" \
                --temperature-end "$TEMP_END" \
                --itq-init \
                --skip-final-baselines
            log "  Joint готов"
            cooldown
        fi

        JOINT_CKPT="$JOINT_DIR/checkpoints/checkpoint_epoch${EPOCHS_JOINT}"
    fi

    # ----- 3) Baselines comparison на real_300k -----
    if [ -d "$BASELINES_DIR_300K" ] && [ -f "$BASELINES_DIR_300K/comparison.json" ]; then
        log "[Phase 1/$CODE_BITS] Baselines (300k) уже посчитаны — пропускаю"
    else
        log "[Phase 1/$CODE_BITS] baselines comparison на real_300k → $BASELINES_DIR_300K"
        "$PY" scripts/run_baselines_comparison.py \
            --data-dir "$DATA_300K" \
            --output-dir "$BASELINES_DIR_300K" \
            --code-bits "$CODE_BITS" \
            --pq-subspaces "$PQ_SUBS" \
            --pq-centroids 256 \
            --distill-from-artifact "$FROZEN_DIR" \
            --joint-checkpoint "$JOINT_CKPT"
        log "  Baselines (300k) готовы"
    fi

    log "[Phase 1/$CODE_BITS] завершено"
    echo
done

log "============================================================"
log "PHASE 1 ЗАВЕРШЁН — переходим к real_1m baselines"
log "============================================================"
echo

# ----- PHASE 2: baselines на real_1m с переиспользованием артефактов -----
# 256 бит уже есть из прогона D, не повторяем
for CODE_BITS in 64 128 512; do
    PQ_SUBS=$(pq_subspaces_for_code_bits "$CODE_BITS")

    log "------------------------------------------------------------"
    log "[Phase 2] real_1m, длина $CODE_BITS бит (PQ: M=$PQ_SUBS)"
    log "------------------------------------------------------------"

    FROZEN_DIR="artifacts/distill/cb${CODE_BITS}_v3_sweep"
    JOINT_CKPT="artifacts/joint_distill/length_sweep/cb${CODE_BITS}/checkpoints/checkpoint_epoch${EPOCHS_JOINT}"
    BASELINES_DIR_1M="artifacts/baselines/length_sweep_1m/cb${CODE_BITS}"

    # Проверка наличия артефактов из Phase 1
    if [ ! -f "$FROZEN_DIR/hash_model.pt" ]; then
        log "[ERROR Phase 2/$CODE_BITS] нет $FROZEN_DIR/hash_model.pt — Phase 1 не завершился?"
        continue
    fi
    if [ ! -d "$JOINT_CKPT" ]; then
        log "[ERROR Phase 2/$CODE_BITS] нет $JOINT_CKPT — Phase 1 не завершился?"
        continue
    fi

    if [ -d "$BASELINES_DIR_1M" ] && [ -f "$BASELINES_DIR_1M/comparison.json" ]; then
        log "[Phase 2/$CODE_BITS] Baselines (1m) уже посчитаны — пропускаю"
    else
        log "[Phase 2/$CODE_BITS] baselines comparison на real_1m → $BASELINES_DIR_1M"
        "$PY" scripts/run_baselines_comparison.py \
            --data-dir "$DATA_1M" \
            --output-dir "$BASELINES_DIR_1M" \
            --code-bits "$CODE_BITS" \
            --pq-subspaces "$PQ_SUBS" \
            --pq-centroids 256 \
            --distill-from-artifact "$FROZEN_DIR" \
            --joint-checkpoint "$JOINT_CKPT"
        log "  Baselines (1m) готовы"
    fi

    log "[Phase 2/$CODE_BITS] завершено"
    echo
done

log "============================================================"
log "ВСЯ СЕРИЯ ЗАВЕРШЕНА"
log "============================================================"
log ""
log "Результаты:"
log "  Phase 1 (real_300k): artifacts/baselines/length_sweep_300k/cb{64,128,256,512}/"
log "  Phase 2 (real_1m):   artifacts/baselines/length_sweep_1m/cb{64,128,512}/"
log "  (256 на real_1m — переиспользуется из прогона D, не входит в Phase 2)"
log ""
log "Сводка через:"
log "  $PY scripts/aggregate_length_sweep.py"
