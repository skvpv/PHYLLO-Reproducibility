#!/usr/bin/env bash
# Train one arm x seed on CP-02 (1 GPU). Faithful mmseg recipe; only arm/seed/paths vary.
# Usage: run_arm_train.sh <segnext|deeplab> <default|sanitized|sizectrl> <seed> [--resume]
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODEL="$1"; ARM="$2"; SEED="$3"; RESUME="${4:-}"
HERE="${PHYLLO_PKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PHYLLO_PLANTSEG_REPO:?set PHYLLO_PLANTSEG_REPO}"
: "${PHYLLO_ARMS_DIR:?set PHYLLO_ARMS_DIR (dir with train_*.txt, cohort_1557.txt)}"
: "${PHYLLO_WORK_ROOT:?set PHYLLO_WORK_ROOT}"
: "${PHYLLO_IMG_ROOT:?set PHYLLO_IMG_ROOT}"
: "${PHYLLO_ANN_ROOT:?set PHYLLO_ANN_ROOT}"
export PHYLLO_COMPUTE_PROFILE_ID="${PHYLLO_COMPUTE_PROFILE_ID:-CP-02}"   # bind frozen compute profile
case "$MODEL" in
  segnext) CFG="$HERE/configs/phyllo_segnext_mscan-l_512x512_40k.py" ;;
  deeplab) CFG="$HERE/configs/phyllo_deeplabv3plus_r101_512x512_160k.py" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
case "$ARM" in
  default)   TRAIN_LIST="$PHYLLO_ARMS_DIR/train_default_qc.txt"; VAL_LIST="$PHYLLO_ARMS_DIR/val_default_qc.txt" ;;
  sanitized) TRAIN_LIST="$PHYLLO_ARMS_DIR/train_sanitized.txt"; VAL_LIST="$PHYLLO_ARMS_DIR/val_sanitized.txt" ;;
  sizectrl)  TRAIN_LIST="$PHYLLO_ARMS_DIR/train_sizectrl.txt"; VAL_LIST="$PHYLLO_ARMS_DIR/val_sizectrl.txt" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
export PHYLLO_TRAIN_LIST="$TRAIN_LIST"
export PHYLLO_VAL_LIST="$VAL_LIST"
export PHYLLO_COHORT_LIST="$PHYLLO_ARMS_DIR/cohort_1557.txt"
export PHYLLO_SEED="$SEED"
WORKDIR="$PHYLLO_WORK_ROOT/$MODEL/$ARM/seed$SEED"
mkdir -p "$WORKDIR"
RUN_ID="${PHYLLO_RUN_ID:?Slurm launcher must bind a unique RUN ID}"
record_status() {
  python "$HERE/scripts/write_run_record.py" --run-id "$RUN_ID" --model "$MODEL" \
    --arm "$ARM" --seed "$SEED" --status "$1" --config "$CFG" \
    --train-list "$TRAIN_LIST" --val-list "$VAL_LIST" \
    --test-list "$PHYLLO_COHORT_LIST" --out "$WORKDIR/run_record.json"
}
trap 'record_status FAILED' ERR
record_status RUNNING
echo "[run_arm_train] model=$MODEL arm=$ARM seed=$SEED cfg=$CFG workdir=$WORKDIR cp=$PHYLLO_COMPUTE_PROFILE_ID"
# cudnn_benchmark=True is CONFIRMED and intentional (source-of-truth:
# configs/_base_/default_runtime.py sets cudnn_benchmark=True; effective configs retain
# deterministic=False). It must remain. No CR required.
python "$PHYLLO_PLANTSEG_REPO/tools/train.py" "$CFG" \
  --work-dir "$WORKDIR" ${RESUME:+--resume} \
  --cfg-options env_cfg.cudnn_benchmark=True
record_status TRAINING_COMPLETED
trap - ERR
echo "[run_arm_train] DONE $WORKDIR"
