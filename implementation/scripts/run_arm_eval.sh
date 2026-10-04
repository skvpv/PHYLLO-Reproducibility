#!/usr/bin/env bash
# Evaluate one trained arm x seed on the fixed 1557 cohort -> confusion npz for bootstrap.
# Usage: run_arm_eval.sh <segnext|deeplab> <default|sanitized|sizectrl> <seed>
#
# CORRECTION (provenance only; evaluation is unchanged): the run record now stores THIS
# arm's actual train/val lists instead of always default_qc. Evaluation still runs on the
# fixed cohort_1557 with the trained checkpoint, so no scientific value changes.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODEL="$1"; ARM="$2"; SEED="$3"
HERE="${PHYLLO_PKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PHYLLO_PLANTSEG_REPO:?}"; : "${PHYLLO_ARMS_DIR:?}"; : "${PHYLLO_WORK_ROOT:?}"
: "${PHYLLO_IMG_ROOT:?}"; : "${PHYLLO_ANN_ROOT:?}"; : "${PHYLLO_RESULTS_ROOT:?}"
export PHYLLO_COMPUTE_PROFILE_ID="${PHYLLO_COMPUTE_PROFILE_ID:-CP-02}"
case "$MODEL" in
  segnext) CFG="$HERE/configs/phyllo_segnext_mscan-l_512x512_40k.py"; BNAME="segnext-l";;
  deeplab) CFG="$HERE/configs/phyllo_deeplabv3plus_r101_512x512_160k.py"; BNAME="deeplabv3plus-r101";;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
# Record the arm's real lists (metadata for the run record), matching run_arm_train.sh.
case "$ARM" in
  default)   TRAIN_LIST="$PHYLLO_ARMS_DIR/train_default_qc.txt"; VAL_LIST="$PHYLLO_ARMS_DIR/val_default_qc.txt" ;;
  sanitized) TRAIN_LIST="$PHYLLO_ARMS_DIR/train_sanitized.txt";  VAL_LIST="$PHYLLO_ARMS_DIR/val_sanitized.txt" ;;
  sizectrl)  TRAIN_LIST="$PHYLLO_ARMS_DIR/train_sizectrl.txt";   VAL_LIST="$PHYLLO_ARMS_DIR/val_sizectrl.txt" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
export PHYLLO_SEED="$SEED"
export PHYLLO_COHORT_LIST="$PHYLLO_ARMS_DIR/cohort_1557.txt"
export PHYLLO_TRAIN_LIST="$TRAIN_LIST"
export PHYLLO_VAL_LIST="$VAL_LIST"
WORKDIR="$PHYLLO_WORK_ROOT/$MODEL/$ARM/seed$SEED"
RUN_ID="${PHYLLO_RUN_ID:?}"
record_status() {
  python "$HERE/scripts/write_run_record.py" --run-id "$RUN_ID" --model "$MODEL" \
    --arm "$ARM" --seed "$SEED" --status "$1" --config "$CFG" \
    --train-list "$TRAIN_LIST" --val-list "$VAL_LIST" \
    --test-list "$PHYLLO_COHORT_LIST" --out "$WORKDIR/run_record.json"
}
trap 'record_status FAILED' ERR
CKPT="$(ls -t "$WORKDIR"/iter_*.pth 2>/dev/null | head -1 || true)"
if [ -f "$WORKDIR/last_checkpoint" ]; then
  LAST="$(head -n 1 "$WORKDIR/last_checkpoint" | tr -d '\r')"
  [ -f "$LAST" ] || LAST="$WORKDIR/$LAST"
  [ -f "$LAST" ] && CKPT="$LAST"
fi
[ -z "$CKPT" ] && { echo "no checkpoint in $WORKDIR"; exit 3; }
OUT="$PHYLLO_RESULTS_ROOT/$BNAME/$ARM"; mkdir -p "$OUT"
PYTHONPATH="$HERE/src:$PHYLLO_PLANTSEG_REPO${PYTHONPATH:+:$PYTHONPATH}" python -m phyllo.eval.evaluate_arm \
  --config "$CFG" --checkpoint "$CKPT" \
  --cohort-list "$PHYLLO_ARMS_DIR/cohort_1557.txt" \
  --img-root "$PHYLLO_IMG_ROOT" --ann-root "$PHYLLO_ANN_ROOT" \
  --out-npz "$OUT/seed$SEED.npz" --run-id "${PHYLLO_RUN_ID:?}"
record_status COMPLETED; trap - ERR
echo "[run_arm_eval] wrote $OUT/seed$SEED.npz"
