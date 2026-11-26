#!/usr/bin/env bash
# Descriptive non-paired corrected-split run. Usage: MODEL [seed]
# CORRECTION: fail fast if the RUN-014 corrected 70-10-20 split lists are absent + bind CP-02.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODEL="$1"; SEED="${2:-0}"; ROOT="${PHYLLO_PKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${PHYLLO_CORRECTED_DIR:?}"; : "${PHYLLO_WORK_ROOT:?}"; : "${PHYLLO_RESULTS_ROOT:?}"
: "${PHYLLO_PLANTSEG_REPO:?}"; : "${PHYLLO_IMG_ROOT:?}"; : "${PHYLLO_ANN_ROOT:?}"
export PHYLLO_COMPUTE_PROFILE_ID="${PHYLLO_COMPUTE_PROFILE_ID:-CP-02}"   # bind frozen compute profile
case "$MODEL" in
  segnext) CFG="$ROOT/configs/phyllo_segnext_mscan-l_512x512_40k.py"; BNAME=segnext-l ;;
  deeplab) CFG="$ROOT/configs/phyllo_deeplabv3plus_r101_512x512_160k.py"; BNAME=deeplabv3plus-r101 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
export PHYLLO_TRAIN_LIST="$PHYLLO_CORRECTED_DIR/train_corrected.txt"
export PHYLLO_VAL_LIST="$PHYLLO_CORRECTED_DIR/val_corrected.txt"
export PHYLLO_COHORT_LIST="$PHYLLO_CORRECTED_DIR/test_corrected.txt"
for f in "$PHYLLO_TRAIN_LIST" "$PHYLLO_VAL_LIST" "$PHYLLO_COHORT_LIST"; do
  [ -f "$f" ] || { echo "corrected split list missing: $f (run RUN-014 build_corrected_benchmark.py first)"; exit 4; }
done
export PHYLLO_SEED="$SEED"
WORK="$PHYLLO_WORK_ROOT/$MODEL/corrected/seed$SEED"; mkdir -p "$WORK"
RUN_ID="${PHYLLO_RUN_ID:?}"
record_status() {
  python "$ROOT/scripts/write_run_record.py" --run-id "$RUN_ID" --model "$MODEL" \
    --arm corrected --seed "$SEED" --status "$1" --config "$CFG" \
    --train-list "$PHYLLO_TRAIN_LIST" --val-list "$PHYLLO_VAL_LIST" \
    --test-list "$PHYLLO_COHORT_LIST" --out "$WORK/run_record.json"
}
trap 'record_status FAILED' ERR
record_status RUNNING
RESUME=(); [ -f "$WORK/last_checkpoint" ] && RESUME=(--resume)
python "$PHYLLO_PLANTSEG_REPO/tools/train.py" "$CFG" --work-dir "$WORK" "${RESUME[@]}"
CKPT="$(ls -t "$WORK"/iter_*.pth | head -1)"
OUT="$PHYLLO_RESULTS_ROOT/$BNAME/corrected"; mkdir -p "$OUT"
PYTHONPATH="$ROOT/src:$PHYLLO_PLANTSEG_REPO${PYTHONPATH:+:$PYTHONPATH}" python -m phyllo.eval.evaluate_arm --config "$CFG" \
  --checkpoint "$CKPT" --cohort-list "$PHYLLO_COHORT_LIST" \
  --img-root "$PHYLLO_IMG_ROOT" --ann-root "$PHYLLO_ANN_ROOT" \
  --out-npz "$OUT/seed$SEED.npz" --run-id "$RUN_ID"
record_status COMPLETED; trap - ERR
