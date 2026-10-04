#!/usr/bin/env bash
# Sensitivity A/B only; cannot support H1-H3. Usage: MODEL raw|relabel [seed]
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODEL="$1"; MODE="$2"; SEED="${3:-0}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${PHYLLO_SENSITIVITY_DIR:?}"; : "${PHYLLO_WORK_ROOT:?}"; : "${PHYLLO_RESULTS_ROOT:?}"
: "${PHYLLO_IMG_ROOT:?}"; : "${PHYLLO_ANN_ROOT:?}"; : "${PHYLLO_ARMS_DIR:?}"
case "$MODEL" in
  segnext) CFG="$ROOT/configs/phyllo_segnext_mscan-l_512x512_40k.py"; BNAME=segnext-l ;;
  deeplab) CFG="$ROOT/configs/phyllo_deeplabv3plus_r101_512x512_160k.py"; BNAME=deeplabv3plus-r101 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
LIST="$PHYLLO_SENSITIVITY_DIR/test_raw_1561.txt"; RELABEL=()
[ "$MODE" = relabel ] && RELABEL=(--relabel-map "$PHYLLO_SENSITIVITY_DIR/relabel_map.json")
[ "$MODE" = raw ] || [ "$MODE" = relabel ] || { echo "mode must be raw or relabel"; exit 2; }
WORK="$PHYLLO_WORK_ROOT/$MODEL/default/seed$SEED"
CKPT="$(ls -t "$WORK"/iter_*.pth | head -1)"
export PHYLLO_PLANTSEG_REPO PHYLLO_IMG_ROOT PHYLLO_ANN_ROOT PHYLLO_SEED="$SEED"
export PHYLLO_TRAIN_LIST="$PHYLLO_ARMS_DIR/train_default_qc.txt"
export PHYLLO_VAL_LIST="$PHYLLO_ARMS_DIR/val_default_qc.txt" PHYLLO_COHORT_LIST="$LIST"
OUT="$PHYLLO_RESULTS_ROOT/sensitivity/$BNAME/$MODE"; mkdir -p "$OUT"
PYTHONPATH="$ROOT/src:$PHYLLO_PLANTSEG_REPO${PYTHONPATH:+:$PYTHONPATH}" python -m phyllo.eval.evaluate_arm --config "$CFG" \
  --checkpoint "$CKPT" --cohort-list "$LIST" --img-root "$PHYLLO_IMG_ROOT" \
  --ann-root "$PHYLLO_ANN_ROOT" --out-npz "$OUT/seed$SEED.npz" \
  --run-id "${PHYLLO_RUN_ID:?}" "${RELABEL[@]}"
