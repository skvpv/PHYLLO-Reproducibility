# PRIMARY-1 — SegNeXt-MSCAN-L, frozen recipe (Baseline_Config_Freeze).
# BASE-FIRST: inherits the released config verbatim; P2 overrides ONLY non-claim-bearing
# fields — training-set list (arm), seed, output/eval paths, checkpoint interval. The
# frozen recipe (AdamW beta(0.9,0.999) wd 0.01, lr 6e-5, schedule_40k/40k, batch 16,
# crop 512, MSCAN-L pretrained, num_classes=116, reduce_zero_label=False,
# ignore_index=255) is untouched.
#
# Arm/seed/paths are supplied via environment variables so ONE config serves every
# seed x arm combination without duplicating the recipe:
#   PHYLLO_PLANTSEG_REPO  absolute path to tqwei05/PlantSeg @1a3dd4d
#   PHYLLO_TRAIN_LIST     arm-specific train split list (default-qc | sanitized | sizectrl)
#   PHYLLO_VAL_LIST       arm-specific validation list (never the test cohort)
#   PHYLLO_COHORT_LIST    fixed 1557 cohort list (final test evaluation only)
#   PHYLLO_SEED           training seed (0|1|2)
#   PHYLLO_WORKDIR        work_dir for checkpoints/logs (resume-safe)

_base_ = ["{{$PHYLLO_PLANTSEG_REPO:/MISSING}}/configs/segnext/segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py"]

custom_imports = dict(
    imports=["phyllo.dataset_prep.mmseg_key_dataset"],
    allow_failed_imports=False)

_SEED = {{'$PHYLLO_SEED:0'}}
randomness = dict(seed=_SEED, deterministic=False)  # cudnn determinism handled at runtime

# Arm = which images remain in TRAIN. Only the train ann_file changes across arms.
_IMG_ROOT = "{{$PHYLLO_IMG_ROOT:/MISSING}}"
_ANN_ROOT = "{{$PHYLLO_ANN_ROOT:/MISSING}}"
_DATA = dict(data_prefix=dict(img_path=_IMG_ROOT, seg_map_path=_ANN_ROOT))
train_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_TRAIN_LIST:/MISSING}}", **_DATA))
# Validation stays on the released validation partition. The fixed test cohort is
# touched only by final test/evaluation, preventing model-selection leakage.
val_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_VAL_LIST:/MISSING}}", **_DATA))
test_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_COHORT_LIST:/MISSING}}", **_DATA))

# Checkpoint every 4k iters and keep enough to resume; 7-day wall / resumable (CP-02).
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=4000,
                    max_keep_ckpts=3, save_last=True))
# work_dir is set on the CLI (--work-dir "$PHYLLO_WORKDIR"); resume via --resume.
