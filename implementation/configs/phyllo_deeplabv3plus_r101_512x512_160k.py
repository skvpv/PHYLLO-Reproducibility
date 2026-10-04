# PRIMARY-2 — DeepLabv3+-R101, frozen recipe (Baseline_Config_Freeze).
# BASE-FIRST: inherits the released config verbatim; the effective schedule is
# schedule_160k / 160k (the "40k" in the filename is an mmseg naming artefact; the
# released config inherits schedule_160k — verified audit finding C-04). P2 overrides
# ONLY non-claim-bearing fields (arm train list, seed, paths, checkpoint interval).
# Frozen: SGD mom 0.9 wd 5e-4, lr 0.01, PolyLR pow0.9 / 160k, batch 4, crop 512,
# resnet101_v1c pretrained, num_classes=116, reduce_zero_label=False, ignore_index=255.

_base_ = ["{{$PHYLLO_PLANTSEG_REPO:/MISSING}}/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_plantseg115-512x512.py"]

custom_imports = dict(
    imports=["phyllo.dataset_prep.mmseg_key_dataset"],
    allow_failed_imports=False)

_SEED = {{'$PHYLLO_SEED:0'}}
randomness = dict(seed=_SEED, deterministic=False)

_IMG_ROOT = "{{$PHYLLO_IMG_ROOT:/MISSING}}"
_ANN_ROOT = "{{$PHYLLO_ANN_ROOT:/MISSING}}"
_DATA = dict(data_prefix=dict(img_path=_IMG_ROOT, seg_map_path=_ANN_ROOT))
train_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_TRAIN_LIST:/MISSING}}", **_DATA))
val_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_VAL_LIST:/MISSING}}", **_DATA))
test_dataloader = dict(dataset=dict(
    type="PhylloPlantSeg115Dataset",
    ann_file="{{$PHYLLO_COHORT_LIST:/MISSING}}", **_DATA))

# 160k schedule => checkpoint every 16k (matches schedule_160k val/ckpt interval), resumable.
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=16000,
                    max_keep_ckpts=3, save_last=True))
