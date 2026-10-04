# Baseline_Config_Freeze — PHYLLO-001 (frozen at P1.GD via CR-001)

Source of truth: github.com/tqwei05/PlantSeg @ commit `1a3dd4d9224bcc97a5850af7dd1c423abc24eae0`. All values transcribed from the released configs (resolving `_base_` inheritance). `run.sh` invokes exactly 7 configs on a single GPU (`CUDA_VISIBLE_DEVICES=0`). Every config sets `num_classes=116`, `reduce_zero_label=False`, `ignore_index=255`.

## Effective configuration — the 7 released benchmark baselines

| Baseline (config) | Optimizer | LR | Sched / max_iters | Per-GPU batch | Effective batch (1 GPU) | Crop | Pretrained |
|---|---|---|---|---|---|---|---|
| deeplabv3_r50-d8 …40k… | SGD (mom 0.9, wd 5e-4) | 0.01 | PolyLR pow0.9 / **160k** | 4 | 4 | 512×512 | R50_v1c |
| deeplabv3_r101-d8 …40k… | SGD | 0.01 | PolyLR / **160k** | 4 | 4 | 512×512 | R101_v1c |
| deeplabv3plus_r50-d8 …40k… | SGD | 0.01 | PolyLR / **160k** | 4 | 4 | 512×512 | R50_v1c |
| **deeplabv3plus_r101-d8 …40k…** (PRIMARY-2) | SGD | 0.01 | PolyLR / **160k** | 4 | 4 | 512×512 | `open-mmlab://resnet101_v1c` |
| san_b16_plantseg115 | AdamW (wd 1e-4) | 1e-4 | schedule_160k / 160k | 8 | 8 | 640×640 | CLIP ViT-B/16-224 |
| **san_l14_plantseg115** (OPTIONAL/CR) | AdamW (wd 1e-4) | 1e-4 | schedule_160k / 160k | 4 | 4 | 640×640 | CLIP ViT-L/14-336 (`…0b5df9cb.pth`) |
| **segnext_mscan-l …40k…** (PRIMARY-1) | AdamW (β 0.9,0.999; wd 0.01) | 6e-5 | schedule_40k / **40k** | 16 | 16 | 512×512 | MSCAN-L (`…cef260d4.pth`) |

Common (deeplab/segnext) train pipeline: RandomResize scale (2048,512) ratio (0.5,2.0) → RandomCrop 512 (cat_max_ratio 0.75) → RandomFlip 0.5 → PhotoMetricDistortion. SAN: crop 640, cat_max_ratio 1.0, ResizeShortestEdge. Checkpoint/val interval 16k (schedule_160k) / per schedule_40k. Pretrained-checkpoint SHA-256 to be captured on first P2 download from these frozen immutable openmmlab URLs.

## Paper-vs-code discrepancies → AUDIT FINDINGS (ASR-008 / claim C-04)
1. **Optimizer:** paper states SGD uniformly; released code = SGD (DeepLab) but **AdamW** (SAN, SegNeXt).
2. **Learning rate:** paper 0.001; code = **0.01** (DeepLab), **1e-4** (SAN), **6e-5** (SegNeXt).
3. **Batch size:** paper 16; effective single-GPU = **4** (DeepLab, SAN-L14), **8** (SAN-B16), **16** (SegNeXt). The `4xb4` filename implies a 4-GPU×4 design (global 16) but `run.sh` runs on one GPU.
4. **Iterations:** filenames say `40k`; DeepLab & SAN inherit **schedule_160k (160k)**; only SegNeXt uses schedule_40k (40k).
5. **Weight decay:** paper 5e-4; code = 5e-4 (DeepLab), 1e-4 (SAN), **1e-2** (SegNeXt).

These confirm the paper's single stated recipe (SGD/lr 0.001/batch 16) does not describe the released heterogeneous configs — direct evidence for the reproducibility/comparability audit (CT-01/CT-02).

## Frozen re-benchmark baseline set (implementable; from run.sh only)
- **PRIMARY-1 (3 seeds):** SegNeXt-MSCAN-L — cheapest (40k) top-tier (44.52 mIoU) hybrid.
- **PRIMARY-2 (1 seed):** DeepLabv3+-R101 — representative CNN (160k, SGD).
- **OPTIONAL / budget-permitting (CR-gated):** SAN-ViT-L/14 (transformer, 160k, heaviest); CT-03 BLV (needs a derived config → freeze via CR before any run).
ConvNeXt-L and UPerNet-Swin-L are NOT invoked by run.sh and are REMOVED from scope (no invented configs in P2).
