# Mask Encoding Report — PHYLLO-001

## Evidence identity

- Dataset: PlantSeg v7
- Zenodo record: `10.5281/zenodo.17719108`
- Dataset archive MD5: `9358a66dff88cdd15c4fe009763c40a3`
- Repository: `https://github.com/tqwei05/PlantSeg`
- Repository commit: `1a3dd4d9224bcc97a5850af7dd1c423abc24eae0`
- Masks inspected: `7,774 of 7,774`
- Inspection method: complete programmatic scan of every PNG mask in the MD5-verified archive

## Empirical mask encoding

- All 7,774 masks are non-interlaced, 8-bit grayscale PNG files.
- Global pixel-value range: `0–115`.
- Sorted global unique values: every integer from `0` through `115`.
- Background pixel value: `0`.
- Disease pixel values: `1–115`.
- Mapping rule: Metadata class `Index=i` corresponds to mask disease value `i+1`.
- All 115 expected `Index+1` disease values were observed.
- Ignore value configured by code: `255`.
- Released masks containing value `255`: `0`.
- Masks containing no background pixel (`0`): `1`.
- Masks mapped successfully to Metadata.csv: `7,774`.
- Masks using only `{0, Index+1}`: `7,762`.
- Masks containing an unexpected disease-class value: `12`.

## Released code semantics

- `mmseg/datasets/plantseg115.py` contains 116 class entries:
  - position `0`: empty/background entry;
  - positions `1–115`: 115 disease names.
- `configs/_base_/datasets/plantseg115.py` sets `reduce_zero_label=False`.
- `BaseSegDataset` defaults to `ignore_index=255`.
- The segmentation decode head defaults to `ignore_index=255` and passes it to the loss and accuracy calculation.
- `IoUMetric` defaults to `ignore_index=255`.
- `IoUMetric` obtains `num_classes` from `len(dataset_meta['classes'])`, which is `116`.
- Every model configuration invoked by the released `run.sh` resolves to `num_classes=116`:
  - DeepLabv3 ResNet-50/101;
  - DeepLabv3+ ResNet-50/101;
  - SAN ViT-B/16 and ViT-L/14;
  - SegNeXt MSCAN-L.
- Other repository configurations containing `num_classes=115` are not the configurations invoked by `run.sh`.

## Metric-definition conclusion

The released benchmark code uses a 116-label space consisting of one background label and 115 disease labels. It does not reduce label zero, and it ignores only label 255. Because no mask contains value 255, all released mask pixels participate in the configured loss and evaluation. Consequently, the released code’s class-space definition differs from the paper’s textual presentation of `C=115` disease classes and must be explicitly handled in the re-benchmark protocol.

The re-benchmark should preserve both quantities separately:

1. reproduction metric using the released 116-label code path, including background;
2. disease-only metric over labels `1–115`, excluding background.

Neither should be silently substituted for the other.

## Verified mask-class inconsistencies

| Split | Mask | Metadata Index | Expected value | Observed disease value |
|---|---|---:|---:|---:|
| test | `broccoli_alternaria_leaf_spot_Bing_0021.png` | 23 | 24 | 25 |
| test | `broccoli_alternaria_leaf_spot_Bing_0045.png` | 23 | 24 | 25 |
| test | `broccoli_alternaria_leaf_spot_Bing_0054.png` | 23 | 24 | 25 |
| test | `wheat_head_scab_Bing_0287.png` | 104 | 105 | 106 |
| train | `bean_mosaic_virus_23.png` | 12 | 13 | 82 |
| train | `blueberry_mummy_berry_Bing_0058.png` | 20 | 21 | 22 |
| train | `broccoli_alternaria_leaf_spot_Bing_0012.png` | 23 | 24 | 25 |
| train | `broccoli_alternaria_leaf_spot_Bing_0035.png` | 23 | 24 | 25 |
| train | `broccoli_alternaria_leaf_spot_Bing_0039.png` | 23 | 24 | 25 |
| train | `cucumber_powdery_mildew_google_0067.png` | 50 | 51 | 49 |
| train | `eggplant_phomopsis_fruit_rot_Bing_0015.png` | 52 | 53 | 54 |
| train | `ginger_leaf_spot_17.png` | 56 | 57 | 58 |

## Risk disposition

- `ASR-005`: factual encoding and released-code behavior resolved and verified.
- The 12 inconsistent masks constitute a separate verified annotation-integrity defect.
- Correction policy, corrected-mask release, and sensitivity analysis must be frozen before P2 implementation.
- No source file has been modified during this audit.
