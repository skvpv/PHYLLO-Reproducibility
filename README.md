# PHYLLO: PlantSeg Integrity Audit and Reproducibility Package

This repository contains the implementation and evidence package for a reproducible integrity audit of the PlantSeg semantic-segmentation benchmark. The study evaluates cross-split near-duplicate leakage, the causal effect of removing test-linked duplicates from training, label-space inflation caused by including the background class, and a component-confined corrected split.

## Main findings

- Hard cross-split near duplicates affect 186 of 1,561 released test images (11.92%; Wilson 95% CI: 10.40–13.62%).
- Removing test-linked duplicates produced positive point estimates, but the pre-registered causal criterion was not met because the confidence intervals included zero. H2 is therefore reported as rejected/inconclusive, not as evidence of no effect.
- Including background in the released 116-label metric increases mIoU by 0.44 points for SegNeXt-MSCAN-L and 0.49 points for DeepLabV3+-R101, with confidence intervals excluding zero for both models.
- Results on the corrected component-confined split are descriptive because its test set differs from the released test set.

## Repository structure

```text
implementation/
  configs/                Training configurations
  dataset_information/    Dataset, class-map, and mask-encoding records
  environment/            Conda and pip environment specifications
  scripts/                Dataset preparation, training, evaluation, and validation scripts
  splits/                 Fixed cohorts and training/validation arm manifests
publication_evidence/
  configs/                Frozen analysis configurations
  raw/                    Per-run evidence used by the analysis
  statistics/             Statistical results and cross-checks
  tables/                 Manuscript-ready tables
  figures/                Figure sources and rendered SVG, PDF, and PNG files
  scripts/                Frozen analysis and rendering code
reproduce_analysis.sh      Regenerates statistics, tables, and figures
verify_repository.sh      Validates syntax, JSON files, expected outputs, and checksums
```

## Dataset

The experiments use PlantSeg v7, available from [Zenodo](https://doi.org/10.5281/zenodo.17719108). The dataset is not redistributed in this repository. Users must obtain it from the official source and comply with its licence and terms.

The fixed split lists and derived evidence in this repository are provided for academic, non-commercial reproducibility under the conditions described in `LICENSE-DATA`.

## Environment

The executed environment used Python 3.11. The frozen specifications are available under `implementation/environment/`.

```bash
conda env create -f implementation/environment/environment.yml
conda activate phyllo
```

On headless Linux systems, use the recorded `opencv-python-headless` dependency. Model training additionally requires a CUDA-compatible PyTorch installation and the versions recorded in `pip_freeze_ACTUAL.txt`.

## Reproduce statistics, tables, and figures

From the repository root:

```bash
bash reproduce_analysis.sh
bash verify_repository.sh
```

The analysis uses a fixed bootstrap seed (`20260917`), 10,000 replicates, frozen thresholds, and the included per-image evidence. Successful execution should report:

```text
H1: CONFIRMED
H2: REJECTED
H3: CONFIRMED
tables: 14
condition-stratified rows: 144
figures: 8
```

## Retraining and reevaluation

The complete experimental pipeline is provided under `implementation/scripts/`. Training configurations for SegNeXt-MSCAN-L and DeepLabV3+-R101 are under `implementation/configs/`. The fixed default, sanitized, and size-control arms are under `implementation/splits/`.

Full retraining requires the separately downloaded PlantSeg dataset, compatible pretrained weights, and a CUDA-capable GPU. Paths are intentionally not hard-coded; configure dataset and output locations for the local system before execution.

## Statistical design

- Primary leakage operating point: pHash distance at most 3 or CLIP similarity at least 0.98.
- Fixed evaluation cohort: 1,557 images after excluding 12 defective masks.
- H2 comparison: paired image-level bootstrap over identical evaluation images, with seed averaging within each baseline and Holm correction across baselines.
- H3 comparison: released 116-label mIoU minus disease-only mIoU, evaluated using the same paired bootstrap design.
- Corrected-split results: descriptive and non-paired.

## Limitations

- H2 is underpowered with three SegNeXt seeds and one DeepLabV3+ seed; its confidence intervals include zero.
- Two representative architectures were evaluated; broader architectural generalization remains untested.
- Near-duplicate prevalence depends on detector thresholds, although the frozen hard band achieved 100/100 precision in the blinded audit.
- Twelve masks inconsistent with their metadata class were excluded from the primary analysis; sensitivity results are included.
- The corrected-split comparison uses a different test set and supports descriptive, not paired causal, conclusions.
- The findings concern PlantSeg v7 and should not be generalized automatically to other plant-disease datasets.

## Licensing

- Original source code: MIT License (`LICENSE-CODE`).
- PlantSeg-derived splits, embeddings, raw evidence, statistics, tables, and figures: CC BY-NC 4.0 conditions (`LICENSE-DATA`).

## Citation

Citation metadata are provided in `CITATION.cff`. A versioned archival DOI can be added after creating the corresponding release deposit.

## Author

V. Sivakumar — GitHub: [mitsv](https://github.com/mitsv)
