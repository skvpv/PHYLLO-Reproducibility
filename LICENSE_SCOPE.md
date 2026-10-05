# Licence scope

This document clarifies the existing licences of the PHYLLO release. It does not relicense the data or grant rights held by third parties.

| Material / repository path | Applicable terms |
| --- | --- |
| Original PHYLLO source code in `implementation/scripts/`, `publication_evidence/scripts/`, `reproduce_analysis.sh` and `verify_repository.sh` | MIT, under `LICENSE-CODE`. Any third-party code remains subject to its own licence and notices. |
| PlantSeg-derived split and cohort lists in `implementation/splits/` and `publication_evidence/raw/ARMS__*` | CC BY-NC 4.0, under `LICENSE-DATA`. |
| PlantSeg-derived records in `publication_evidence/raw/`, including duplicate edges, components, embeddings, audit records, corrected assignments and per-run evaluation arrays | CC BY-NC 4.0, under `LICENSE-DATA`. |
| Derived statistics, tables and figures in `publication_evidence/statistics/`, `publication_evidence/tables/` and `publication_evidence/figures/` | CC BY-NC 4.0, under `LICENSE-DATA`. |
| Source metadata reproduced in `implementation/dataset_information/Category_Map.csv` and `publication_evidence/raw/Category_Map.csv` | Applicable PlantSeg source terms and attribution requirements remain in force; no additional rights are granted here. |
| Original PlantSeg images and masks, obtained separately | Applicable source dataset and image-level licence terms. These materials are not redistributed in this repository. |
| External dependencies and pretrained weights, obtained separately | Their respective licences; neither PHYLLO licence replaces them. |

The MIT licence applies to original software and associated documentation within its scope; it does not apply to the derived data merely because code and data share this repository. The current data licence is CC BY-NC 4.0, not CC BY 4.0. The list above describes the identified code and data categories; it does not create a new blanket licence for unlisted third-party material.

## Source attribution

PlantSeg v7: https://doi.org/10.5281/zenodo.17719108

Users must retain the attribution required by the source release and the applicable licences. Cite the PlantSeg dataset and associated paper as well as the PHYLLO resource. Partition reassignment, feature extraction and evaluation do not grant additional rights to the underlying images or masks.

## Release and archive consistency

This clarification does not change the existing v1.0.0 tag or previously deposited archives. A future archived release should include this scope document and matching licence notices. Any manuscript or repository metadata describing a particular release must report the licences actually supplied with that release.

A proposed CC BY 4.0 release of selected records requires a separate determination of the rights that can be granted for those records. No such relicensing is made by this document.
