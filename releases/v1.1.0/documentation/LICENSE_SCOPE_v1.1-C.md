# PHYLLO v1.1-C — Licence scope and third-party materials

**Release-candidate notice.** This document describes the intended licence scope of the derived resource. It does not grant rights held by third parties, change historical licences, or establish that journal-specific licensing requirements have been satisfied. Public distribution requires a separate rights and policy decision.

| Release path or material | Declared scope and restrictions |
| --- | --- |
| `code/` and author-owned analysis, evaluation and partitioning software | MIT under `LICENSE-CODE`, only to the extent the authors own the code. Bundled third-party code or notices must be reviewed and retain their applicable terms. |
| `qc/`, `graph/`, `splits/`, `eval/`, `statistics/`, `tables/` | PlantSeg-derived records declared CC BY-NC 4.0 under `LICENSE-DATA`, subject to source rights and attribution. |
| `review/round1/` and `review/validation_v1.1/` | Derived review data declared CC BY-NC 4.0 where redistribution rights exist; reviewer-level data also require privacy, consent and disclosure clearance. Signed declarations and private correspondence are not part of the public release. |
| `protocol/` and author-created scientific-method descriptions | Original authored documentation may accompany the release, subject to applicable authorship and rights; no claim of public preregistration is made. |
| `configs/` and `environment/` | Author-owned configuration content may be MIT; third-party source code, dependencies and version metadata retain their own terms. Local installation paths are not reusable distribution artefacts. |
| Source-derived metadata (including `qc/Category_Map.csv`) | Applicable PlantSeg source terms and attribution requirements continue to apply. |
| Original PlantSeg photographs and masks | **Not redistributed.** Obtain from https://doi.org/10.5281/zenodo.17719108 and follow source-record and image-level terms. |
| CLIP, segmentation pretrained weights, third-party software | **Not relicensed by PHYLLO.** Obtain and use under their respective terms. |

## Attribution

Users should cite the PlantSeg dataset record and associated Data Descriptor together with the PHYLLO resource. The source archive, source image-level terms and original attribution obligations remain applicable. Derived image identifiers, feature vectors and evaluation contributions do not confer ownership of the underlying photographs.

## Important unresolved release decisions

1. Verify the target journal's current requirements for licensing derived or secondary datasets and whether CC BY-NC 4.0 is acceptable for these specific records. Do not silently substitute CC BY 4.0.
2. Confirm the rights and permissions for publishing reviewer-level classifications, image-pair mappings and review notes. Anonymization alone does not prove consent.
3. Check all `code/` and `configs/` files for third-party licence notices and embedded code before treating them as MIT-covered.
4. Review the final public file inventory and Zenodo licence metadata for agreement with the actual materials and notices.

## Historical release

The original v1.0 repository and archive remain separate historical records. This scope clarification does not modify the v1.0.0 tag or retroactively relicense prior deposits.
