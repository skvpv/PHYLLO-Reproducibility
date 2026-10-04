# Dataset Manifest — PHYLLO-001

## Dataset identity

- Zenodo record: `10.5281/zenodo.17719108`
- Dataset version: `v7`
- Archive: `plantseg.zip`
- Archive MD5: `9358a66dff88cdd15c4fe009763c40a3`
- ZIP integrity: `PASS`
- Total non-directory archive entries: `15552`

## Image and mask counts

| Split | Images | Masks | Images without matching mask stem | Masks without matching image stem |
|---|---:|---:|---:|---:|
| train | 5367 | 5367 | 0 | 0 |
| val | 846 | 846 | 0 | 0 |
| test | 1561 | 1561 | 0 | 0 |

- Total images: `7774`
- Total masks: `7774`

## Official split evidence

- Images and masks are stored in `train`, `val`, and `test` subdirectories.
- Split-specific annotation files are present:
  - `plantseg/annotation_train.json`
  - `plantseg/annotation_val.json`
  - `plantseg/annotation_test.json`

### Metadata split counts

- `Training`: `5367`
- `Validation`: `846`
- `Test`: `1561`

### Annotation JSON counts

| Split | JSON images | Annotation objects | Categories-table entries | Unique category IDs |
|---|---:|---:|---:|---:|
| train | 5367 | 52898 | 0 | 115 |
| val | 846 | 8927 | 0 | 115 |
| test | 1561 | 15569 | 0 | 115 |

- Category IDs present in every split: `0–114` (`115` unique IDs).
- The `categories` array in each annotation JSON contains `0` entries.

## Metadata.csv

- Path: `plantseg/Metadata.csv`
- Data rows: `7774`
- Exact column headers:

  - `Name`
  - `Index`
  - `Plant`
  - `Disease`
  - `Resolution`
  - `Label file`
  - `Mask ratio`
  - `URL`
  - `License`
  - `Split`

## Licence evidence

- Standalone licence file inside archive: `NOT PRESENT`
- Exact `Metadata.csv` licence values and row counts:
  - `CC-BY-NC`: `7741`
  - `CC0`: `33`
- Annotation JSON licence entry: `{'id': 1, 'name': '', 'url': ''}`
