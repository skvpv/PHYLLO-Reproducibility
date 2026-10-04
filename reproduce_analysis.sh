#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_ROOT="$REPOSITORY_ROOT/publication_evidence"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$EVIDENCE_ROOT"

PYTHONHASHSEED=0 "$PYTHON_BIN" scripts/make_statistics.py \
    --config configs/statistics.json --pep-root .

"$PYTHON_BIN" scripts/make_tables.py \
    --config configs/tables.json --pep-root .

"$PYTHON_BIN" scripts/make_condition_tables.py \
    --config configs/condition_tables.json --pep-root .

MPLBACKEND=Agg "$PYTHON_BIN" scripts/make_figures.py \
    --config configs/figures.json --pep-root .

echo "Analysis reproduction completed."
