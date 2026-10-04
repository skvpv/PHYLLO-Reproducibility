#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPOSITORY_ROOT"

"$PYTHON_BIN" -m compileall -q implementation/scripts publication_evidence/scripts

"$PYTHON_BIN" -c '
import json
from pathlib import Path
for path in Path(".").rglob("*.json"):
    with path.open(encoding="utf-8") as handle:
        json.load(handle)
print("JSON validation: PASS")
'

condition_count=$(find publication_evidence/statistics -maxdepth 1 \
    -name 'condition_secondary__*.json' | wc -l)
table_count=$(find publication_evidence/tables -maxdepth 1 -name 'T*.csv' | wc -l)
figure_count=$(find publication_evidence/figures/source -maxdepth 1 -name 'F*.json' | wc -l)

test "$condition_count" -eq 12
test "$table_count" -eq 14
test "$figure_count" -eq 8

if [[ -f MANIFEST.sha256 ]]; then
    sha256sum -c MANIFEST.sha256
fi

echo "Python compilation: PASS"
echo "Expected outputs: PASS (12 condition files, 14 tables, 8 figures)"
echo "Repository verification: PASS"
