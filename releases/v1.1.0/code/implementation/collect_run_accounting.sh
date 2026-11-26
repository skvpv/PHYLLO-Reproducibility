#!/usr/bin/env bash
# Usage: collect_run_accounting.sh JOBID [output.csv]
set -euo pipefail
JOBID="$1"; OUT="${2:-results/accounting_${JOBID}.csv}"
mkdir -p "$(dirname "$OUT")"
sacct -j "$JOBID" --units=G --parsable2 \
  --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,AllocTRES,MaxRSS,MaxVMSize,ExitCode \
  > "$OUT"
echo "$OUT"
