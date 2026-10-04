#!/usr/bin/env bash
# Stage a researcher-obtained PlantSeg-v7 archive without changing the source file.
# Usage: stage_dataset.sh /path/to/archive.zip /path/to/staging
set -euo pipefail
ARCHIVE="$1"; DEST="$2"; EXPECTED=9358a66dff88cdd15c4fe009763c40a3
[ -f "$ARCHIVE" ] || { echo "archive not found: $ARCHIVE"; exit 2; }
ACTUAL="$(md5sum "$ARCHIVE" | awk '{print $1}')"
[ "$ACTUAL" = "$EXPECTED" ] || { echo "MD5 mismatch: $ACTUAL"; exit 3; }
mkdir -p "$DEST"
cp -n "$ARCHIVE" "$DEST/"
unzip -n "$ARCHIVE" -d "$DEST/extracted"
echo "PlantSeg v7 staged; MD5=$ACTUAL"
