#!/usr/bin/env python3
"""Editor Q3 — reproducible random sample of PlantSeg source URLs for a manual primary-source check (v2).

Reads PlantSeg v7 Metadata.csv (columns Name, Split, License, URL required), draws n rows
without replacement (numpy PCG64, seed recorded), and writes a check sheet with blank columns
for the human checker: reachable (Y/N), licence_or_terms_at_source, consistent_with_metadata
(Y/N/unclear), notes. Records the SHA-256 of Metadata.csv and the full licence distribution.
No network access is made by this script; the checks are manual.

Usage: python3 -I sample_source_urls.py --metadata <Metadata.csv> --out <new.csv> [--n 50] [--seed 20261104]
Exit 0 written; 2 refused.
"""
import argparse, csv, hashlib, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REQ = ("Name", "Split", "License", "URL")
COLS = ["seed", "row", "Name", "Split", "License", "URL", "reachable", "licence_or_terms_at_source",
        "consistent_with_metadata", "notes"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr); return 2
    try:
        with open(a.metadata, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f); rows = list(rd); cols = rd.fieldnames or []
    except FileNotFoundError:
        print(f"REFUSED: {a.metadata} not found", file=sys.stderr); return 2
    miss = [c for c in REQ if c not in cols]
    if miss:
        print(f"REFUSED: Metadata.csv lacks column(s) {miss}", file=sys.stderr); return 2
    if not 1 <= a.n <= len(rows):
        print(f"REFUSED: --n {a.n} outside 1..{len(rows)}", file=sys.stderr); return 2
    blank = [i for i, r in enumerate(rows) if not (r.get("URL") or "").strip()]
    pick = sorted(np.random.Generator(np.random.PCG64(a.seed)).choice(len(rows), a.n, replace=False).tolist())
    tmp = Path(str(out) + ".incomplete")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS)
        for i in pick:
            r = rows[i]; w.writerow([a.seed, i, r["Name"], r["Split"], r["License"], r["URL"], "", "", "", ""])
    os.rename(tmp, out)
    meta = {"metadata_sha256": hashlib.sha256(Path(a.metadata).read_bytes()).hexdigest(), "n_rows": len(rows),
            "licence_distribution": dict(Counter(r["License"] for r in rows)), "rows_without_url": len(blank),
            "seed": a.seed, "n_sampled": a.n, "sample_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(str(out) + ".provenance.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())
