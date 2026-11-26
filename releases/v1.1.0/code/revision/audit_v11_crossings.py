#!/usr/bin/env python3
"""PHYLLO — audit of soft-band candidate pairs that cross a built v1.1 partition.

Reviewing the pairs that crossed v1.0 does not establish review coverage of v1.1: regrouping
moves images, so soft edges that were within one v1.0 partition can cross v1.1. This audit
classifies every frozen soft edge that crosses the built partition:

  reviewed_r3_not_grouped      blinded Round-3 decision C4/C5 (scorer v3 COMPLETE)
  reviewed_r1_binary_nondup    round-1 human binary consensus "non-duplicate" (binary rubric)
  CONFLICT_r1_dup_excluded     round-1 "duplicate" left ungrouped by --round1-soft-policy exclude
  CONFLICT_r3_grouped_comparison  Round-3 comparison item decided C1-C3 (not a grouping input)
  unreviewed                   no human decision -> written to next_review_batch.csv

Hard failures (exit 2): inputs fail verification; build or scorer artifacts do not match their
recorded SHA-256; the build's recorded confirmed-list SHA-256 differs from the scorer's; any
hard edge or grouping edge crosses partitions; a Round-3 target decided C1-C3 is not grouped.
Exit 0: CONVERGED (no unreviewed crossing pairs, no conflicts). Exit 5: NOT_CONVERGED (review
batch or researcher decision required). --out must not exist (staged, then renamed).

Usage:
  python3 -I audit_v11_crossings.py --repo <PHYLLO-Reproducibility> --build-dir <builder output>
      [--scored-dir <scorer v3 output>] --out <new dir>
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RAW = "publication_evidence/raw"
INPUTS = {"edges": f"{RAW}/RUN-001__edges.json",
          "round1_key": f"{RAW}/RUN-001__manual_audit_KEY.json",
          "round1_combined": f"{RAW}/RUN-001__manual_audit_reviewer_combined.json"}


class AuditError(Exception):
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_repo(repo):
    man = Path(repo, "MANIFEST.sha256")
    if not man.is_file():
        raise AuditError(f"{man} not found")
    listed = {}
    for line in man.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(None, 1); listed[os.path.normpath(p.strip())] = h
    out = {}
    for rel in INPUTS.values():
        p = Path(repo, rel)
        if not p.is_file() or listed.get(os.path.normpath(rel)) != sha256(p):
            raise AuditError(f"frozen input missing, unlisted or modified: {rel}")
        out[rel] = sha256(p)
    return out


def verify_artifacts(d, status_name, required):
    st_p = Path(d, status_name)
    if not st_p.is_file():
        raise AuditError(f"{st_p} not found")
    st = load_json(st_p)
    arts = st.get("artifact_sha256") or {}
    for name in required:
        if name not in arts:
            raise AuditError(f"{name} not recorded in {status_name}")
        if not Path(d, name).is_file() or sha256(Path(d, name)) != arts[name]:
            raise AuditError(f"{d}/{name} does not match the SHA-256 recorded in {status_name}")
    return st


def audit(a, staging):
    repo_sha = verify_repo(a.repo)
    bst = verify_artifacts(a.build_dir, "BUILD_STATUS.json",
                           ["corrected_split_manifest_v1.1.json", "grouping_edges_v1.1.csv"])
    if bst.get("status") not in ("BUILT", "BUILT_DRY_RUN"):
        raise AuditError(f"build status {bst.get('status')!r} is not a completed build")
    man = load_json(Path(a.build_dir, "corrected_split_manifest_v1.1.json"))
    asg = man["assignments"]
    grouping = {tuple(sorted((r["a"], r["b"]))): r["source"] for r in rcsv(Path(a.build_dir, "grouping_edges_v1.1.csv"))}
    edges = load_json(Path(a.repo, INPUTS["edges"]))
    hard = {tuple(sorted(e)) for e in edges["hard_edges"]}
    soft = {tuple(sorted(e)) for e in edges["soft_edges"]}
    split = lambda k: asg[k]["split"]
    inside = lambda p: p[0] in asg and p[1] in asg

    errs = []
    hard_cross = [p for p in hard if inside(p) and split(p[0]) != split(p[1])]
    grp_cross = [p for p in grouping if inside(p) and split(p[0]) != split(p[1])]
    if hard_cross:
        errs.append(f"{len(hard_cross)} hard edge(s) cross partitions, e.g. {hard_cross[:2]}")
    if grp_cross:
        errs.append(f"{len(grp_cross)} grouping edge(s) cross partitions, e.g. {grp_cross[:2]}")

    key1 = load_json(Path(a.repo, INPUTS["round1_key"]))
    comb = {c["pair_id"]: c for c in load_json(Path(a.repo, INPUTS["round1_combined"]))}
    r1 = {tuple(sorted(k["mapping"].values())): comb[k["pair_id"]]["consensus_verdict"]
          for k in key1 if k["candidate_kind"] == "soft"}

    r3, r3_meta = {}, {"scored_dir": None}
    if a.scored_dir:
        sp = Path(a.scored_dir, "STATUS.json")
        if not sp.is_file():
            raise AuditError(f"{sp} not found")
        if load_json(sp).get("status") != "COMPLETE":
            raise AuditError(f"scorer status {load_json(sp).get('status')!r} is not COMPLETE")
        sst = verify_artifacts(a.scored_dir, "STATUS.json", ["review_decisions.csv", "confirmed_soft_duplicates.csv"])
        conf_sha = sst["artifact_sha256"]["confirmed_soft_duplicates.csv"]
        built_sha = (man.get("review_provenance") or {}).get("confirmed_sha256")
        if built_sha != conf_sha:
            raise AuditError("the build was not made from this scorer output (confirmed-list SHA-256 differs)")
        for r in rcsv(Path(a.scored_dir, "review_decisions.csv")):
            p = tuple(sorted((r["a"], r["b"])))
            r3[p] = (r["item_kind"], r["final_group_decision"])
        r3_meta = {"scored_dir": str(Path(a.scored_dir).resolve()), "confirmed_sha256": conf_sha}
        for p, (kind, dec) in r3.items():
            if kind == "target_soft" and dec == "grouped" and inside(p) and p not in grouping:
                errs.append(f"Round-3 grouped target {p} is not a grouping edge in this build")
    if errs:
        raise AuditError("; ".join(errs))

    rows = []
    for p in sorted(soft):
        if not inside(p) or split(p[0]) == split(p[1]):
            continue
        kind, dec = r3.get(p, (None, None))
        if dec == "not_grouped":
            status = "reviewed_r3_not_grouped"
        elif dec == "grouped":
            status = "CONFLICT_r3_grouped_comparison"
        elif r1.get(p) == "non-duplicate":
            status = "reviewed_r1_binary_nondup"
        elif r1.get(p) == "duplicate":
            status = "CONFLICT_r1_dup_excluded"
        else:
            status = "unreviewed"
        rows.append({"a": p[0], "b": p[1], "split_a": split(p[0]), "split_b": split(p[1]),
                     "pattern": "-".join(sorted((split(p[0]), split(p[1])))), "status": status})
    with open(staging / "crossing_pairs_v1.1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "split_a", "split_b", "pattern", "status"])
        w.writeheader(); w.writerows(rows)
    batch = [r for r in rows if r["status"] == "unreviewed"]
    with open(staging / "next_review_batch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["a", "b"]); w.writerows([[r["a"], r["b"]] for r in batch])
    conflicts = [r for r in rows if r["status"].startswith("CONFLICT")]
    converged = not batch and not conflicts
    summary = {"verdict": "CONVERGED" if converged else "NOT_CONVERGED",
               "build_status": bst.get("status"), "build_version": man.get("version"),
               "build_manifest_sha256": bst["artifact_sha256"]["corrected_split_manifest_v1.1.json"],
               "soft_edges_frozen": len(soft), "soft_edges_crossing_v11": len(rows),
               "by_status": dict(Counter(r["status"] for r in rows)),
               "by_pattern": dict(Counter(r["pattern"] for r in rows)),
               "next_review_batch": len(batch), "conflicts": len(conflicts),
               "hard_edges_crossing": 0, "grouping_edges_crossing": 0,
               "round3": r3_meta, "repo_input_sha256": repo_sha,
               "note": "round-1 labels are binary (duplicate/non-duplicate), not C1-C5"}
    with open(staging / "crossing_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True); ap.add_argument("--build-dir", required=True)
    ap.add_argument("--scored-dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve(); staging = out.parent / f".{out.name}.incomplete"
    if out.exists() or staging.exists() or not out.parent.is_dir():
        print(f"REFUSED: --out must be new (and no leftover {staging.name}); parent must exist", file=sys.stderr)
        return 2
    staging.mkdir()
    try:
        s = audit(a, staging)
    except AuditError as e:
        (staging / "AUDIT_FAILED.txt").write_text(str(e) + "\n")
        print(f"AUDIT FAILED: {e}\n(staging left at {staging})", file=sys.stderr)
        return 2
    arts = {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
    with open(staging / "AUDIT_STATUS.json", "w", encoding="utf-8") as f:
        json.dump({"status": s["verdict"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "auditor_sha256": sha256(__file__), "artifact_sha256": arts}, f, indent=1)
    os.rename(staging, out)
    print(json.dumps({k: v for k, v in s.items() if k != "repo_input_sha256"}, indent=1))
    return 0 if s["verdict"] == "CONVERGED" else 5


if __name__ == "__main__":
    sys.exit(main())
