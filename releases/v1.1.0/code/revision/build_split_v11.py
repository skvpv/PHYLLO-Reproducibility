#!/usr/bin/env python3
"""PHYLLO corrected partition v1.1 — group construction and class-exact allocation (builder v2).

APPROVED DESIGN (unchanged from the 4 Nov 2026 decision):
  * Grouping graph = frozen hard edges  UNION  soft edges confirmed as C1-C3 by the blinded
    review (taken from a v3-scorer output folder whose STATUS.json is COMPLETE), plus — by
    explicit policy — the soft pairs given the binary consensus "duplicate" in round 1.
  * Allocation objective (image counts, every image weighs the same):
        J = sum_c sum_s |n_cs - tau_s N_c|  +  sum_s |n_s - tau_s N|,   tau = (0.70, 0.10, 0.20)
  * Groups are never split; the 12 mask-QC exclusions stay excluded (population = the 7,762
    keys of the frozen v1.0 assignment); every class occurring in >= 3 groups must appear in
    all three partitions; deterministic.

ALGORITHM: (i) order groups by decreasing size, rarest-class group frequency, smallest member
key; (ii) greedy placement minimizing J (ties train > val > test); (iii) coverage repair;
(iv) first-improvement local search over single-group moves and same-class swaps until no
move lowers J by > 1e-9 without breaking coverage.

GROUPING POLICIES (mutually exclusive):
  --scored-dir        approved rule: hard + Round-3 C1-C3 (+ round-1 duplicates by policy)
  --dry-run-no-review hard (+ round-1 duplicates by policy); labelled DRY RUN
  --group-all-soft    CR-PHYLLO-R3-01 Option C, PROVISIONAL: hard + ALL frozen soft candidate
                      pairs ("candidate containment"). Soft pairs are candidates, not confirmed
                      duplicates; no review output is read. Builder v3.

BUILDER v2 SAFEGUARDS (unchanged in v3):
  * frozen inputs must be listed in, and match, MANIFEST.sha256;
  * --scored-dir must be a v3-scorer output with status COMPLETE; the confirmed list must
    match the SHA-256 recorded in its STATUS.json; every confirmed pair must be a frozen soft
    edge (anything else is rejected); confirmed pairs touching QC-excluded images are listed;
  * --dry-run-no-review (explicit) builds without review edges, labelled DRY RUN;
  * per-image class from Category_Map filename prefix, validated against every v1.0 group's
    recorded class set;
  * hard failure (exit 2, nothing promoted) on: input problems, coverage violations, any
    grouping edge split across partitions, any image missing or duplicated;
  * --out must not exist; staged in .<out>.incomplete and renamed after BUILD_STATUS.json.

Usage:
  python3 -I build_split_v11.py --repo <PHYLLO-Reproducibility> --out <new dir> \
      (--scored-dir <v3 scorer output> | --dry-run-no-review) \
      --round1-soft-policy {include,exclude}
Exit codes: 0 built; 2 refused/failed (reason in stderr and, if staged, BUILD_FAILED.txt).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "val", "test")
TAU = (0.70, 0.10, 0.20)
RAW = "publication_evidence/raw"
INPUTS = {
    "v10_manifest": f"{RAW}/RUN-014__corrected_split_manifest.json",
    "edges": f"{RAW}/RUN-001__edges.json",
    "round1_key": f"{RAW}/RUN-001__manual_audit_KEY.json",
    "round1_combined": f"{RAW}/RUN-001__manual_audit_reviewer_combined.json",
    "category_map": f"{RAW}/Category_Map.csv",
}


class BuildError(Exception):
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


def verify_inputs(repo):
    man = Path(repo, "MANIFEST.sha256")
    if not man.is_file():
        raise BuildError(f"{man} not found; frozen inputs cannot be verified")
    listed = {}
    with open(man, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                h, path = line.rstrip("\n").split(None, 1)
                listed[os.path.normpath(path.strip())] = h
    out = {}
    for rel in INPUTS.values():
        p = Path(repo, rel)
        if not p.is_file():
            raise BuildError(f"frozen input missing: {rel}")
        exp = listed.get(os.path.normpath(rel))
        if exp is None:
            raise BuildError(f"frozen input not listed in MANIFEST.sha256: {rel}")
        got = sha256(p)
        if exp != got:
            raise BuildError(f"checksum mismatch for {rel}: manifest {exp}, file {got}")
        out[rel] = got
    out["MANIFEST.sha256"] = sha256(man)
    return out


def load_confirmed(scored_dir, soft_set):
    """Confirmed C1-C3 target pairs from a COMPLETE v3-scorer output folder."""
    sd = Path(scored_dir)
    st_p, conf_p = sd / "STATUS.json", sd / "confirmed_soft_duplicates.csv"
    if not st_p.is_file():
        raise BuildError(f"{st_p} not found; --scored-dir must be a v3 scorer output folder")
    st = load_json(st_p)
    if st.get("status") != "COMPLETE":
        raise BuildError(f"scorer status is {st.get('status')!r}, not COMPLETE")
    exp = (st.get("artifact_sha256") or {}).get("confirmed_soft_duplicates.csv")
    if not conf_p.is_file() or exp is None:
        raise BuildError("confirmed_soft_duplicates.csv missing or not recorded in STATUS.json")
    if sha256(conf_p) != exp:
        raise BuildError("confirmed_soft_duplicates.csv does not match the SHA-256 in STATUS.json")
    pairs, bad = [], []
    with open(conf_p, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), 2):
            p = tuple(sorted(((r.get("a") or "").strip(), (r.get("b") or "").strip())))
            if not p[0] or p[0] == p[1]:
                bad.append(f"line {i}: invalid pair {p}")
            elif p not in soft_set:
                bad.append(f"line {i}: {p[0]} | {p[1]} is not a frozen soft edge")
            else:
                pairs.append(p)
    if bad:
        raise BuildError(f"{len(bad)} invalid confirmed pair(s): {bad[:5]}")
    if len(set(pairs)) != len(pairs):
        raise BuildError("confirmed list contains a repeated pair")
    return pairs, {"scored_dir": str(sd.resolve()), "scorer_status_sha256": sha256(st_p),
                   "confirmed_sha256": exp, "scorer_input_sha256": st.get("input_sha256"),
                   "scorer_script_sha256": st.get("script_sha256")}


def class_map(repo, keys, asg):
    with open(Path(repo, INPUTS["category_map"]), newline="", encoding="utf-8") as f:
        cmap = list(csv.DictReader(f))
    names = sorted(((r["Disease"].lower().replace(" ", "_"), int(r["Index"]) + 1) for r in cmap),
                   key=lambda t: (-len(t[0]), t[0]))
    cls = {}
    for k in keys:
        stem = k.split("/", 1)[1].lower()
        hit = [lab for nm, lab in names if stem.startswith(nm + "_")]
        if not hit:
            raise BuildError(f"no Category_Map class matches {k}")
        cls[k] = hit[0]
    by_gid = defaultdict(set)
    for k in keys:
        by_gid[asg[k]["gid"]].add(k)
    for gid, ks in by_gid.items():
        if {cls[k] for k in ks} != set(asg[next(iter(ks))]["classes"]):
            raise BuildError(f"class validation failed for v1.0 group {gid}")
    return cls


class UF:
    def __init__(self, nodes):
        self.p = {n: n for n in nodes}

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


class State:
    def __init__(self, groups, gcls, N_c, N):
        self.groups, self.gcls, self.N_c, self.N = groups, gcls, N_c, N
        self.n_cs = {c: [0, 0, 0] for c in N_c}
        self.g_cs = {c: [0, 0, 0] for c in N_c}
        self.n_s = [0, 0, 0]
        self.ngroups_c = Counter(c for g in gcls for c in g)
        self.asg = {}

    def _cterm(self, c, row):
        return sum(abs(row[i] - TAU[i] * self.N_c[c]) for i in range(3))

    def _gterm(self, row):
        return sum(abs(row[i] - TAU[i] * self.N) for i in range(3))

    def J(self):
        return sum(self._cterm(c, self.n_cs[c]) for c in self.N_c) + self._gterm(self.n_s)

    def delta_move(self, g, src, dst):
        d = 0.0
        for c, cnt in self.gcls[g].items():
            row = self.n_cs[c]; new = row[:]
            if src is not None:
                new[src] -= cnt
            new[dst] += cnt
            d += self._cterm(c, new) - self._cterm(c, row)
        sz = len(self.groups[g]); new = self.n_s[:]
        if src is not None:
            new[src] -= sz
        new[dst] += sz
        return d + self._gterm(new) - self._gterm(self.n_s)

    def move(self, g, src, dst):
        for c, cnt in self.gcls[g].items():
            if src is not None:
                self.n_cs[c][src] -= cnt; self.g_cs[c][src] -= 1
            self.n_cs[c][dst] += cnt; self.g_cs[c][dst] += 1
        sz = len(self.groups[g])
        if src is not None:
            self.n_s[src] -= sz
        self.n_s[dst] += sz
        self.asg[g] = dst

    def breaks_coverage(self, g, src):
        return any(self.ngroups_c[c] >= 3 and self.g_cs[c][src] <= 1 for c in self.gcls[g])

    def violations(self):
        return [c for c in sorted(self.N_c) if self.ngroups_c[c] >= 3 and min(self.g_cs[c]) == 0]


def allocate(groups, gcls, N_c, N, min_name, rarest, repair=True):
    st = State(groups, gcls, N_c, N)
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), rarest[g], min_name[g]))
    for g in order:
        best = min(range(3), key=lambda s: (st.delta_move(g, None, s), s))
        st.move(g, None, best)
    for _ in range(3 * len(groups) if repair else 0):
        v = st.violations()
        if not v:
            break
        c = v[0]; dst = st.g_cs[c].index(0)
        cands = [(st.delta_move(g, st.asg[g], dst), min_name[g], g) for g in order
                 if c in gcls[g] and st.asg[g] != dst and not st.breaks_coverage(g, st.asg[g])]
        if not cands:
            break
        _, _, g = min(cands); st.move(g, st.asg[g], dst)
    by_class = defaultdict(list)
    for g in order:
        for c in gcls[g]:
            by_class[c].append(g)
    passes = 0
    while True:
        passes += 1; improved = False
        for g in order:
            src = st.asg[g]
            if st.breaks_coverage(g, src):
                continue
            for dst in range(3):
                if dst != src and st.delta_move(g, src, dst) < -1e-9:
                    st.move(g, src, dst); improved = True; break
        for c in sorted(by_class):
            gs = by_class[c]
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    a, b = gs[i], gs[j]; sa, sb = st.asg[a], st.asg[b]
                    if sa == sb:
                        continue
                    d1 = st.delta_move(a, sa, sb); st.move(a, sa, sb)
                    d2 = st.delta_move(b, sb, sa); st.move(b, sb, sa)
                    touched = set(gcls[a]) | set(gcls[b])
                    ok = (d1 + d2 < -1e-9) and all(st.ngroups_c[k] < 3 or min(st.g_cs[k]) > 0 for k in touched)
                    if ok:
                        improved = True
                    else:
                        st.move(a, sb, sa); st.move(b, sa, sb)
        if not improved:
            break
    return st, passes


def build(args, staging):
    repo = Path(args.repo)
    input_sha = verify_inputs(repo)
    asg10 = load_json(repo / INPUTS["v10_manifest"])["assignments"]
    keys = sorted(asg10); kset = set(keys)
    edges = load_json(repo / INPUTS["edges"])
    hard = {tuple(sorted(e)) for e in edges["hard_edges"]}
    soft = {tuple(sorted(e)) for e in edges["soft_edges"]}
    universe = {x for p in hard | soft for x in p} | kset
    cls = class_map(repo, keys, asg10)

    key1 = load_json(repo / INPUTS["round1_key"])
    comb = {c["pair_id"]: c for c in load_json(repo / INPUTS["round1_combined"])}
    round1_dup = sorted(tuple(sorted(k["mapping"].values())) for k in key1
                        if k["candidate_kind"] == "soft" and comb[k["pair_id"]]["consensus_verdict"] == "duplicate")
    for p in round1_dup:
        if p not in soft:
            raise BuildError(f"round-1 soft duplicate {p} is not a frozen soft edge")
    review_meta = {"mode": "DRY RUN (no Round-3 review edges)"}
    review_pairs = []
    if args.group_all_soft:
        # CR-PHYLLO-R3-01 (Option C): candidate containment. Every frozen soft candidate pair is a
        # grouping edge, irrespective of any review label. These pairs are CANDIDATES, not
        # confirmed duplicates; no review output is read or required.
        review_meta = {"mode": "candidate containment: all frozen soft edges (CR-PHYLLO-R3-01)",
                       "change_request": "CR-PHYLLO-R3-01"}
        extra = sorted(soft)
    else:
        if args.scored_dir:
            review_pairs, review_meta = load_confirmed(args.scored_dir, soft)
            review_meta["mode"] = "Round-3 confirmed pairs from COMPLETE scorer output"
        extra = sorted(set(review_pairs) | (set(round1_dup) if args.round1_soft_policy == "include" else set()))
    for p in extra:
        if not set(p) <= universe:
            raise BuildError(f"grouping edge endpoint outside the image universe: {p}")
    qc_skipped = [p for p in extra if not set(p) <= kset]
    hard_used = sorted(p for p in hard if set(p) <= kset)
    soft_used = [p for p in extra if set(p) <= kset]

    uf = UF(keys)
    for a, b in hard_used + soft_used:
        uf.u(a, b)
    comp = defaultdict(list)
    for k in keys:
        comp[uf.f(k)].append(k)
    groups = sorted((sorted(v) for v in comp.values()), key=lambda m: m[0])
    gcls = [Counter(cls[k] for k in m) for m in groups]
    N_c = Counter(cls[k] for k in keys); N = len(keys)
    freq = Counter(c for g in gcls for c in g)
    rarest = [min(freq[c] for c in g) for g in gcls]
    min_name = [m[0] for m in groups]
    st, passes = allocate(groups, gcls, N_c, N, min_name, rarest, repair=not args._disable_repair)

    # ---------------- post-conditions (hard failures)
    assignments = {}
    for gi, m in enumerate(groups):
        for k in m:
            assignments[k] = {"gid": gi, "split": SPLITS[st.asg[gi]], "group_size": len(m),
                              "classes": sorted(gcls[gi]), "class": cls[k]}
    errs = []
    if set(assignments) != kset or len(assignments) != N:
        errs.append("assignment does not cover the 7,762-key population exactly once")
    split_edges = [p for p in hard_used + soft_used if assignments[p[0]]["split"] != assignments[p[1]]["split"]]
    if split_edges:
        errs.append(f"{len(split_edges)} grouping edge(s) split across partitions, e.g. {split_edges[:3]}")
    viol = st.violations()
    if viol:
        errs.append(f"coverage violated for classes {viol[:10]} (present in >= 3 groups but absent from a partition)")
    if errs:
        raise BuildError("; ".join(errs))

    lists = {}
    for s in SPLITS:
        body = "".join(k + "\n" for k in sorted(k for k, v in assignments.items() if v["split"] == s)).encode()
        (staging / f"{s}_corrected_v1.1.txt").write_bytes(body)
        lists[s] = hashlib.sha256(body).hexdigest()
    rows = []
    for c in sorted(N_c):
        r = {"label": c, "n_images": N_c[c], "n_groups": freq[c]}
        for i, s in enumerate(SPLITS):
            r[s] = st.n_cs[c][i]; r[f"{s}_prop"] = round(st.n_cs[c][i] / N_c[c], 6)
        r["max_abs_dev_images"] = round(max(abs(st.n_cs[c][i] - TAU[i] * N_c[c]) for i in range(3)), 6)
        r["max_abs_dev_pp"] = round(100 * max(abs(st.n_cs[c][i] / N_c[c] - TAU[i]) for i in range(3)), 6)
        rows.append(r)
    with open(staging / "per_class_counts_v1.1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(staging / "grouping_edges_v1.1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["a", "b", "source"])
        w.writerows([[a, b, "hard"] for a, b in hard_used])
        w.writerows([[a, b, "soft_candidate" if args.group_all_soft else
                      ("round3_confirmed" if (a, b) in set(review_pairs) else "round1_binary_duplicate")]
                     for a, b in soft_used])
    if qc_skipped:
        qc_name = ("soft_candidates_with_QC_excluded_endpoint.csv" if args.group_all_soft
                   else "confirmed_pairs_with_QC_excluded_endpoint.csv")
        with open(staging / qc_name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["a", "b"]); w.writerows(qc_skipped)
    dev = sorted(r["max_abs_dev_pp"] for r in rows)
    manifest = {
        "version": ("corrected partition v1.1-C — PROVISIONAL (candidate containment, CR-PHYLLO-R3-01)"
                    if args.group_all_soft else
                    "corrected partition v1.1" + ("" if args.scored_dir else " — DRY RUN (no Round-3 review edges)")),
        "builder": "build_split_v11.py v3", "total_images": N, "n_groups": len(groups),
        "grouping_policy": "group-all-soft" if args.group_all_soft else "reviewed-c1-c3",
        "provisional": bool(args.group_all_soft),
        "grouping_edges": ({"hard": len(hard_used), "soft_candidate": len(soft_used),
                            "soft_candidate_with_QC_excluded_endpoint": len(qc_skipped)} if args.group_all_soft else
                           {"hard": len(hard_used), "round3_confirmed": len([p for p in soft_used if p in set(review_pairs)]),
                            "round1_binary_duplicate": len([p for p in soft_used if p not in set(review_pairs)]),
                            "confirmed_with_QC_excluded_endpoint": len(qc_skipped)}),
        "interpretation": ("soft grouping edges are detector CANDIDATES (estimated mostly non-duplicates); "
                           "containment does not establish duplicate removal or detector recall")
                          if args.group_all_soft else "grouping edges are hard edges and human-reviewed duplicates",
        "round1_soft_policy": args.round1_soft_policy, "review_provenance": review_meta,
        "split_image_counts": dict(zip(SPLITS, st.n_s)), "objective_J_images": st.J(),
        "local_search_passes": passes, "coverage_violations": [],
        "classes_lt3_groups": [c for c in sorted(N_c) if freq[c] < 3],
        "per_class_max_abs_dev_pp": {"median": dev[len(dev) // 2], "max": dev[-1]},
        "per_class_max_abs_dev_images_max": max(r["max_abs_dev_images"] for r in rows),
        "list_sha256": lists, "input_sha256": input_sha, "assignments": assignments,
    }
    with open(staging / "corrected_split_manifest_v1.1.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return {k: v for k, v in manifest.items() if k not in ("assignments", "input_sha256")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scored-dir")
    src.add_argument("--dry-run-no-review", action="store_true")
    src.add_argument("--group-all-soft", action="store_true",
                     help="CR-PHYLLO-R3-01 Option C: group every frozen soft candidate pair (PROVISIONAL)")
    ap.add_argument("--round1-soft-policy", choices=("include", "exclude"),
                    help="required with --scored-dir/--dry-run-no-review; not allowed with --group-all-soft")
    ap.add_argument("--_disable-repair", action="store_true", help=argparse.SUPPRESS)  # test hook only
    a = ap.parse_args(argv)
    if a.group_all_soft and a.round1_soft_policy:
        ap.error("--round1-soft-policy is meaningless with --group-all-soft (all soft pairs are grouped)")
    if not a.group_all_soft and not a.round1_soft_policy:
        ap.error("--round1-soft-policy is required with --scored-dir or --dry-run-no-review")
    out = Path(a.out).resolve(); staging = out.parent / f".{out.name}.incomplete"
    refuse = None
    if out.exists() or out.is_symlink():
        refuse = f"--out {out} already exists"
    elif staging.exists():
        refuse = f"staging directory {staging} exists; inspect and remove it manually"
    elif not out.parent.is_dir():
        refuse = f"parent directory {out.parent} does not exist"
    if refuse:
        print("REFUSED: " + refuse, file=sys.stderr); return 2
    staging.mkdir()
    try:
        summary = build(a, staging)
    except BuildError as e:
        (staging / "BUILD_FAILED.txt").write_text(str(e) + "\n")
        print(f"BUILD FAILED: {e}\n(staging left at {staging} for inspection; nothing promoted)", file=sys.stderr)
        return 2
    arts = {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
    with open(staging / "BUILD_STATUS.json", "w", encoding="utf-8") as f:
        json.dump({"status": "BUILT" if (a.scored_dir or a.group_all_soft) else "BUILT_DRY_RUN",
                   "grouping_policy": "group-all-soft" if a.group_all_soft else "reviewed-c1-c3",
                   "provisional": bool(a.group_all_soft),
                   "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "builder_sha256": sha256(__file__), "args": {k: v for k, v in vars(a).items() if not k.startswith("_")},
                   "artifact_sha256": arts}, f, indent=1)
    os.rename(staging, out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
