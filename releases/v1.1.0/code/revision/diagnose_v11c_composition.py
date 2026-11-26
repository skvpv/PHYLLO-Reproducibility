#!/usr/bin/env python3
"""PHYLLO CR-PHYLLO-R3-01 — class-composition and nearest-neighbour diagnostics for a built
partition (v1). Read-only with respect to the build and the frozen records.

Computes, for a build produced by build_split_v11.py (BUILD_STATUS checksums verified):
  AC2 partition checks  0 coverage violations; every class within 3 images of target in every
                        partition; split proportions within +/-0.5 pp of 70/10/20; no class with
                        > 70% of its images in one group.
  AC4 composition       per class: share of images in groups of >= 50 images (all such groups and
                        the partition they are in); criterion: no class above 50%.
  NN similarity         for every test (and val) image: max CLIP cosine and min pHash distance to
                        any TRAIN image, for this build and for the frozen v1.0 partition (RUN-014),
                        overall and per class (descriptive; no criterion). Also counts test/val-train
                        pairs inside the detector bands (d <= 9 or s >= 0.95) that are NOT frozen
                        edges (should be 0 if the frozen edge list is complete over these images).
Outputs (new --out dir): DIAGNOSTICS.json, per_class_composition.csv, nn_similarity_per_class.csv
Exit 0 all AC2/AC4 criteria pass; 6 a criterion fails (report still written); 2 refused.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

RAW = "publication_evidence/raw"
INPUTS = {"edges": f"{RAW}/RUN-001__edges.json", "embeddings": f"{RAW}/RUN-001__embeddings.npz",
          "v10_manifest": f"{RAW}/RUN-014__corrected_split_manifest.json", "category_map": f"{RAW}/Category_Map.csv"}
TAU = {"train": 0.70, "val": 0.10, "test": 0.20}
CRIT = {"max_class_dev_images": 3.0, "split_prop_tol_pp": 0.5, "max_class_share_one_group": 0.70,
        "large_group_size": 50, "max_class_share_large_groups": 0.50}


class DiagError(Exception):
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(repo, build):
    man = Path(repo, "MANIFEST.sha256")
    if not man.is_file():
        raise DiagError(f"{man} not found")
    listed = {}
    for line in man.read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1); listed[os.path.normpath(p.strip())] = h
    ins = {}
    for rel in INPUTS.values():
        p = Path(repo, rel)
        if not p.is_file() or listed.get(os.path.normpath(rel)) != sha256(p):
            raise DiagError(f"frozen input missing or not matching MANIFEST.sha256: {rel}")
        ins[rel] = sha256(p)
    st_p = Path(build, "BUILD_STATUS.json")
    if not st_p.is_file():
        raise DiagError(f"{st_p} not found")
    st = json.loads(st_p.read_text())
    if st.get("status") not in ("BUILT", "BUILT_DRY_RUN"):
        raise DiagError(f"build status {st.get('status')!r} is not a completed build")
    for name in ("corrected_split_manifest_v1.1.json", "per_class_counts_v1.1.csv"):
        if sha256(Path(build, name)) != (st.get("artifact_sha256") or {}).get(name):
            raise DiagError(f"{name} does not match BUILD_STATUS.json")
    return ins, st


def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int16)


def nn_to_train(split_of, keys, idx, ph, cl, frozen_pairs, chunk=256):
    """Per query image (val/test): max CLIP cosine and min pHash distance to any train image."""
    tr = np.array([idx[k] for k in keys if split_of[k] == "train"])
    tr_keys = [k for k in keys if split_of[k] == "train"]
    out, unlisted = {}, 0
    q = [k for k in keys if split_of[k] in ("val", "test")]
    for s in range(0, len(q), chunk):
        qk = q[s:s + chunk]; qi = np.array([idx[k] for k in qk])
        S = cl[qi] @ cl[tr].T
        D = popcount(np.bitwise_xor(ph[qi][:, None], ph[tr][None, :]))
        for r, k in enumerate(qk):
            out[k] = (float(S[r].max()), int(D[r].min()))
            band = np.nonzero((D[r] <= 9) | (S[r] >= 0.95))[0]
            unlisted += sum(tuple(sorted((k, tr_keys[j]))) not in frozen_pairs for j in band)
    return out, unlisted


def summ(v):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mean": round(float(v.mean()), 6), "median": round(float(np.median(v)), 6),
            "p90": round(float(np.percentile(v, 90)), 6), "max": round(float(v.max()), 6)}


def diagnose(a, staging):
    repo, build = Path(a.repo), Path(a.build_dir)
    ins, bst = verify(repo, build)
    man = json.loads(Path(build, "corrected_split_manifest_v1.1.json").read_text())
    asg = man["assignments"]; keys = sorted(asg)
    asg10 = json.loads(Path(repo, INPUTS["v10_manifest"]).read_text())["assignments"]
    if set(asg10) != set(keys):
        raise DiagError("build population differs from the v1.0 population")
    with open(Path(repo, INPUTS["category_map"]), newline="", encoding="utf-8") as f:
        cname = {int(r["Index"]) + 1: f'{r["Plant"]} | {r["Disease"]}' for r in csv.DictReader(f)}
    cls = {k: asg[k]["class"] for k in keys}
    N = len(keys); fails = []

    # ---------------- AC2
    n_s = Counter(asg[k]["split"] for k in keys)
    prop_pp = {s: round(100 * n_s[s] / N, 4) for s in TAU}
    prop_ok = all(abs(prop_pp[s] - 100 * TAU[s]) <= CRIT["split_prop_tol_pp"] for s in TAU)
    with open(Path(build, "per_class_counts_v1.1.csv"), newline="") as f:
        pcc = list(csv.DictReader(f))
    max_dev = max(float(r["max_abs_dev_images"]) for r in pcc)
    by_c = defaultdict(list)
    for k in keys:
        by_c[cls[k]].append(k)
    g_members = defaultdict(list)
    for k in keys:
        g_members[asg[k]["gid"]].append(k)
    rows = []
    for c in sorted(by_c):
        ks = by_c[c]; nc = len(ks)
        per_g = Counter(asg[k]["gid"] for k in ks)
        big = [k for k in ks if asg[k]["group_size"] >= CRIT["large_group_size"]]
        rows.append({"label": c, "class": cname.get(c, "?"), "n_images": nc,
                     "train": sum(asg[k]["split"] == "train" for k in ks), "val": sum(asg[k]["split"] == "val" for k in ks),
                     "test": sum(asg[k]["split"] == "test" for k in ks),
                     "max_share_one_group": round(max(per_g.values()) / nc, 6),
                     "share_in_groups_ge50": round(len(big) / nc, 6),
                     "large_group_partitions": ";".join(sorted({asg[k]["split"] for k in big}))})
    worst_one = max(r["max_share_one_group"] for r in rows)
    worst_big = max(r["share_in_groups_ge50"] for r in rows)
    ac2 = {"coverage_violations": man.get("coverage_violations"), "split_proportions_pp": prop_pp,
           "split_proportions_within_tol": prop_ok, "max_class_dev_images": max_dev,
           "max_class_share_one_group": worst_one}
    ac2["pass"] = (man.get("coverage_violations") == [] and prop_ok and max_dev <= CRIT["max_class_dev_images"]
                   and worst_one <= CRIT["max_class_share_one_group"])
    if not ac2["pass"]:
        fails.append("AC2 partition check failed")
    large = sorted(({"gid": g, "size": len(m), "split": asg[m[0]]["split"], "n_classes": len({cls[k] for k in m})}
                    for g, m in g_members.items() if len(m) >= CRIT["large_group_size"]), key=lambda d: -d["size"])
    ac4 = {"large_groups": large, "max_class_share_in_groups_ge50": worst_big,
           "classes_above_criterion": [r["class"] for r in rows if r["share_in_groups_ge50"] > CRIT["max_class_share_large_groups"]],
           "classes_with_any_share_ge50": sum(r["share_in_groups_ge50"] > 0 for r in rows)}
    ac4["pass"] = not ac4["classes_above_criterion"]
    if not ac4["pass"]:
        fails.append("AC4 composition criterion failed")

    # ---------------- nearest-neighbour similarity
    E = np.load(Path(repo, INPUTS["embeddings"]), allow_pickle=False)
    idx = {str(k): i for i, k in enumerate(E["keys"])}
    missing = [k for k in keys if k not in idx]
    if missing:
        raise DiagError(f"{len(missing)} image(s) lack frozen descriptors, e.g. {missing[:3]}")
    ph = E["phash_u64"].astype(np.uint64); cl = E["clip_emb"].astype(np.float32)
    ed = json.loads(Path(repo, INPUTS["edges"]).read_text())
    frozen = {tuple(sorted(e)) for e in ed["hard_edges"] + ed["soft_edges"]}
    nn = {}
    for tag, sp in (("v1.1", {k: asg[k]["split"] for k in keys}), ("v1.0", {k: asg10[k]["split"] for k in keys})):
        res, unl = nn_to_train(sp, keys, idx, ph, cl, frozen)
        block = {"band_pairs_not_in_frozen_edges": int(unl)}
        for part in ("test", "val"):
            q = [k for k in keys if sp[k] == part]
            block[part] = {"clip_max_cos": summ([res[k][0] for k in q]), "phash_min_d": summ([res[k][1] for k in q]),
                           "n_clip_ge_0.95": sum(res[k][0] >= 0.95 for k in q), "n_phash_le_9": sum(res[k][1] <= 9 for k in q)}
        block["per_class_test_clip_mean"] = {c: (round(float(np.mean([res[k][0] for k in by_c[c] if sp[k] == "test"])), 6)
                                                 if any(sp[k] == "test" for k in by_c[c]) else None) for c in sorted(by_c)}
        nn[tag] = block
    nnrows = [{"label": c, "class": cname.get(c, "?"), "test_clip_nn_mean_v1.1": nn["v1.1"]["per_class_test_clip_mean"][c],
               "test_clip_nn_mean_v1.0": nn["v1.0"]["per_class_test_clip_mean"][c],
               "share_in_groups_ge50": next(r["share_in_groups_ge50"] for r in rows if r["label"] == c)} for c in sorted(by_c)]
    for nm, data in (("per_class_composition.csv", rows), ("nn_similarity_per_class.csv", nnrows)):
        with open(staging / nm, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for b in nn.values():
        b.pop("per_class_test_clip_mean")
    rep = {"status": "PASS" if not fails else "FAIL", "fail_reasons": fails, "criteria": CRIT,
           "AC2": ac2, "AC4": ac4, "nearest_neighbour_to_train": nn,
           "build_manifest_sha256": bst["artifact_sha256"]["corrected_split_manifest_v1.1.json"],
           "build_version": man.get("version"), "grouping_policy": man.get("grouping_policy"),
           "provisional": man.get("provisional"), "input_sha256": ins, "tool_sha256": sha256(__file__),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "interpretation": ("NN similarity is descriptive. Lower test-train similarity under v1.1-C reflects candidate "
                              "containment and possible test-composition shift; it is not evidence of duplicate removal.")}
    (staging / "DIAGNOSTICS.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True); ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); stg = out.parent / f".{out.name}.incomplete"
    if out.exists() or stg.exists():
        print(f"REFUSED: {out} or its staging directory already exists", file=sys.stderr); return 2
    stg.mkdir(parents=False)
    try:
        rep = diagnose(a, stg)
    except (DiagError, KeyError, OSError, json.JSONDecodeError) as e:
        (stg / "DIAG_FAILED.txt").write_text(f"{e}\n")
        print(f"REFUSED: {e}", file=sys.stderr); return 2
    os.rename(stg, out)
    print(json.dumps({k: rep[k] for k in ("status", "fail_reasons")}, indent=1))
    return 0 if rep["status"] == "PASS" else 6


if __name__ == "__main__":
    sys.exit(main())
