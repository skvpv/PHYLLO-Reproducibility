"""Sanitized (arm B) and size-control (arm C) removal manifests for H2.

DEFAULT-QC (arm A) = released split/config after the uniform 12-mask QC exclusions
(8 defective train masks removed from train; the fixed test cohort is 1561 - 4 = 1557).

SANITIZED (arm B): from the DEFAULT-QC train+val pools, remove EVERY image that belongs
to a HARD duplicate cluster which contains at least one fixed-cohort (1557) test image.

SIZE-CONTROL (arm C): remove the SAME number of images from each original split as B,
drawn from images OUTSIDE test-intersecting hard clusters, using ONE deterministic
NumPy PCG64 seed-20260917 manifest that minimizes L1 distance from B's per-class removal
counts within each split, with lexicographic filename order for ties; reused across all
training seeds.

All frozen per Hypothesis_Register H2.design and Blueprint §8. Deterministic and
framework-independent; the caller supplies the cluster membership (from the frozen
detector) and per-image class labels.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set
import numpy as np

SEED = 20260917
SPLITS = ("train", "val")   # removals only touch train and val pools


@dataclass
class ArmManifests:
    sanitized_remove: Dict[str, List[str]]     # split -> removed image keys
    sizectrl_remove: Dict[str, List[str]]      # split -> removed image keys
    sanitized_counts: Dict[str, int]
    sizectrl_counts: Dict[str, int]
    per_class_removed_sanitized: Dict[str, Dict[int, int]]
    per_class_removed_sizectrl: Dict[str, Dict[int, int]]


def build_sanitized(clusters: Sequence[Sequence[str]],
                    fixed_cohort_test: Set[str],
                    default_qc_pool: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Return {split: [removed keys]} for arm B.

    clusters: HARD connected components (lists of image keys 'split/name').
    fixed_cohort_test: the 1557 test-image keys.
    default_qc_pool: {'train': set(keys), 'val': set(keys)} after QC exclusions.
    """
    remove = {s: set() for s in SPLITS}
    for comp in clusters:
        if any(k in fixed_cohort_test for k in comp):     # cluster intersects cohort
            for k in comp:
                for s in SPLITS:
                    if k in default_qc_pool[s]:
                        remove[s].add(k)
    return {s: sorted(remove[s]) for s in SPLITS}


def build_size_control(sanitized_remove: Dict[str, List[str]],
                       default_qc_pool: Dict[str, Set[str]],
                       test_intersecting_keys: Set[str],
                       image_class: Dict[str, int],
                       seed: int = SEED) -> Dict[str, List[str]]:
    """Return {split: [removed keys]} for arm C.

    For each split independently: choose exactly len(sanitized_remove[split]) images from
    the eligible pool (default-QC pool minus any key in a test-intersecting hard cluster),
    minimizing L1 distance from arm-B's per-class removal counts, deterministic PCG64,
    lexicographic tie-break.

    image_class maps image key -> its (single) disease/class id used for matching the
    per-class removal profile. For masks that are multi-class, the caller passes a stable
    representative class id (e.g. the rarest present class) so the profile is well-defined.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    out: Dict[str, List[str]] = {}
    for s in SPLITS:
        target_n = len(sanitized_remove[s])
        target_profile = Counter(image_class[k] for k in sanitized_remove[s])
        eligible = sorted(k for k in default_qc_pool[s]
                          if k not in test_intersecting_keys)
        chosen: List[str] = []
        chosen_profile: Counter = Counter()
        # Greedy class-matched selection: repeatedly pick the class most under-represented
        # relative to target, then take its lexicographically-first eligible unused image;
        # RNG only breaks exact ties among equally-eligible classes for auditable
        # determinism. Falls back to remaining eligible images if a class is exhausted.
        by_class: Dict[int, List[str]] = {}
        for k in eligible:
            by_class.setdefault(image_class[k], []).append(k)
        for c in by_class:
            by_class[c].sort()
        used: Set[str] = set()
        while len(chosen) < target_n:
            deficits = {c: target_profile.get(c, 0) - chosen_profile.get(c, 0)
                        for c in set(target_profile) | set(by_class)}
            # candidate classes with remaining eligible images
            cand = [c for c in deficits if by_class.get(c) and
                    any(k not in used for k in by_class[c])]
            if not cand:
                # profile exhausted: fill from all remaining eligible, lexicographic
                remaining = [k for k in eligible if k not in used]
                if not remaining:
                    break
                chosen.append(remaining[0]); used.add(remaining[0])
                chosen_profile[image_class[remaining[0]]] += 1
                continue
            max_def = max(deficits[c] for c in cand)
            top = sorted(c for c in cand if deficits[c] == max_def)
            # deterministic RNG tie-break among equally-deficit classes
            c = top[0] if len(top) == 1 else top[int(rng.integers(0, len(top)))]
            pick = next(k for k in by_class[c] if k not in used)
            chosen.append(pick); used.add(pick); chosen_profile[c] += 1
        out[s] = sorted(chosen)
    return out


def build_arm_manifests(clusters, fixed_cohort_test, default_qc_pool,
                        test_intersecting_keys, image_class) -> ArmManifests:
    san = build_sanitized(clusters, fixed_cohort_test, default_qc_pool)
    sc = build_size_control(san, default_qc_pool, test_intersecting_keys, image_class)

    def prof(rem):
        return {s: dict(Counter(image_class[k] for k in rem[s])) for s in SPLITS}

    return ArmManifests(
        sanitized_remove=san, sizectrl_remove=sc,
        sanitized_counts={s: len(san[s]) for s in SPLITS},
        sizectrl_counts={s: len(sc[s]) for s in SPLITS},
        per_class_removed_sanitized=prof(san),
        per_class_removed_sizectrl=prof(sc),
    )
