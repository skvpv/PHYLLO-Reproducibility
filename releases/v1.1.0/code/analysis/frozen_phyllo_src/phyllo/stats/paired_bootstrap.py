"""Paired image-level bootstrap over the fixed n=1557 cohort (EBSAP S1.6, H2/H3).

Frozen procedure (Hypothesis_Register H2.seed_aggregation, Blueprint S1.6):
  * >= 10,000 replicates.
  * For each replicate draw ONE index vector of 1557 images WITH replacement and
    REUSE it for every arm and every seed (paired).
  * Recompute each seed's DATASET-LEVEL mIoU_disease from the SUMMED sampled per-image
    confusion contributions (never average predictions, never average per-image mIoU).
  * Per seed: delta_leak = mIoU_disease(DEFAULT) - mIoU_disease(SANITIZED);
              delta_size = mIoU_disease(DEFAULT) - mIoU_disease(SIZE-CONTROL);
              delta_attr = delta_leak - delta_size.
  * Average the deltas ACROSS seeds within the replicate.
  * Report the bootstrap distribution of the seed-mean deltas (95% percentile CI),
    plus per-seed point deltas and between-seed SD.

Inputs are per-image confusion contributions with shape (N=1557, C=116) for
area_intersect and area_union, keyed by (arm, seed). Image ordering MUST be identical
across arms/seeds (enforced by the caller via a shared cohort index file).

This module is framework-independent and deterministic given the RNG seed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from phyllo.metrics.miou import miou_disease, miou_116

BOOTSTRAP_SEED = 20260917       # frozen PCG64 seed (also used for manifests/audit)
N_REPLICATES = 10000            # EBSAP floor; caller may raise, never lower


@dataclass
class ConfusionSet:
    """Per-image confusion contributions for one (arm, seed)."""
    area_intersect: np.ndarray   # (N, C) int64
    area_union: np.ndarray       # (N, C) int64

    def __post_init__(self):
        self.area_intersect = np.asarray(self.area_intersect, dtype=np.int64)
        self.area_union = np.asarray(self.area_union, dtype=np.int64)
        if self.area_intersect.shape != self.area_union.shape:
            raise ValueError("intersect/union shape mismatch")

    @property
    def n(self) -> int:
        return self.area_intersect.shape[0]


@dataclass
class BootstrapResult:
    name: str
    point: float
    ci_lo: float
    ci_hi: float
    excludes_zero: bool
    replicates: int
    p_value_two_sided: float
    per_seed_point: Dict[int, float] = field(default_factory=dict)
    between_seed_sd: Optional[float] = None


def _sampled_miou(cs: ConfusionSet, idx: np.ndarray, which: str) -> float:
    ai = cs.area_intersect[idx].sum(axis=0)
    au = cs.area_union[idx].sum(axis=0)
    return miou_disease(ai, au) if which == "disease" else miou_116(ai, au)


def _percentile_ci(dist: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    a = (1 - level) / 2 * 100
    return float(np.percentile(dist, a)), float(np.percentile(dist, 100 - a))


def _two_sided_p(dist: np.ndarray) -> float:
    """Smoothed achieved significance level for zero from a bootstrap distribution."""
    r = dist.size
    p_le = (int(np.sum(dist <= 0)) + 1) / (r + 1)
    p_ge = (int(np.sum(dist >= 0)) + 1) / (r + 1)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def h2_paired_bootstrap(default: Dict[int, ConfusionSet],
                        sanitized: Dict[int, ConfusionSet],
                        sizectrl: Dict[int, ConfusionSet],
                        n_replicates: int = N_REPLICATES,
                        seed: int = BOOTSTRAP_SEED,
                        level: float = 0.95
                        ) -> Dict[str, BootstrapResult]:
    """Bootstrap delta_leak, delta_size, delta_attr with seed averaging.

    Each of `default`/`sanitized`/`sizectrl` maps training-seed -> ConfusionSet, all on
    the identical cohort (same N, same image order). PRIMARY-2 (1 seed) simply passes a
    single-key mapping and no seed averaging occurs.
    """
    seeds = sorted(default)
    if sorted(sanitized) != seeds or sorted(sizectrl) != seeds:
        raise ValueError("arms must share the same training-seed set")
    n = default[seeds[0]].n
    for m in (default, sanitized, sizectrl):
        for s in seeds:
            if m[s].n != n:
                raise ValueError("all arms/seeds must share cohort size N")
    if n_replicates < N_REPLICATES:
        raise ValueError(f"n_replicates must be >= {N_REPLICATES} (EBSAP floor)")

    rng = np.random.Generator(np.random.PCG64(seed))
    dl = np.empty(n_replicates); ds = np.empty(n_replicates); da = np.empty(n_replicates)
    # per-seed accumulation of point deltas (full-cohort, no resampling) for reporting
    full = np.arange(n)
    per_seed_leak, per_seed_size, per_seed_attr = {}, {}, {}
    for s in seeds:
        d = _sampled_miou(default[s], full, "disease")
        san = _sampled_miou(sanitized[s], full, "disease")
        sc = _sampled_miou(sizectrl[s], full, "disease")
        per_seed_leak[s] = d - san
        per_seed_size[s] = d - sc
        per_seed_attr[s] = (d - san) - (d - sc)

    for r in range(n_replicates):
        idx = rng.integers(0, n, size=n)      # ONE shared index vector for all arms/seeds
        leaks, sizes, attrs = [], [], []
        for s in seeds:
            d = _sampled_miou(default[s], idx, "disease")
            san = _sampled_miou(sanitized[s], idx, "disease")
            sc = _sampled_miou(sizectrl[s], idx, "disease")
            leaks.append(d - san)
            sizes.append(d - sc)
            attrs.append((d - san) - (d - sc))
        dl[r] = float(np.mean(leaks))
        ds[r] = float(np.mean(sizes))
        da[r] = float(np.mean(attrs))

    out = {}
    for name, dist, per_seed in (("delta_leak", dl, per_seed_leak),
                                 ("delta_size", ds, per_seed_size),
                                 ("delta_attr", da, per_seed_attr)):
        lo, hi = _percentile_ci(dist, level)
        vals = np.array(list(per_seed.values()), dtype=float)
        out[name] = BootstrapResult(
            name=name, point=float(np.mean(vals)), ci_lo=lo, ci_hi=hi,
            excludes_zero=(lo > 0 or hi < 0), replicates=n_replicates,
            p_value_two_sided=_two_sided_p(dist),
            per_seed_point={int(k): float(v) for k, v in per_seed.items()},
            between_seed_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
    return out


def h3_paired_bootstrap(default: Dict[int, ConfusionSet],
                        n_replicates: int = N_REPLICATES,
                        seed: int = BOOTSTRAP_SEED,
                        level: float = 0.95) -> BootstrapResult:
    """Bootstrap delta_bg = mIoU_116 - mIoU_disease on the DEFAULT arm (H3)."""
    seeds = sorted(default)
    n = default[seeds[0]].n
    rng = np.random.Generator(np.random.PCG64(seed))
    dist = np.empty(n_replicates)
    full = np.arange(n)
    per_seed = {}
    for s in seeds:
        cs = default[s]
        ai = cs.area_intersect[full].sum(0); au = cs.area_union[full].sum(0)
        per_seed[s] = miou_116(ai, au) - miou_disease(ai, au)
    for r in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        vals = []
        for s in seeds:
            cs = default[s]
            ai = cs.area_intersect[idx].sum(0); au = cs.area_union[idx].sum(0)
            vals.append(miou_116(ai, au) - miou_disease(ai, au))
        dist[r] = float(np.mean(vals))
    lo, hi = _percentile_ci(dist, level)
    vals = np.array(list(per_seed.values()), dtype=float)
    return BootstrapResult("delta_bg", float(np.mean(vals)), lo, hi,
                           (lo > 0 or hi < 0), n_replicates, _two_sided_p(dist),
                           {int(k): float(v) for k, v in per_seed.items()},
                           float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
