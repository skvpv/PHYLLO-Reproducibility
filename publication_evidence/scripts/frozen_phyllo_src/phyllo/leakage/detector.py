"""Frozen near-duplicate detector: exact blocked all-pairs, edges, clusters, census.

Blueprint §4/§5 (FROZEN):
  * Hard edge = pHash Hamming <= 3 OR CLIP cosine >= 0.98.
  * Soft edge = (pHash Hamming in 4..9 OR CLIP cosine in 0.95..<0.98) AND not hard.
  * Clusters = connected components of the HARD graph over all 7,774 images.
  * Exact blocked all-pairs comparison (NO ANN).
  * Sensitivity grid: pHash {<=3,<=5,<=7,<=9} x CLIP {0.95,0.96,0.97,0.98,0.99}.
H1 leakage rate = % of the 1561 v7 test images with >= 1 HARD cross-split (test<->train)
near-duplicate.

Pure NumPy over precomputed uint64 pHash and float32 L2-normalized CLIP embeddings, so
it is deterministic and unit-testable without any GPU or model. Popcount via a 16-bit
lookup table.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from phyllo.leakage.components import connected_components

# frozen thresholds
PHASH_HARD = 3
PHASH_SOFT_HI = 9
CLIP_HARD = 0.98
CLIP_SOFT_LO = 0.95
# sensitivity grid
GRID_PHASH = (3, 5, 7, 9)
GRID_CLIP = (0.95, 0.96, 0.97, 0.98, 0.99)

_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    return (_POP16[(x & 0xFFFF).astype(np.uint32)]
            + _POP16[((x >> np.uint64(16)) & 0xFFFF).astype(np.uint32)]
            + _POP16[((x >> np.uint64(32)) & 0xFFFF).astype(np.uint32)]
            + _POP16[((x >> np.uint64(48)) & 0xFFFF).astype(np.uint32)]).astype(np.int16)


@dataclass
class DetectorResult:
    hard_edges: List[Tuple[str, str]]
    soft_edges: List[Tuple[str, str]]
    clusters: List[List[str]]
    leaked_test_keys: List[str]     # test images with >=1 HARD train near-dup
    n_test: int
    leakage_rate: float


def _iter_blocks(n: int, block: int):
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            yield i0, i1, j0, j1


def detect(keys: Sequence[str], phash_u64: np.ndarray, clip_emb: np.ndarray,
           splits: Sequence[str],
           phash_hard: int = PHASH_HARD, clip_hard: float = CLIP_HARD,
           phash_soft_hi: int = PHASH_SOFT_HI, clip_soft_lo: float = CLIP_SOFT_LO,
           block: int = 1024) -> DetectorResult:
    """Exact blocked all-pairs detection.

    keys[i] identifies image i ('split/name'); splits[i] in {'train','val','test'};
    phash_u64[i] uint64; clip_emb[i] float32 L2-normalized (dot == cosine).
    """
    n = len(keys)
    phash_u64 = phash_u64.astype(np.uint64)
    clip_emb = np.ascontiguousarray(clip_emb, dtype=np.float32)
    is_test = np.array([s == "test" for s in splits])
    is_train = np.array([s == "train" for s in splits])

    hard_edges: List[Tuple[str, str]] = []
    soft_edges: List[Tuple[str, str]] = []
    leaked = np.zeros(n, dtype=bool)

    for i0, i1, j0, j1 in _iter_blocks(n, block):
        a = phash_u64[i0:i1][:, None]
        b = phash_u64[j0:j1][None, :]
        ham = _popcount_u64(np.bitwise_xor(a, b))              # (bi, bj)
        cos = clip_emb[i0:i1] @ clip_emb[j0:j1].T              # (bi, bj)
        hard = (ham <= phash_hard) | (cos >= clip_hard)
        soft = ((ham >= phash_hard + 1) & (ham <= phash_soft_hi)) | \
               ((cos >= clip_soft_lo) & (cos < clip_hard))
        soft = soft & ~hard
        bi = np.arange(i0, i1)[:, None]
        bj = np.arange(j0, j1)[None, :]
        upper = bi < bj                                        # unique unordered pairs
        for mask, bucket in ((hard & upper, hard_edges), (soft & upper, soft_edges)):
            ii, jj = np.nonzero(mask)
            for x, y in zip(ii + i0, jj + j0):
                bucket.append((keys[x], keys[y]))
        # cross-split census: test<->train hard
        hi = hard.copy()
        # rows that are test and cols that are train, OR rows train & cols test
        rt = is_test[i0:i1][:, None] & is_train[j0:j1][None, :]
        tr = is_train[i0:i1][:, None] & is_test[j0:j1][None, :]
        cross = hi & (rt | tr) & (bi != bj)
        # mark leaked test images (either side)
        ri, cj = np.nonzero(cross)
        for x, y in zip(ri + i0, cj + j0):
            if is_test[x]:
                leaked[x] = True
            if is_test[y]:
                leaked[y] = True

    clusters = connected_components(keys, hard_edges)
    n_test = int(is_test.sum())
    leaked_keys = [keys[i] for i in range(n) if leaked[i]]
    rate = len(leaked_keys) / n_test if n_test else float("nan")
    return DetectorResult(hard_edges, soft_edges, clusters, sorted(leaked_keys),
                          n_test, rate)


def sensitivity_census(keys, phash_u64, clip_emb, splits,
                       grid_phash=GRID_PHASH, grid_clip=GRID_CLIP) -> Dict[str, float]:
    """Leakage-rate surface across the frozen sensitivity grid (robustness only)."""
    out = {}
    for ph in grid_phash:
        for cl in grid_clip:
            r = detect(keys, phash_u64, clip_emb, splits,
                       phash_hard=ph, clip_hard=cl)
            out[f"pHash<={ph}|CLIP>={cl:.2f}"] = r.leakage_rate
    return out
