"""Wilson score confidence interval for a binomial proportion.

Frozen use (Hypothesis_Register H1): 95% Wilson CI on
  - the hard cross-split near-duplicate leakage rate over the 1561 v7 test images;
  - the manual-audit detector precision (hard and soft).

No dependency on any ML framework: pure Python + math, so it is unit-testable
on any machine and identical on the HPC.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# z for common two-sided levels; 0.95 is the frozen level (alpha=0.05).
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    point: float
    lo: float
    hi: float
    level: float
    method: str = "wilson"


def wilson_interval(successes: int, n: int, level: float = 0.95) -> Proportion:
    """Two-sided Wilson score interval.

    Args:
        successes: number of positive events (e.g. leaked test images).
        n:          total trials (e.g. 1561 test images).
        level:      confidence level (0.95 frozen).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError("successes must be in [0, n]")
    z = _Z.get(round(level, 2))
    if z is None:
        raise ValueError(f"unsupported level {level}; use one of {sorted(_Z)}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    lo, hi = center - half, center + half
    return Proportion(successes, n, p, max(0.0, lo), min(1.0, hi), level)


if __name__ == "__main__":  # tiny self-check
    r = wilson_interval(23, 1561)
    print(f"leakage {r.point:.4%}  95% Wilson CI [{r.lo:.4%}, {r.hi:.4%}]")
