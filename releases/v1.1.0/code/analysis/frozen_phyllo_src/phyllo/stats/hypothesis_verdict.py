"""Hypothesis verdict (C7A) — evaluated from frozen thresholds only.

Rules (OPS_P2 §7, I-12):
  * VERIFY locked_content_hash FIRST. Recompute over the canonical locked block
    (SCHEMAS §2.1: list of {hypothesis_id, locked}, JSON sort_keys, compact separators)
    and compare to the manifest value. MISMATCH -> raise; a verdict against a mutated
    hypothesis is meaningless.
  * NEVER edit any locked-block field, never recompute the hash to match an edit.
  * Set exactly one verdict per hypothesis: CONFIRMED | PARTIALLY_SUPPORTED | REJECTED.
  * A rejected hypothesis is a valid outcome — reported, never softened.

This module only DECIDES from supplied statistics; it does not run experiments and does
not write the register. The evidence-run outputs (bootstrap results, Wilson CI) are
passed in. Frozen acceptance rules are read here so the decision is auditable, but the
authoritative thresholds live in Hypothesis_Register.md and are re-read at G3.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

CONFIRMED = "CONFIRMED"
PARTIAL = "PARTIALLY_SUPPORTED"
REJECTED = "REJECTED"


def canonical_locked_block(register_path: str) -> Optional[bytes]:
    """SCHEMAS_v10_1 §2.1 — identical to validate_researchos.py."""
    if yaml is None:
        raise RuntimeError("PyYAML required to recompute the hypothesis locked hash")
    doc = yaml.safe_load(open(register_path, encoding="utf-8").read())
    if not isinstance(doc, dict):
        return None
    blocks = [{"hypothesis_id": h.get("hypothesis_id"), "locked": h["locked"]}
              for h in (doc.get("hypotheses") or [])
              if isinstance(h, dict) and "locked" in h]
    if not blocks:
        return None
    return json.dumps(blocks, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def recompute_locked_hash(register_path: str) -> str:
    b = canonical_locked_block(register_path)
    if b is None:
        raise RuntimeError("no locked blocks found")
    return "sha256:" + hashlib.sha256(b).hexdigest()


def verify_locked_hash(register_path: str, expected_hash: str) -> None:
    actual = recompute_locked_hash(register_path)
    if actual != expected_hash:
        raise RuntimeError(
            f"HYPOTHESIS HASH MISMATCH — HALT. recomputed {actual} != manifest "
            f"{expected_hash}. A verdict against a mutated hypothesis is invalid.")


@dataclass
class Verdict:
    hypothesis_id: str
    verdict: str
    verdict_evidence: str


def verdict_h1(leakage_rate: float, ci_lo: float, ci_hi: float,
               threshold: float = 0.01) -> Verdict:
    """H1: hard cross-split near-duplicate leakage rate >= 1% of the 1561 test images.

    The preregistered acceptance criterion is the point estimate >=1%. The Wilson
    interval is descriptive uncertainty and is not an added acceptance condition.
    """
    ev = f"leakage_rate={leakage_rate:.4%} (95% Wilson CI [{ci_lo:.4%}, {ci_hi:.4%}]), threshold>=1%"
    return Verdict("H1", CONFIRMED if leakage_rate >= threshold else REJECTED, ev)


def verdict_h2(delta_leak_point: float, dl_lo: float, dl_hi: float,
               delta_attr_point: float, da_lo: float, da_hi: float,
               margin: float = 0.005) -> Verdict:
    """H2: effects are on a 0..1 scale, so 0.5 mIoU points is 0.005.

    SUPPORTED only if delta_leak >= 0.005 with 95% CI excluding 0
    AND delta_attr > 0 with 95% CI excluding 0.

    CONFIRMED if both conditions hold;
    PARTIALLY_SUPPORTED if delta_leak CI excludes 0 and point>0 but either the 0.5-pt
    margin is unmet or the attribution (delta_attr) CI does not exclude 0
    (leakage effect present but not attributable/not large enough);
    REJECTED otherwise.
    """
    dl_excl = (dl_lo > 0 or dl_hi < 0)
    da_excl = (da_lo > 0 or da_hi < 0)
    ev = (f"delta_leak={delta_leak_point:.4f} (95% CI [{dl_lo:.4f},{dl_hi:.4f}], "
          f"excl0={dl_excl}); delta_attr={delta_attr_point:.4f} "
          f"(95% CI [{da_lo:.4f},{da_hi:.4f}], excl0={da_excl}); margin>={margin}")
    if delta_leak_point >= margin and dl_excl and delta_attr_point > 0 and da_excl:
        return Verdict("H2", CONFIRMED, ev)
    if delta_leak_point > 0 and dl_excl:
        return Verdict("H2", PARTIAL, ev)
    return Verdict("H2", REJECTED, ev)


def verdict_h3(all_baselines_delta_bg_positive_excl0: bool, detail: str) -> Verdict:
    """H3 (frozen): delta_bg > 0 with 95% CI excluding 0 for ALL evaluated baselines.

    CONFIRMED if the condition holds for every evaluated baseline;
    PARTIALLY_SUPPORTED if it holds for some but not all;
    REJECTED if it holds for none.
    """
    # `detail` encodes the per-baseline booleans; caller derives the aggregate flags.
    if all_baselines_delta_bg_positive_excl0:
        return Verdict("H3", CONFIRMED, detail)
    if "any_true=True" in detail:
        return Verdict("H3", PARTIAL, detail)
    return Verdict("H3", REJECTED, detail)
