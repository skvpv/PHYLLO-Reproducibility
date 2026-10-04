"""Deterministic component-confined corrected split (70/10/20).

Performance-corrected implementation (RUN-014).

Scientific behaviour is UNCHANGED. The three phases (greedy weighted-L1 assignment,
preregistered coverage repair, and best-improvement true pairwise swaps), their
deterministic ordering, their tie-breaking, and their termination at the same local
optimum are identical to the reference implementation preserved for testing at
``tests/reference/corrected_benchmark_reference.py``.

Only the *cost per candidate evaluation* changed. The reference recomputed the full
objective (and full coverage scan, plus a full assignment copy) for every candidate,
making the pairwise-swap sweep ~cubic in the number of groups. This version maintains
incremental sufficient statistics so that:

  * a full objective value is obtained in O(#classes) instead of O(#class-incidences),
    and is computed in the *same summation order* as the reference objective, so it is
    bit-identical for any given assignment (used for greedy/coverage decisions, base
    refresh, and every recorded trace value);
  * a pairwise-swap objective *delta* and its coverage feasibility are obtained in
    O(#classes in the two swapped groups) with no assignment copy and no full scan.

See PERFORMANCE_CORRECTION.md for the complexity analysis and the equivalence argument.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

SPLITS = ("train", "val", "test")
TARGET = {"train": .70, "val": .10, "test": .20}

# Index view of the same frozen constants (identical float objects/values as TARGET).
_SPLIT_IDX = {s: i for i, s in enumerate(SPLITS)}
_TARGET_LIST = [TARGET[s] for s in SPLITS]


@dataclass
class Group:
    gid: int
    members: List[str]
    classes: Set[int]
    size: int = field(init=False)
    min_name: str = field(init=False)

    def __post_init__(self):
        self.members = sorted(self.members)
        if not self.members:
            raise ValueError("empty group")
        self.size = len(self.members)
        self.min_name = self.members[0]


def _frequencies(groups: Sequence[Group]):
    frequencies = {}
    for group in groups:
        for class_id in group.classes:
            frequencies[class_id] = frequencies.get(class_id, 0) + 1
    return frequencies


def _sort_key(group: Group, frequencies):
    rarest = min((frequencies[c] for c in group.classes), default=10**9)
    return (-group.size, rarest, group.min_name)


# --------------------------------------------------------------------------------------
# Reference-identical batch helpers. These are retained verbatim from the original module
# so external callers/tests keep working and so the fast path can be cross-checked. The
# optimized assignment routine below does NOT call them on its hot path.
# --------------------------------------------------------------------------------------
def _objective(groups, assignment):
    total = sum(g.size for g in groups)
    by_split = {s: 0 for s in SPLITS}
    class_total = {}
    class_split = {}
    for group in groups:
        split = assignment.get(group.gid)
        if split is not None:
            by_split[split] += group.size
        for class_id in group.classes:
            class_total[class_id] = class_total.get(class_id, 0) + group.size
            class_split.setdefault(class_id, {s: 0 for s in SPLITS})
            if split is not None:
                class_split[class_id][split] += group.size
    score = sum(abs(by_split[s] / total - TARGET[s]) for s in SPLITS) if total else 0.0
    for class_id, class_n in class_total.items():
        score += sum(abs(class_split[class_id][s] / class_n - TARGET[s])
                     for s in SPLITS) / max(1, len(class_total))
    return score


def _coverage(groups, assignment):
    group_count, split_count = {}, {}
    for group in groups:
        for class_id in group.classes:
            group_count[class_id] = group_count.get(class_id, 0) + 1
            split_count.setdefault(class_id, {s: 0 for s in SPLITS})
            if group.gid in assignment:
                split_count[class_id][assignment[group.gid]] += 1
    return group_count, split_count


def _constraints_hold(groups, assignment):
    total, counts = _coverage(groups, assignment)
    return all(total[c] < 3 or all(counts[c][s] > 0 for s in SPLITS) for c in total)


# --------------------------------------------------------------------------------------
# Incremental sufficient statistics.
# --------------------------------------------------------------------------------------
class _SplitState:
    """Maintains, under an evolving assignment, exactly the sufficient statistics the
    reference objective and coverage functions derive from scratch on every call:

      by_split[s]            image-count assigned to split s
      cls_img[c][s]          image-count of groups containing class c assigned to s
      cls_grp[c][s]          group-count of groups containing class c assigned to s
      cls_img_total[c]       constant: total image-count over groups containing c
      cls_grp_total[c]       constant: total group-count over groups containing c

    ``class_order`` reproduces the reference's ``class_total`` insertion order (groups in
    their ORIGINAL order, classes in set-iteration order) so that ``objective`` sums the
    per-class terms in the identical order and is therefore bit-identical to
    ``_objective`` for any assignment. ``n_violations`` counts classes that appear in
    >=3 groups but are not yet present in all three splits (0 iff coverage holds).
    """

    __slots__ = ("total", "n_classes", "K", "class_order", "cls_img_total",
                 "cls_grp_total", "by_split", "cls_img", "cls_grp", "assign", "n_violations")

    def __init__(self, groups: Sequence[Group]):
        self.total = sum(g.size for g in groups)
        class_order: List[int] = []
        cls_img_total: Dict[int, int] = {}
        cls_grp_total: Dict[int, int] = {}
        for group in groups:                      # ORIGINAL order (matches _objective)
            for class_id in group.classes:        # set-iteration order (matches _objective)
                if class_id not in cls_img_total:
                    class_order.append(class_id)
                    cls_img_total[class_id] = 0
                    cls_grp_total[class_id] = 0
                cls_img_total[class_id] += group.size
                cls_grp_total[class_id] += 1
        self.class_order = class_order
        self.n_classes = len(class_order)
        self.K = max(1, self.n_classes)
        self.cls_img_total = cls_img_total
        self.cls_grp_total = cls_grp_total
        self.by_split = [0, 0, 0]
        self.cls_img = {c: [0, 0, 0] for c in class_order}
        self.cls_grp = {c: [0, 0, 0] for c in class_order}
        self.assign: Dict[int, str] = {}
        self.n_violations = 0                     # nothing assigned yet -> no split present,
        # but a class only "violates" once it is under an assignment we score; during the
        # greedy build we never query violations, and after the greedy every group is
        # assigned, at which point n_violations is rebuilt via recount().

    # -- objective (bit-identical to _objective for the current assignment) -------------
    def objective(self) -> float:
        total = self.total
        if not total:
            score = 0.0
        else:
            bs = self.by_split
            score = sum(abs(bs[i] / total - _TARGET_LIST[i]) for i in range(3))
        K = self.K
        cls_img = self.cls_img
        cls_img_total = self.cls_img_total
        for c in self.class_order:
            row = cls_img[c]
            cn = cls_img_total[c]
            score += sum(abs(row[i] / cn - _TARGET_LIST[i]) for i in range(3)) / K
        return score

    def _class_violates(self, c) -> bool:
        if self.cls_grp_total[c] < 3:
            return False
        g = self.cls_grp[c]
        return not (g[0] > 0 and g[1] > 0 and g[2] > 0)

    def recount_violations(self) -> None:
        self.n_violations = sum(1 for c in self.class_order if self._class_violates(c))

    def constraints_hold(self) -> bool:
        return self.n_violations == 0

    # -- mutation -----------------------------------------------------------------------
    def assign_group(self, group: Group, split: str) -> None:
        """Assign a currently-unassigned group to ``split`` (greedy build)."""
        i = _SPLIT_IDX[split]
        self.by_split[i] += group.size
        for c in group.classes:
            self.cls_img[c][i] += group.size
            self.cls_grp[c][i] += 1
        self.assign[group.gid] = split

    def move_group(self, group: Group, src: str, dst: str) -> None:
        """Relocate an assigned group from ``src`` to ``dst`` (coverage repair)."""
        si, di = _SPLIT_IDX[src], _SPLIT_IDX[dst]
        size = group.size
        self.by_split[si] -= size
        self.by_split[di] += size
        for c in group.classes:
            row = self.cls_img[c]; grp = self.cls_grp[c]
            row[si] -= size; row[di] += size
            grp[si] -= 1; grp[di] += 1
        self.assign[group.gid] = dst

    def apply_swap(self, left: Group, right: Group) -> None:
        """Swap the splits of two assigned groups (pairwise-swap phase)."""
        a = _SPLIT_IDX[self.assign[left.gid]]
        b = _SPLIT_IDX[self.assign[right.gid]]
        self.by_split[a] += right.size - left.size
        self.by_split[b] += left.size - right.size
        for c in left.classes:
            self.cls_img[c][a] -= left.size; self.cls_img[c][b] += left.size
            self.cls_grp[c][a] -= 1; self.cls_grp[c][b] += 1
        for c in right.classes:
            self.cls_img[c][b] -= right.size; self.cls_img[c][a] += right.size
            self.cls_grp[c][b] -= 1; self.cls_grp[c][a] += 1
        self.assign[left.gid], self.assign[right.gid] = \
            self.assign[right.gid], self.assign[left.gid]

    # -- pairwise-swap candidate evaluation (no copy, no full scan) ---------------------
    def swap_delta_feasible(self, left: Group, right: Group):
        """Return ``(delta, feasible)`` for swapping ``left`` and ``right`` without
        mutating any state. ``delta`` is the change in the objective; ``feasible`` is the
        coverage predicate ``_constraints_hold`` would return for the swapped assignment.

        Only the two occupied splits (a, b) and the classes contained in the two groups
        can change, so both are computed from those terms alone.
        """
        a = _SPLIT_IDX[self.assign[left.gid]]
        b = _SPLIT_IDX[self.assign[right.gid]]
        total = self.total
        Ta, Tb = _TARGET_LIST[a], _TARGET_LIST[b]

        # split-balance term (only a and b change)
        ba, bb = self.by_split[a], self.by_split[b]
        nba = ba - left.size + right.size
        nbb = bb + left.size - right.size
        old_split = abs(ba / total - Ta) + abs(bb / total - Tb)
        new_split = abs(nba / total - Ta) + abs(nbb / total - Tb)
        delta = new_split - old_split

        # per-class terms + coverage feasibility (affected classes only)
        left_classes = left.classes
        right_classes = right.classes
        affected = left_classes | right_classes
        K = self.K
        viol_change = 0
        for c in affected:
            ia = self.cls_img[c][a]; ib = self.cls_img[c][b]
            ga = self.cls_grp[c][a]; gb = self.cls_grp[c][b]
            nia, nib, nga, ngb = ia, ib, ga, gb
            if c in left_classes:      # left leaves a, enters b
                nia -= left.size; nib += left.size; nga -= 1; ngb += 1
            if c in right_classes:     # right leaves b, enters a
                nib -= right.size; nia += right.size; ngb -= 1; nga += 1
            cn = self.cls_img_total[c]
            old_ab = abs(ia / cn - Ta) + abs(ib / cn - Tb)
            new_ab = abs(nia / cn - Ta) + abs(nib / cn - Tb)
            delta += (new_ab - old_ab) / K
            if self.cls_grp_total[c] >= 3:
                third = 3 - a - b                              # the split index not in {a,b}
                gt = self.cls_grp[c][third]
                before = not (ga > 0 and gb > 0 and gt > 0)
                after = not (nga > 0 and ngb > 0 and gt > 0)
                viol_change += int(after) - int(before)
        feasible = (self.n_violations + viol_change) == 0
        return delta, feasible


def assign_corrected_split(
    groups: List[Group],
    diagnostics: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Frozen greedy assignment, coverage repair, and deterministic pair swaps.

    ``diagnostics``, if given, is called with small progress dicts (phase, counters,
    current objective, elapsed seconds). It never influences the result.
    """
    if len({member for group in groups for member in group.members}) != sum(g.size for g in groups):
        raise ValueError("an image occurs in more than one group")

    def _emit(**kw):
        if diagnostics is not None:
            diagnostics(kw)

    t0 = time.perf_counter()
    frequencies = _frequencies(groups)
    ordered = sorted(groups, key=lambda g: _sort_key(g, frequencies))
    group_map = {g.gid: g for g in groups}
    state = _SplitState(groups)
    trace: List[dict] = []

    # Greedy weighted-L1 assignment; SPLITS iteration provides train->val->test ties.
    # Each candidate objective is the full (bit-identical) objective of the partial
    # assignment with this group tentatively placed -- exactly the reference decision.
    for k, group in enumerate(ordered):
        best = None  # (objective, split_index)
        for split in SPLITS:
            state.assign_group(group, split)
            obj = state.objective()
            # undo the tentative placement
            i = _SPLIT_IDX[split]
            state.by_split[i] -= group.size
            for c in group.classes:
                state.cls_img[c][i] -= group.size
                state.cls_grp[c][i] -= 1
            del state.assign[group.gid]
            cand = (obj, _SPLIT_IDX[split])
            if best is None or cand < best:
                best = cand
        chosen = SPLITS[best[1]]
        state.assign_group(group, chosen)
        trace.append({"stage": "greedy", "objective": state.objective(),
                      "gid": group.gid, "to": chosen})
        if diagnostics is not None and (k + 1) % 500 == 0:
            _emit(phase="greedy", assigned=k + 1, n=len(ordered),
                  objective=trace[-1]["objective"], elapsed=time.perf_counter() - t0)
    state.recount_violations()
    _emit(phase="greedy_done", objective=state.objective(),
          violations=state.n_violations, elapsed=time.perf_counter() - t0)

    # Coverage repair: satisfy the preregistered all-split class condition whenever >=3
    # independent groups make it feasible. A move may not break coverage already achieved
    # by any other eligible class contained in the same group. Full (bit-identical)
    # objective per candidate -> identical decision to the reference.
    for _ in range(len(groups) * 3):
        missing = None
        for c in sorted(state.cls_grp_total):
            if state.cls_grp_total[c] >= 3:
                grp = state.cls_grp[c]
                for s in SPLITS:
                    if grp[_SPLIT_IDX[s]] == 0:
                        missing = (c, s)
                        break
            if missing is not None:
                break
        if missing is None:
            break
        class_id, destination = missing
        best = None  # (objective, min_name, gid, group, source)
        for group in ordered:
            source = state.assign[group.gid]
            if class_id not in group.classes or source == destination:
                continue
            # do not drop another eligible class below coverage in the source split
            blocked = False
            for c in group.classes:
                if state.cls_grp_total[c] >= 3 and state.cls_grp[c][_SPLIT_IDX[source]] <= 1:
                    blocked = True
                    break
            if blocked:
                continue
            state.move_group(group, source, destination)
            obj = state.objective()
            state.move_group(group, destination, source)  # undo
            cand = (obj, group.min_name, group.gid, group, source)
            if best is None or cand[:3] < best[:3]:
                best = cand
        if best is None:
            break
        _, _, gid, group, source = best
        state.move_group(group, source, destination)
        trace.append({"stage": "coverage_repair", "objective": state.objective(),
                      "gid": gid, "to": destination, "class": class_id})
    state.recount_violations()
    _emit(phase="coverage_done", objective=state.objective(),
          violations=state.n_violations, elapsed=time.perf_counter() - t0)

    # True pairwise swaps, best-improvement, coverage-preserving. Same neighbourhood,
    # same 1e-12 improvement threshold, same (objective, names, gids) tie-break, same
    # termination as the reference.
    #
    # The reference recomputes a full objective and a full coverage scan (plus an
    # assignment copy) for every one of the O(N^2) candidate pairs. Here each candidate
    # is screened in O(#classes in the pair) with an incremental objective *delta* and an
    # incremental coverage-feasibility test. The reference selects the feasible improving
    # swap minimizing (full_objective(trial), name_left, name_right, gid_left, gid_right).
    # Because the reference's primary key is a full-resummed float, ties in that float can
    # be resolved by its floating-point value *before* the (name, gid) key engages. To
    # reproduce that decision exactly we cannot rely on ``base + delta`` (which differs
    # from a fresh full resum by rounding). So selection is done in two passes per sweep:
    #
    #   pass 1  scan all pairs, find the minimum feasible improving delta ``dmin``;
    #   pass 2  scan all pairs again; for the few candidates within a tiny window of
    #           ``dmin`` recompute the BIT-IDENTICAL full objective (same summation order
    #           as the reference ``_objective``) and select by
    #           (full_objective, name_left, name_right, gid_left, gid_right).
    #
    # Over-inclusion in the window is harmless: the exact-objective key still selects the
    # global minimum. The reference's winner has the minimum full objective, hence a delta
    # within a few ulps of ``dmin``, hence is always inside the window -- so the selection
    # is provably identical. See PERFORMANCE_CORRECTION.md.
    TIE_WINDOW = 1e-9
    sweep = 0
    while True:
        base = state.objective()
        gids = sorted(state.assign)
        n = len(gids)

        # pass 1: minimum feasible improving delta (O(1) memory)
        dmin = None
        for i in range(n):
            left = group_map[gids[i]]
            left_split = state.assign[gids[i]]
            for j in range(i + 1, n):
                if state.assign[gids[j]] == left_split:
                    continue
                delta, feasible = state.swap_delta_feasible(left, group_map[gids[j]])
                if feasible and delta < -1e-12 and (dmin is None or delta < dmin):
                    dmin = delta
        if dmin is None:
            break

        # pass 2: exact, reference-identical selection among near-best candidates
        window = dmin + TIE_WINDOW
        best = None  # (full_objective, name_left, name_right, gid_left, gid_right, l, r)
        for i in range(n):
            left = group_map[gids[i]]
            left_split = state.assign[gids[i]]
            for j in range(i + 1, n):
                if state.assign[gids[j]] == left_split:
                    continue
                right = group_map[gids[j]]
                delta, feasible = state.swap_delta_feasible(left, right)
                if not (feasible and delta < -1e-12 and delta <= window):
                    continue
                state.apply_swap(left, right)
                obj = state.objective()            # bit-identical to _objective(trial)
                state.apply_swap(left, right)       # undo (apply_swap is its own inverse)
                cand = (obj, left.min_name, right.min_name, gids[i], gids[j])
                if best is None or cand < best[:5]:
                    best = cand + (left, right)

        left, right = best[5], best[6]
        state.apply_swap(left, right)
        trace.append({"stage": "pair_swap", "objective": best[0],
                      "gid_a": best[3], "gid_b": best[4]})
        sweep += 1
        if diagnostics is not None:
            _emit(phase="pair_swap", sweep=sweep, gid_a=best[3], gid_b=best[4],
                  objective=best[0], elapsed=time.perf_counter() - t0)
    state.recount_violations()

    # Final outputs (identical shape and values to the reference).
    infeasible = []
    for c in sorted(state.cls_grp_total):
        if state.cls_grp_total[c] >= 3:
            grp = state.cls_grp[c]
            if not (grp[0] and grp[1] and grp[2]):
                infeasible.append({"class": c, "n_groups": state.cls_grp_total[c],
                                   "splits_present": [s for s in SPLITS if grp[_SPLIT_IDX[s]]]})
    members = {member: {"gid": group.gid, "split": state.assign[group.gid],
                        "group_size": group.size, "classes": sorted(group.classes)}
               for group in groups for member in group.members}
    split_counts = {s: state.by_split[_SPLIT_IDX[s]] for s in SPLITS}
    final_obj = state.objective()
    _emit(phase="done", objective=final_obj, sweeps=sweep,
          elapsed=time.perf_counter() - t0)
    return {"assignments": members, "split_image_counts": split_counts,
            "objective_trace": trace, "final_objective": final_obj,
            "infeasible_classes": infeasible, "constraints_satisfied": not infeasible,
            "n_groups": len(groups), "total_images": sum(g.size for g in groups)}
