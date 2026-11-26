"""Connected components of the HARD near-duplicate graph (frozen detector).

Implementation_Blueprint §4: duplicate clusters are the connected components of
the HARD edge graph (pHash-hard OR CLIP-hard). The soft graph is sensitivity-only
and never defines clusters.

Deterministic union-find with union-by-size and path compression. Node identity is
the image key (relative path like 'train/foo.jpg'); ordering of returned clusters is
deterministic (sorted by lexicographic minimum member) so downstream artifacts are
reproducible.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple


class UnionFind:
    def __init__(self, nodes: Iterable[str]):
        self.parent: Dict[str, str] = {n: n for n in nodes}
        self.size: Dict[str, int] = {n: 1 for n in self.parent}

    def add(self, n: str) -> None:
        if n not in self.parent:
            self.parent[n] = n
            self.size[n] = 1

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def connected_components(nodes: Iterable[str],
                         hard_edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """Return HARD-graph components as sorted member lists, deterministically ordered.

    Every node in `nodes` appears in exactly one component (singletons included).
    """
    uf = UnionFind(nodes)
    for a, b in hard_edges:
        uf.union(a, b)
    groups: Dict[str, List[str]] = {}
    for n in uf.parent:
        groups.setdefault(uf.find(n), []).append(n)
    comps = [sorted(m) for m in groups.values()]
    comps.sort(key=lambda m: m[0])  # order by lexicographic minimum member
    return comps
