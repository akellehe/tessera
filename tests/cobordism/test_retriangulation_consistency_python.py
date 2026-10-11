"""Re-triangulation consistency of the primal/dual machinery (#268).

Acceptance (iv) of the Regge-mediation milestone (rooted in #246/#247): the dual
volumes and the dual Regge action must be **geometric**, not artifacts of how the
complex happens to be labelled or subdivided. This pins down exactly which
re-triangulations leave the realizability residual r_U and the dual Lorentzian
Regge action |S_Regge| invariant.

Findings (all on the regular-tetrahedron surface, |S_Regge| = π·√3, area = √3):

  * Vertex **relabeling** — a re-triangulation that changes only the vertex ids,
    preserving both the primal geometry and the dual — leaves |S_Regge|, the
    residual r_U, and the total dual volume **exactly** invariant. (iv) holds.

  * The **total dual volume** (Σ over the hinges of their circumcentric dual
    cells) equals the surface area under relabeling AND under a metric-preserving
    flat subdivision — the dual cells genuinely partition the surface regardless
    of triangulation.

  * A flat **subdivision** (split a flat face with a coplanar centroid vertex —
    primal geometry preserved, but the DUAL changes) does NOT leave |S_Regge|
    invariant: it drops to ¾·π√3. The new flat vertex (zero deficit) takes a
    share of the dual volume away from the curved vertices, so the deficit-
    weighted sum Σ_h |★h|·ε_h shrinks even though Σ_h |★h| and every ε_h are
    unchanged. This is the discrete Regge action's known triangulation-dependence,
    and it bounds acceptance (iv): the invariance is a property of dual-metric-
    preserving moves (relabeling), not of arbitrary subdivision.
"""

from __future__ import annotations

import math
import unittest

import pytest
import cmath

try:
    import tessera
    cob = tessera.cobordism
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

# Regular-tetrahedron surface (closed S²), and the same with face (1,2,3) split by
# a flat centroid vertex 4 (the three centroid edges have squared length 1/3).
_TETRA = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
_SUBDIV = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
_SUBDIV_EDGES = {(1, 4): 1.0 / 3.0, (2, 4): 1.0 / 3.0, (3, 4): 1.0 / 3.0}


def _surface(faces, edge_sq=None):
    edge_sq = edge_sq or {}
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.create_vertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.create_simplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        e.set_length(cmath.sqrt(complex(edge_sq.get((min(a, b), max(a, b)), 1.0))))
        e.set_phase(0.0)
    return st


def _regge_magnitude(st):
    return abs(tessera.ReggeSolver(st, tessera.MatterConfiguration()).dual_regge_action())


def _total_dual_volume(st):
    tessera.ReggeSolver(st, tessera.MatterConfiguration())  # materialize lattice in C++
    return sum(s.dual_volume() for s in st.get_simplices()
               if len(s.get_vertices()) == 1)


def _relabel(faces, perm):
    return [tuple(perm[v] for v in f) for f in faces]


class RelabelingInvariance(unittest.TestCase):
    """A relabeling preserves both metrics, so r_U and |S_Regge| are invariant."""

    PERMS = [{0: 0, 1: 1, 2: 2, 3: 3}, {0: 3, 1: 1, 2: 0, 3: 2},
             {0: 2, 1: 3, 2: 1, 3: 0}]

    def test_regge_action_invariant_under_relabeling(self):
        base = _regge_magnitude(_surface(_TETRA))
        self.assertAlmostEqual(base, math.pi * math.sqrt(3.0), places=7)
        for perm in self.PERMS[1:]:
            self.assertAlmostEqual(_regge_magnitude(_surface(_relabel(_TETRA, perm))),
                                   base, places=9)

    def test_residual_invariant_under_relabeling(self):
        # k=0 graph-Laplacian residual of a fixed (per-vertex) target; the value is
        # carried by the ORIGINAL vertex id, so relabeling is a pure re-labelling.
        def residual(perm):
            st = _surface(_relabel(_TETRA, perm))
            es = cob.EigenstateSynthesis(st, 0)
            inv = {v: k for k, v in perm.items()}
            order = [int(c[0]) for c in es.cell_simplices()]
            psi = [complex(inv[vid] + 1.0) for vid in order]
            return float(es.residual(psi))
        base = residual(self.PERMS[0])
        for perm in self.PERMS[1:]:
            self.assertAlmostEqual(residual(perm), base, places=9)


class DualPartitionIsGeometric(unittest.TestCase):
    def test_total_dual_volume_equals_area_under_both(self):
        area = math.sqrt(3.0)
        self.assertAlmostEqual(_total_dual_volume(_surface(_TETRA)), area, places=6)
        # a flat subdivision keeps the partition: Σ dual cells still tile the surface
        self.assertAlmostEqual(
            _total_dual_volume(_surface(_SUBDIV, _SUBDIV_EDGES)), area, places=6)


class SubdivisionFinding(unittest.TestCase):
    """The bound on acceptance (iv): a flat subdivision preserves the primal
    geometry and the total dual volume, but NOT the deficit-weighted |S_Regge|."""

    def test_flat_subdivision_redistributes_regge_action(self):
        base = _regge_magnitude(_surface(_TETRA))
        sub = _regge_magnitude(_surface(_SUBDIV, _SUBDIV_EDGES))
        self.assertLess(sub, base)                       # NOT invariant
        # the new flat vertex takes ¼ of the subdivided face's dual share from the
        # three curved vertices, so the deficit-weighted action drops to exactly ¾.
        self.assertAlmostEqual(sub / base, 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
