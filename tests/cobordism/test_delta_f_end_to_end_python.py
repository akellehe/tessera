# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""End-to-end incremental ΔF = Δ‖∇S_Regge‖² + Γ·Δr_U across the move classes (#461).

The Emergent Color Topology optimizer's objective is `F = ‖∇S_Regge‖² + Γ·r_U`
(extremize the action, δS=0). The incremental `ΔF` must reproduce a full `F`
recompute to ~machine precision while touching only the moved region for the
geometry term:

  * **Stage-2 edge-length perturbation** on the **k=2** b₂ register: the full
    assembly `ΔF = Δ‖∇S‖²(local) + Γ·Δr_U(recompute)` equals the full `F` delta.
  * **Stage-1 surgical cone-out** (a topology change): the local `Δ‖∇S‖²` equals the
    full `‖∇S‖²` delta even as cells are removed; and the surgery shifts the `b₂`
    register dimension, with `r_U` exactly recomputable on the new complex.

The geometry term is hinge-local and exact; `r_U` is a global spectral quantity, so
its exact delta is a before/after `residual_for_periods` recompute.
"""
import unittest

import tessera as T
import cmath

cob = T.cobordism

_GAMMA = 1.0
_TOL = 1e-11


def _grad_norm2(st):
    rs = T.ReggeSolver(st, T.MatterConfiguration())
    return sum(abs(z) ** 2 for z in rs.action_gradient_exact())


def _tops(st):
    return {tuple(sorted(v.get_id() for v in c.get_vertices()))
            for c in st.get_top_simplices()}


def _holed_s3():
    # A refined S^3 opened by a disjoint pair of surgical cone-outs (raising b_2 by 1):
    # the b_2 color register, built from first principles. Returns the complex and the
    # two emergent hole tetrahedra (the over-constrained 2-cycle period rows).
    st = _refined_s3()
    cells = sorted(_tops(st))
    pair = None
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            if set(a).isdisjoint(b):
                pair = (a, b)
                break
        if pair:
            break
    a, b = pair
    sc = cob.SurgicalCone(st)
    sc.cone_out(list(a))
    sc.cone_out(list(b))
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.01 * (i % 7))))
    return st, [[list(a), list(b)]]


def _cdt4(n=160):
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED, T.Toroid())
    st.build(n)
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.011 * (i % 5))))
    return st


def _sphere3():
    sig = T.Signature(3, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(3))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
    return st


def _verts(st):
    return {v.get_id() for v in st.get_vertex_list().to_vector() if v is not None}


def _edges(st):
    return {(min(e.get_source().get_id(), e.get_target().get_id()),
             max(e.get_source().get_id(), e.get_target().get_id()))
            for e in st.get_edge_list().to_vector()}


def _refined_s3(n_refine=12):
    # A refined S^3 (Betti [1,0,0,1]) with enough tetrahedra that a disjoint top-cell
    # pair exists — the minimal S^3 has none (every facet pair shares a ridge).
    sig = T.Signature(3, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(3))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
    for seed in range(n_refine):
        mv = T.AddMove(st, seed, False, T.PachnerMode.PreGeometric, False)
        if mv.propose():
            mv.apply()
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.01 * (i % 6))))
    return st


class DeltaFEndToEndTest(unittest.TestCase):
    def test_delta_F_edge_perturbation_on_k2_register(self):
        # ΔF = Δ‖∇S‖²(local) + Γ·Δr_U(recompute) == full F delta, on the b₂ register.
        st, windows = _holed_s3()
        es = cob.EigenstateSynthesis(st, 2)
        holes = [list(h) for h in windows[0]]
        target = [complex(1.0), complex(0.3)]   # 2 holes, 1 mode ⇒ non-carriable
        rs = T.ReggeSolver(st, T.MatterConfiguration())

        def full_F():
            return _grad_norm2(st) + _GAMMA * es.residual_for_periods(holes, target)

        e = st.get_edge_list().to_vector()[5]
        ev = {e.get_source().get_id(), e.get_target().get_id()}
        aff = [list(c) for c in _tops(st) if ev.issubset(set(c))]
        E = rs.affected_edges_of_cells(aff)

        before_F = full_F()
        before_gn = rs.gradient_norm2_over_edges(E)
        before_ru = es.residual_for_periods(holes, target)
        orig = (e.get_length() * e.get_length())
        e.set_length(cmath.sqrt(complex(orig * 1.06)))
        after_F = full_F()
        after_gn = rs.gradient_norm2_over_edges(E)
        after_ru = es.residual_for_periods(holes, target)
        e.set_length(cmath.sqrt(complex(orig)))

        d_full = after_F - before_F
        d_incr = (after_gn - before_gn) + _GAMMA * (after_ru - before_ru)
        self.assertGreater(before_ru, 1.0)          # the register actually scores
        self.assertLess(abs(d_full - d_incr), 1e-12)

    def test_delta_gradnorm_across_surgical_coneout_with_multiplicities(self):
        # Stage-1 topology change on a CDT (high edge multiplicity): cone-out
        # DECREMENTS shared-edge multiplicity rather than deleting the edge, and the
        # local Δ‖∇S‖² (which captures the decremented edges as part of the removed
        # cell's neighborhood) equals the full Δ.
        st = _cdt4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        tops_before = _tops(st)

        # the complex genuinely has edge multiplicity: edges shared by >2 top cells
        from collections import Counter
        share = Counter()
        for c in tops_before:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    share[(c[i], c[j])] += 1
        self.assertTrue(any(n > 2 for n in share.values()),
                        "fixture lacks edge multiplicity")
        before_full = _grad_norm2(st)

        sc = cob.SurgicalCone(st)
        coned = None
        for c in list(tops_before):
            ok, _reason = sc.cone_out(list(c))
            if ok:
                coned = c
                break
        if coned is None:
            self.skipTest("no surgical cone-out accepted on this build")

        # the decrement path ran: most of the coned cell's edges are still covered by
        # surviving cells, so they survive (were decremented, not removed).
        edges_after = {(min(e.get_source().get_id(), e.get_target().get_id()),
                        max(e.get_source().get_id(), e.get_target().get_id()))
                       for e in st.get_edge_list().to_vector()}
        cell_edges = [(coned[i], coned[j])
                      for i in range(len(coned)) for j in range(i + 1, len(coned))]
        survived = [e for e in cell_edges if e in edges_after]
        self.assertTrue(len(survived) > 0,
                        "cone-out removed every edge — decrement path not exercised")

        affected = [list(x) for x in (tops_before ^ _tops(st))]
        st0 = _cdt4()  # fresh identical before-complex (no reliance on rollback)
        rs0 = T.ReggeSolver(st0, T.MatterConfiguration())
        E = sorted(set(map(tuple, rs0.affected_edges_of_cells(affected)))
                   | set(map(tuple, rs.affected_edges_of_cells(affected))))
        self.assertTrue(E)
        d_full = _grad_norm2(st) - before_full
        d_loc = rs.gradient_norm2_over_edges(E) - rs0.gradient_norm2_over_edges(E)
        self.assertLess(abs(d_full - d_loc), 1e-9)

    def test_surgery_shifts_register_and_ru_recomputes(self):
        # A disjoint PAIR of cone-outs opens a b₂ hole (the register dimension shifts);
        # r_U is exactly recomputable on the post-surgery complex, and its arbitrary-k
        # analytic gradient stays exact there (the Euler identity Σℓ²∂r_U = −2·r_U
        # under the V^2 weights: L is degree −1 in ℓ², r_U quadratic in it).
        st = _refined_s3()
        cells = sorted(_tops(st))
        pair = None
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                if set(a).isdisjoint(b):
                    pair = (a, b)
                    break
            if pair:
                break
        self.assertIsNotNone(pair, "refined S^3 must contain a disjoint cell pair")
        a, b = pair

        b2_before = list(cob.ChainComplex.from_spacetime(st).betti_numbers())[2]
        sc = cob.SurgicalCone(st)
        self.assertTrue(sc.cone_out(list(a))[0])   # opens the manifold (b₃ → 0)
        self.assertTrue(sc.cone_out(list(b))[0])   # disjoint ⇒ raises b₂ by 1
        b2_after = list(cob.ChainComplex.from_spacetime(st).betti_numbers())[2]
        self.assertEqual(b2_after, b2_before + 1, "surgery did not shift the register")

        # r_U over BOTH emergent holes (2 holes, 1 harmonic mode ⇒ over-constrained,
        # so r_U > 0) is exactly evaluable, and its analytic gradient is exact.
        es = cob.EigenstateSynthesis(st, 2)
        holes = [list(a), list(b)]
        target = [complex(1.0), complex(0.3)]
        r_u = es.residual_for_periods(holes, target)
        self.assertGreater(r_u, 1.0)
        g = es.residual_for_periods_gradient(holes, target)
        l2 = {tuple(sorted((e.get_source().get_id(), e.get_target().get_id()))):
              (e.get_length() * e.get_length()).real for e in st.get_edge_list().to_vector()}
        edges = [tuple(sorted(c)) for c in
                 cob.ChainComplex.from_spacetime(st).k_simplex_vertices(1)]
        euler = sum(l2[edges[i]] * g[i] for i in range(len(edges)))
        self.assertLess(abs(euler + 2.0 * r_u) / r_u, 1e-9,
                        "Σℓ²∂r_U = −2·r_U failed on the post-surgery complex")


class ConeInverseTest(unittest.TestCase):
    """Cone-out is the EXACT inverse of cone-in: it removes only the k apex
    ("out") edges of the created top cell (the orphans), never the covered base
    edges — so cone-in followed by cone-out restores the structure bit-for-bit."""

    def test_cone_out_is_exact_inverse_of_cone_in(self):
        st = _sphere3()
        sc = cob.SurgicalCone(st)
        # open a boundary so a cone-in onto a boundary triangle is admissible
        self.assertTrue(sc.cone_out([0, 1, 2, 3])[0])
        tops0, edges0, verts0 = _tops(st), _edges(st), _verts(st)

        # cone-in on the boundary triangle {0,1,2} → a fresh apex + the new cell
        self.assertTrue(sc.cone_in([0, 1, 2])[0])
        apex = (_verts(st) - verts0).pop()                  # the one fresh vertex
        added_edges = _edges(st) - edges0
        # exactly the k = d apex ("out") edges were added (apex→0, apex→1, apex→2)
        self.assertEqual(added_edges, {(min(apex, b), max(apex, b)) for b in (0, 1, 2)})
        created = sorted([apex, 0, 1, 2])

        # the explicit inverse move (NOT rollback): cone the created cell out
        self.assertTrue(sc.cone_out(created)[0])

        # structure restored bit-for-bit: only the k apex edges + apex went away,
        # the covered base edges {0,1},{0,2},{1,2} survived.
        self.assertEqual(_tops(st), tops0, "cone-out is not the inverse of cone-in")
        self.assertEqual(_edges(st), edges0, "cone-out removed a covered base edge")
        self.assertEqual(_verts(st), verts0, "cone-out left/removed a stray vertex")


if __name__ == "__main__":
    unittest.main()
