# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""§5.0 bulk eigenstate synthesis — fixed-boundary interior fill (#147).

The realizability oracle (#138) fills the *interior* of a bulk W_AB whose boundary
is pinned, matching a target output eigenstate. This extends EigenstateSynthesis
(#133) with a fixed-boundary mode: the tunable edges split into a boundary set dW
(edges on a codim-1 face in exactly one top cell — held fixed) and an interior set
(free). setInteriorWeights / setInteriorPhases drive only the interior, so a search
drives the §4b residual r = ||(I - psi psi^dagger) L psi||^2 -> 0 for a target while
dW stays byte-identical; grow_interior() cones a fresh interior vertex via the
boundary-fixed pre-geometric Pachner add (#112), the §4b cone-and-retry restricted
to the interior.

What is verified here:

  * **Partition** — on a solid triangle (dW = the 3 sides) every edge is boundary;
    one interior cone adds an apex with 3 interior edges while dW is unchanged; the
    class's dW agrees with cobordism.Cobordism.boundaryFaces; a 1-complex has no
    boundary (every edge interior) and cannot be grown.
  * **dW provably untouched** — across an interior-fill + growth sweep the boundary
    edge set, the boundary faces, AND the fixed boundary edge weights/phases are
    byte-identical throughout.
  * **Reachable matched (r < 1e-10)** — after interior fill a reachable target
    eigenvector is matched: the constant zero-mode (boundary phases 0), and a
    generic eigenvector reconstructed from a realized fixed-boundary Laplacian.
  * **Unreachable floors** — a generic target the interior cannot realize floors at
    a positive residual, matching an independent numpy global-min oracle (the
    obstruction signal, the analogue of §4b's two-vertex floor).
  * **Cone-and-retry** — a target unreachable at base complexity is matched after
    interior growth; the loop reports the minimal interior complexity reached.
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders (Signature(d) so the d-cells register as top simplices, the
# pre-geometric idiom shared with the boundary-fixed Pachner tests #112).
# --------------------------------------------------------------------------- #
def _spacetime(dim, topology):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topology)


def _solid_triangle():
    """Δ^2: a single triangle 0-1-2 whose boundary ∂W is the three sides
    (S^1). Built through the topology so the vertex-id counter advances —
    grow_interior()'s coned-in apex gets a collision-free id."""
    st = _spacetime(2, tessera.SolidSimplex(2))
    st.build()
    return st


def _bipyramid():
    """Two triangles 012 and 013 sharing the single interior edge 01; the four
    outer edges 02,12,03,13 are the fixed boundary ∂W. The smallest fixed-
    boundary complex with exactly one interior parameter — the analogue of §4b's
    two-vertex single edge, with the boundary pinned."""
    st = _solid_triangle()  # triangle 012, vertex-id counter at 3
    v = {x.get_id(): x for x in st.get_vertex_list().to_vector()}
    v3 = st.create_vertex(3)
    st.create_simplex([v[0], v[1], v3])  # triangle 013
    return st


def _two_vertex_edge():
    """The §4b 1-complex: two vertices, one edge (no boundary structure)."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    a = st.create_vertex(0)
    b = st.create_vertex(1)
    st.create_simplex([a, b])
    return st


# --------------------------------------------------------------------------- #
# numpy oracle + helpers (the same D - A magnitude convention the Hodge / #133
# tests use, in the operator's sorted-vertex-id order).
# --------------------------------------------------------------------------- #
def _cvec(v):
    return [complex(z) for z in v]


def _edge_map(st):
    """{(min_id, max_id): Edge} over the live edges."""
    out = {}
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        out[(min(a, b), max(a, b))] = e
    return out


def _edge_values(st):
    """{(min_id, max_id): (squaredLength, phase)} snapshot of the live edges."""
    return {k: ((e.get_length() * e.get_length()).real, e.get_phase())
            for k, e in _edge_map(st).items()}


def _boundary_faces(st):
    # The codimension-one boundary faces (incidence == 1), the canonical
    # Spacetime-owned derivation. (Formerly cob.Cobordism.boundaryFaces, a thin
    # wrapper over get_boundary(); that retired class is gone, get_boundary() is the
    # identical sorted vertex-id tuples.)
    return frozenset(tuple(sorted(f)) for f in st.get_boundary())


def _np_L(st):
    ids = sorted(v.get_id() for v in st.get_vertex_list().to_vector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n), dtype=complex)
    D = np.zeros(n)
    for e in st.get_edge_list().to_vector():
        s, t = e.get_source().get_id(), e.get_target().get_id()
        if s == t:
            continue
        i, j = idx[s], idx[t]
        w = (e.get_length() * e.get_length()).real
        z = w * np.exp(1j * e.get_phase())
        A[i, j] += z
        A[j, i] += np.conj(z)
        D[i] += abs(w)
        D[j] += abs(w)
    return np.diag(D).astype(complex) - A


def _np_residual(L, psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    r = Lp - lam * psi
    return float(np.vdot(r, r).real)


def _set_boundary(st, es, weights=None, phase_scale=0.0, base_w=1.0):
    """Pin the boundary edges to fixed Hermitian values, in es.boundary_edges()
    order. Returns the snapshot of the boundary edge values so a test can prove
    they never move."""
    em = _edge_map(st)
    bedges = es.boundary_edges()
    for k, key in enumerate(bedges):
        w = base_w if weights is None else weights[k]
        em[key].set_length(cmath.sqrt(complex(w)))
        em[key].set_phase(phase_scale * (k + 1))
    return {key: ((em[key].get_length() * em[key].get_length()).real, em[key].get_phase())
            for key in bedges}


# --------------------------------------------------------------------------- #
# Fixed-boundary searches: scipy L-BFGS-B over the *interior* parameters only,
# leaving dW fixed (the §4b multi-restart machinery, restricted to the interior).
# --------------------------------------------------------------------------- #
def _interior_search(es, psi, n_restarts=40, seed=0, w_bounds=(0.1, 10.0)):
    """Minimize r(psi) over the interior edge weights/phases (boundary fixed).
    Returns the best residual; leaves the complex at the best parameters."""
    from scipy.optimize import minimize

    m = es.num_interior_edges()
    psi = _cvec(psi)
    if m == 0:
        return es.residual(psi)
    rng = np.random.default_rng(seed)
    th_bounds = (-2.0 * math.pi, 2.0 * math.pi)
    bounds = [w_bounds] * m + [th_bounds] * m

    def objective(x):
        es.set_interior_weights(x[:m].tolist())
        es.set_interior_phases(x[m:].tolist())
        return es.residual(psi)

    best_r, best_x = np.inf, None
    for _ in range(n_restarts):
        x0 = np.concatenate([rng.uniform(w_bounds[0], w_bounds[1], size=m),
                             rng.uniform(-math.pi, math.pi, size=m)])
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_r:
            best_r, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    if best_x is not None:
        objective(best_x)
    return best_r


def _fill_interior(es, target_support, n_restarts=40, seed=0, max_cones=4,
                   tol=1e-10, w_bounds=(0.1, 10.0)):
    """The §4b cone-and-retry, restricted to the interior. Holds the target
    fixed on its boundary support (the first len(target_support) sorted-id
    vertices); the auxiliary amplitudes on coned-in apexes are free. Optimizes
    the interior, then grows (boundary-fixed cone), then re-optimizes, until
    r < tol or the growth budget is spent.

    Returns (best_r, interior_vertex_count) — the minimal interior complexity at
    convergence, or the residual floor if it cannot converge."""
    from scipy.optimize import minimize

    support = len(target_support)
    target_support = _cvec(target_support)
    th_bounds = (-2.0 * math.pi, 2.0 * math.pi)
    aux_bounds = (-5.0, 5.0)

    best_r = np.inf
    for _cone in range(max_cones + 1):
        m = es.num_interior_edges()
        n_aux = es.order() - support
        if m == 0 and n_aux == 0:
            best_r = es.residual(target_support)  # base: no interior freedom yet
        else:
            rng = np.random.default_rng(seed + _cone)
            bounds = ([w_bounds] * m + [th_bounds] * m
                      + [aux_bounds] * (2 * n_aux))

            def build_psi(x):
                aux = x[2 * m:]
                return target_support + [complex(aux[2 * k], aux[2 * k + 1])
                                         for k in range(n_aux)]

            def objective(x):
                if m:
                    es.set_interior_weights(x[:m].tolist())
                    es.set_interior_phases(x[m:2 * m].tolist())
                return es.residual(build_psi(x))

            cone_best, cone_x = np.inf, None
            for _ in range(n_restarts):
                x0 = np.concatenate([
                    rng.uniform(w_bounds[0], w_bounds[1], size=m),
                    rng.uniform(-math.pi, math.pi, size=m),
                    rng.uniform(-1.0, 1.0, size=2 * n_aux)])
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
                if res.fun < cone_best:
                    cone_best, cone_x = float(res.fun), np.asarray(res.x, float)
            if cone_x is not None:
                objective(cone_x)
            best_r = cone_best

        if best_r < tol:
            break
        if _cone < max_cones:
            es.grow_interior(1000 + _cone)
    return best_r, es.interior_vertex_count()


# --------------------------------------------------------------------------- #
class FixedBoundaryPartitionTest(unittest.TestCase):
    """The interior/boundary edge split, and that growth enriches the interior
    while ∂W is untouched."""

    def test_solid_triangle_is_all_boundary(self):
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        self.assertEqual(es.order(), 3)
        self.assertEqual(es.num_edges(), 3)
        self.assertEqual(es.num_interior_edges(), 0)
        self.assertEqual(es.num_boundary_edges(), 3)
        self.assertEqual(es.interior_vertex_count(), 0)
        self.assertEqual(sorted(es.boundary_edges()), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(es.interior_edges(), [])
        # The class's ∂W agrees with the cobordism boundary operator.
        self.assertEqual(frozenset(es.boundary_edges()), _boundary_faces(st))

    def test_bipyramid_has_one_interior_edge(self):
        st = _bipyramid()
        es = cob.EigenstateSynthesis(st)
        self.assertEqual(es.order(), 4)
        self.assertEqual(es.num_edges(), 5)
        self.assertEqual(es.num_interior_edges(), 1)
        self.assertEqual(es.num_boundary_edges(), 4)
        self.assertEqual(es.interior_edges(), [(0, 1)])
        self.assertEqual(sorted(es.boundary_edges()),
                         [(0, 2), (0, 3), (1, 2), (1, 3)])
        # The shared edge is interior even though both its endpoints lie on ∂W.
        self.assertEqual(es.interior_vertex_count(), 0)
        self.assertEqual(frozenset(es.boundary_edges()), _boundary_faces(st))

    def test_grow_enriches_interior_fixes_boundary(self):
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        boundary_before = frozenset(es.boundary_edges())
        faces_before = _boundary_faces(st)

        self.assertTrue(es.grow_interior(0))
        self.assertEqual(es.order(), 4)             # one apex coned in
        self.assertEqual(es.num_interior_edges(), 3)  # apex to each corner
        self.assertEqual(es.num_boundary_edges(), 3)
        self.assertEqual(es.interior_vertex_count(), 1)
        # ∂W exactly fixed.
        self.assertEqual(frozenset(es.boundary_edges()), boundary_before)
        self.assertEqual(_boundary_faces(st), faces_before)
        # The interior edges are precisely those incident to the new apex (id 3).
        self.assertEqual(sorted(es.interior_edges()), [(0, 3), (1, 3), (2, 3)])

    def test_one_complex_has_no_boundary_and_cannot_grow(self):
        st = _two_vertex_edge()
        es = cob.EigenstateSynthesis(st)
        # No codim-1 boundary structure: the lone edge is interior (free regime).
        self.assertEqual(es.num_boundary_edges(), 0)
        self.assertEqual(es.num_interior_edges(), 1)
        # Nothing to subdivide -> growth is a no-op.
        self.assertFalse(es.grow_interior(0))
        self.assertEqual(es.order(), 2)

    def test_interior_setters_size_mismatch_raises(self):
        es = cob.EigenstateSynthesis(_bipyramid())  # one interior edge
        with self.assertRaises(Exception):
            es.set_interior_weights([1.0, 2.0])
        with self.assertRaises(Exception):
            es.set_interior_phases([])


class BoundaryUntouchedTest(unittest.TestCase):
    """∂ fixed: the boundary face/edge set AND the fixed boundary edge
    weights/phases are byte-identical through an interior-fill + growth sweep."""

    def test_boundary_invariant_through_fill_and_growth(self):
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        # Pin the three boundary edges to distinct Hermitian values.
        boundary_vals = _set_boundary(
            st, es, weights=[0.7, 1.3, 2.1], phase_scale=0.37)
        boundary_edges0 = frozenset(es.boundary_edges())
        faces0 = _boundary_faces(st)

        def assert_boundary_fixed():
            self.assertEqual(frozenset(es.boundary_edges()), boundary_edges0)
            self.assertEqual(_boundary_faces(st), faces0)
            live = _edge_values(st)
            for key, (w, ph) in boundary_vals.items():
                self.assertIn(key, live)
                self.assertEqual(live[key][0], w)   # exact: never written
                self.assertEqual(live[key][1], ph)

        for step in range(3):
            self.assertTrue(es.grow_interior(step))
            assert_boundary_fixed()
            # An interior fill (random interior weights/phases) touches nothing
            # on ∂W.
            m = es.num_interior_edges()
            rng = np.random.default_rng(step)
            es.set_interior_weights(rng.uniform(0.2, 4.0, size=m).tolist())
            es.set_interior_phases(rng.uniform(-2.0, 2.0, size=m).tolist())
            assert_boundary_fixed()


class ReachableInteriorFillTest(unittest.TestCase):
    """A reachable target eigenvector is matched (r < 1e-10) by the interior
    fill, with the boundary held fixed."""

    def test_constant_zero_mode_is_matched(self):
        # Boundary edges pinned at phase 0 -> the constant vector is the zero
        # mode of L = D - A for any positive interior weights with phase 0, so
        # it is reachable by an interior fill.
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        es.set_weights([1.0] * es.num_edges())
        es.set_phases([0.0] * es.num_edges())
        es.grow_interior(0)
        es.grow_interior(1)

        n = es.order()
        target = np.ones(n) / math.sqrt(n)
        # Break it with a random interior config, then fill.
        m = es.num_interior_edges()
        rng = np.random.default_rng(5)
        es.set_interior_weights(rng.uniform(0.5, 3.0, size=m).tolist())
        es.set_interior_phases(rng.uniform(-1.0, 1.0, size=m).tolist())
        self.assertGreater(es.residual(_cvec(target)), 1e-2)

        best_r = _interior_search(es, target, n_restarts=30, seed=1)
        self.assertLess(best_r, 1e-10)
        # The C++ residual at the realized config matches the numpy oracle, and
        # the matched state is genuinely an eigenvector (L psi parallel to psi).
        self.assertAlmostEqual(es.residual(_cvec(target)),
                               _np_residual(_np_L(st), target), places=12)
        Lp = np.asarray(es.apply(_cvec(target)))
        self.assertAlmostEqual(abs(np.vdot(target, Lp)),
                               np.linalg.norm(Lp), places=6)

    def test_reconstructed_eigenvector_is_matched(self):
        # Grow a richer interior, pin a generic Hermitian boundary, realize a
        # config, and read off a genuine eigenvector of the resulting fixed-
        # boundary Laplacian. It is reachable by construction; the interior fill
        # recovers it from a fresh random start.
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        es.set_weights([1.0] * es.num_edges())
        es.set_phases([0.0] * es.num_edges())
        es.grow_interior(0)
        es.grow_interior(1)

        em = _edge_map(st)
        for k, key in enumerate(es.boundary_edges()):
            em[key].set_length(cmath.sqrt(complex(0.7 + 0.2 * k)))
            em[key].set_phase(0.1 * (k + 1))

        n, m = es.order(), es.num_interior_edges()
        rng = np.random.default_rng(9)
        es.set_interior_weights(rng.uniform(0.5, 2.0, size=m).tolist())
        es.set_interior_phases(rng.uniform(-1.0, 1.0, size=m).tolist())
        L = _np_L(st)
        evals, evecs = np.linalg.eigh(L)
        vstar = evecs[:, n // 2]
        # Reachable: it is exactly an eigenvector of the realized L.
        self.assertLess(es.residual(_cvec(vstar)), 1e-18)

        # Perturb the interior away, then fill back to match it.
        es.set_interior_weights(rng.uniform(0.5, 2.0, size=m).tolist())
        es.set_interior_phases(rng.uniform(-1.0, 1.0, size=m).tolist())
        self.assertGreater(es.residual(_cvec(vstar)), 1e-6)

        best_r = _interior_search(es, vstar, n_restarts=60, seed=0)
        self.assertLess(best_r, 1e-10)


class UnreachableFloorTest(unittest.TestCase):
    """A target the fixed-boundary interior cannot realize floors at a positive
    residual — the spectral obstruction, the analogue of §4b's two-vertex
    floor — verified against an independent numpy global-min oracle."""

    def test_generic_target_floors_above_zero(self):
        st = _bipyramid()  # one interior edge (0,1); four boundary edges fixed
        es = cob.EigenstateSynthesis(st)
        em = _edge_map(st)
        for key in es.boundary_edges():
            em[key].set_length(cmath.sqrt(complex(1.0)))
            em[key].set_phase(0.0)
        em[(0, 1)].set_length(cmath.sqrt(complex(1.0)))
        em[(0, 1)].set_phase(0.0)

        # A generic, fully-specified target; not realizable by any single
        # interior (w, theta) against the pinned boundary.
        target = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
        target /= np.linalg.norm(target)

        w_bounds = (0.1, 10.0)
        best_r = _interior_search(es, target, n_restarts=30, seed=0,
                                  w_bounds=w_bounds)

        # Oracle: the true min over the single interior edge (grid + refine).
        from scipy.optimize import minimize
        edge01 = em[(0, 1)]

        def oracle(x):
            edge01.set_length(cmath.sqrt(complex(x[0])))
            edge01.set_phase(x[1])
            return _np_residual(_np_L(st), target)

        grid = np.inf
        gx = None
        for w in np.linspace(w_bounds[0], w_bounds[1], 60):
            for th in np.linspace(-math.pi, math.pi, 60):
                val = oracle([w, th])
                if val < grid:
                    grid, gx = val, [w, th]
        ref = minimize(oracle, gx, method="L-BFGS-B",
                       bounds=[w_bounds, (-2 * math.pi, 2 * math.pi)])
        oracle_floor = float(ref.fun)

        # Floors bounded away from zero, and the C++ search reaches that floor.
        self.assertGreater(oracle_floor, 1e-2)
        self.assertGreater(best_r, 1e-2)
        self.assertAlmostEqual(best_r, oracle_floor, delta=2e-3)


class ConeAndRetryTest(unittest.TestCase):
    """The §4b cone-and-retry restricted to the interior: a target unreachable at
    base complexity is matched after interior growth, and the loop reports the
    minimal interior complexity reached (or the floor if it cannot converge)."""

    def test_reachable_after_interior_growth_reports_complexity(self):
        st = _solid_triangle()
        es = cob.EigenstateSynthesis(st)
        es.set_weights([1.0] * es.num_edges())   # boundary pinned (phase 0)
        es.set_phases([0.0] * es.num_edges())
        boundary_vals = {key: ((e.get_length() * e.get_length()).real, e.get_phase())
                         for key, e in _edge_map(st).items()
                         if key in set(es.boundary_edges())}

        # A generic target on the three boundary vertices; the auxiliary apex
        # amplitudes added by growth are free.
        target_support = [1.0 + 0.0j, 0.3 + 0.5j, -0.8 + 0.2j]

        # Base complex has no interior freedom -> unreachable.
        self.assertEqual(es.num_interior_edges(), 0)
        self.assertGreater(es.residual(_cvec(target_support)), 1e-2)

        best_r, complexity = _fill_interior(
            es, target_support, n_restarts=40, seed=0, max_cones=4)

        # Matched after interior fill, at a small reported interior complexity.
        self.assertLess(best_r, 1e-10)
        self.assertGreaterEqual(complexity, 1)
        self.assertEqual(complexity, es.interior_vertex_count())

        # ∂W untouched throughout the whole cone-and-retry.
        live = _edge_values(st)
        for key, (w, ph) in boundary_vals.items():
            self.assertEqual(live[key], (w, ph))

    def test_loop_returns_floor_when_it_cannot_converge(self):
        # Same generic, fully-specified target as the floor test, with no growth
        # budget: the loop returns the positive residual floor (the obstruction
        # signal #138 consumes), not a spurious zero.
        st = _bipyramid()
        es = cob.EigenstateSynthesis(st)
        em = _edge_map(st)
        for key in es.boundary_edges():
            em[key].set_length(cmath.sqrt(complex(1.0)))
            em[key].set_phase(0.0)
        target = [1.0, 2.0, 3.0, 4.0]

        best_r, complexity = _fill_interior(
            es, target, n_restarts=30, seed=0, max_cones=0)
        self.assertGreater(best_r, 1e-2)   # floored, not matched
        self.assertEqual(complexity, es.interior_vertex_count())


if __name__ == "__main__":
    unittest.main()
