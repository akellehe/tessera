# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""EigenstateSynthesis hardening — hand-checkable spectra + random complexes.

Complements tests/cobordism/test_eigenstate_synthesis_python.py (residual /
Rayleigh on the two-vertex edge and the square+diagonal testbed) and
test_bulk_eigenstate_synthesis_python.py (the fixed-boundary interior fill).
The gaps closed here:

  * CLOSED-FORM spectra on tiny graphs derived by hand and asserted against
    residual / Rayleigh / apply, not merely against numpy eigh:
      - the triangle K_3, uniform weight w, phase 0  -> spectrum {0, 3w, 3w};
        the constant vector is the harmonic zero mode and (1,0,0) has the exact
        residual 2 w^2;
      - K_3 with a flux Phi = pi threaded through the cycle -> spectrum
        {w, w, 4w}: the magnetic-Laplacian holonomy destroys the zero mode (the
        constant vector floors at the exact residual 8 w^2 / 9);
      - the path P_3 -> spectrum {0, w, 3w}; being a tree (b1 = 0) its spectrum
        is theta-independent (all phases are pure gauge).
  * residual == 0  <=>  L psi || psi, and Rayleigh = realized eigenvalue, on
    RANDOM connected complexes with random Hermitian weights/phases, against a
    numpy eigendecomposition (the base suite only checks two fixed complexes).
  * weight/phase get-set round-trips over the stable EdgeList order — phases
    included, with negative and >2*pi values — with the operator (apply)
    reflecting that order.
  * the general-amplitude two-vertex floor w_min^2 (|c0|^2 - |c1|^2)^2 matched by
    an independent numpy grid+refine GLOBAL-min oracle across amplitudes/seeds
    (the base suite asserts only the analytic closed form).
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders (the hand-crafted-complex idiom shared with the #133 tests).
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.create_vertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.create_simplex([verts[i] for i in simplex])
    return st


def _set_uniform(st, squared_length=1.0, phase=0.0):
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(squared_length)))
        e.set_phase(phase)


def _triangle():
    """K_3: vertices 0-1-2, edges (0,1) (1,2) (0,2). A cycle (b1 = 1)."""
    return _from_simplices(3, [(0, 1), (1, 2), (0, 2)])


def _path3():
    """P_3: 0-1-2, edges (0,1) (1,2). A tree (b1 = 0)."""
    return _from_simplices(3, [(0, 1), (1, 2)])


# --------------------------------------------------------------------------- #
# numpy oracle + helpers (the D - A magnitude convention the Hodge / #133 tests
# use, in the operator's sorted-vertex-id order).
# --------------------------------------------------------------------------- #
def _cvec(v):
    return [complex(z) for z in v]


def _edge_key(e):
    a, b = e.get_source().get_id(), e.get_target().get_id()
    return (min(a, b), max(a, b))


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
        w = (e.get_length()**2).real
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


# --------------------------------------------------------------------------- #
class HandCheckableTriangleTest(unittest.TestCase):
    """K_3 with closed-form spectra — zero flux and a pi flux through the cycle."""

    def test_zero_flux_spectrum_and_residuals(self):
        w = 1.3
        st = _triangle()
        _set_uniform(st, w, 0.0)
        es = cob.EigenstateSynthesis(st)

        # Hand spectrum of L = 3w I - w J (J the all-ones block): {0, 3w, 3w}.
        evals = np.linalg.eigvalsh(_np_L(st))
        np.testing.assert_allclose(sorted(evals), [0.0, 3 * w, 3 * w], atol=1e-9)

        # The constant vector is the harmonic zero mode.
        const = np.ones(3) / math.sqrt(3)
        self.assertLess(es.residual(_cvec(const)), 1e-18)
        self.assertAlmostEqual(es.rayleigh(_cvec(const)), 0.0, places=10)
        np.testing.assert_allclose(es.apply(_cvec(const)), 0.0, atol=1e-10)

        # The sum-zero modes realise the degenerate eigenvalue 3w exactly.
        for v in (np.array([1.0, -1.0, 0.0]), np.array([1.0, 1.0, -2.0])):
            vv = v / np.linalg.norm(v)
            self.assertLess(es.residual(_cvec(vv)), 1e-18)
            self.assertAlmostEqual(es.rayleigh(_cvec(vv)), 3 * w, places=9)
            np.testing.assert_allclose(
                np.asarray(es.apply(_cvec(vv))), 3 * w * vv, atol=1e-9)

        # A localised non-eigenvector (1,0,0): L psi = (2w, -w, -w), lambda = 2w,
        # so r = ||(0, -w, -w)||^2 = 2 w^2 exactly.
        e0 = np.array([1.0, 0.0, 0.0], dtype=complex)
        self.assertAlmostEqual(es.residual(_cvec(e0)), 2 * w * w, places=10)
        self.assertAlmostEqual(es.rayleigh(_cvec(e0)), 2 * w, places=10)
        self.assertAlmostEqual(es.residual(_cvec(e0)),
                               _np_residual(_np_L(st), e0), places=12)

    def test_flux_pi_destroys_the_zero_mode(self):
        # Thread a holonomy Phi = pi around the triangle by putting phase pi on a
        # single edge. The magnetic Laplacian then has spectrum {w, w, 4w}: the
        # zero eigenvalue is gone, so the constant vector is no longer harmonic.
        w = 1.3
        st = _triangle()
        _set_uniform(st, w, 0.0)
        for e in st.get_edge_list().to_vector():
            if _edge_key(e) == (0, 1):
                e.set_phase(math.pi)
        es = cob.EigenstateSynthesis(st)

        evals, evecs = np.linalg.eigh(_np_L(st))
        np.testing.assert_allclose(sorted(evals), [w, w, 4 * w], atol=1e-9)

        # The constant vector floors at the exact residual 8 w^2 / 9 > 0.
        const = np.ones(3, dtype=complex) / math.sqrt(3)
        self.assertGreater(es.residual(_cvec(const)), 1e-3)
        self.assertAlmostEqual(es.residual(_cvec(const)), 8 * w * w / 9.0,
                               places=10)
        self.assertAlmostEqual(es.residual(_cvec(const)),
                               _np_residual(_np_L(st), const), places=12)

        # The genuine min eigenvector is harmonic at eigenvalue w.
        vmin = evecs[:, 0]
        self.assertLess(es.residual(_cvec(vmin)), 1e-16)
        self.assertAlmostEqual(es.rayleigh(_cvec(vmin)), w, places=9)


class HandCheckablePathTest(unittest.TestCase):
    """P_3 closed-form spectrum, and tree => the spectrum is phase-independent."""

    def test_path_spectrum(self):
        w = 1.3
        st = _path3()
        _set_uniform(st, w, 0.0)
        es = cob.EigenstateSynthesis(st)

        evals = np.linalg.eigvalsh(_np_L(st))
        np.testing.assert_allclose(sorted(evals), [0.0, w, 3 * w], atol=1e-9)

        # Hand eigenvectors of the P_3 Laplacian.
        cases = [(np.array([1.0, 1.0, 1.0]), 0.0),       # constant -> 0
                 (np.array([1.0, 0.0, -1.0]), w),        # antisymmetric -> w
                 (np.array([1.0, -2.0, 1.0]), 3 * w)]    # curvature -> 3w
        for v, lam in cases:
            vv = v / np.linalg.norm(v)
            self.assertLess(es.residual(_cvec(vv)), 1e-18)
            self.assertAlmostEqual(es.rayleigh(_cvec(vv)), lam, places=9)
            np.testing.assert_allclose(
                np.asarray(es.apply(_cvec(vv))), lam * vv, atol=1e-9)

    def test_tree_spectrum_is_phase_independent(self):
        # On a tree every phase is pure gauge (b1 = 0): the spectrum is fixed at
        # {0, w, 3w} for any phase assignment, and the constant mode rephases into
        # the (still harmonic) zero mode the C++ residual confirms.
        w = 1.3
        st = _path3()
        _set_uniform(st, w, 0.0)
        es = cob.EigenstateSynthesis(st)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            for e in st.get_edge_list().to_vector():
                e.set_phase(float(rng.uniform(-3.5, 3.5)))
            L = _np_L(st)
            evals, evecs = np.linalg.eigh(L)
            np.testing.assert_allclose(sorted(evals), [0.0, w, 3 * w], atol=1e-9)
            # The numpy zero-mode (a rephased constant) is harmonic for the C++ op.
            zero_mode = evecs[:, int(np.argmin(evals))]
            self.assertLess(es.residual(_cvec(zero_mode)), 1e-16)
            self.assertAlmostEqual(es.rayleigh(_cvec(zero_mode)), 0.0, places=9)


class RandomComplexResidualTest(unittest.TestCase):
    """residual == 0 <=> L psi || psi and Rayleigh = eigenvalue, on random
    connected complexes with random Hermitian weights/phases."""

    @staticmethod
    def _random_complex(seed, n):
        rng = np.random.default_rng(seed)
        # A random spanning tree (connected), then a few extra chords.
        perm = list(rng.permutation(n))
        edges = set()
        for k in range(1, n):
            j = perm[int(rng.integers(0, k))]
            edges.add((min(perm[k], j), max(perm[k], j)))
        extra = rng.integers(1, n)
        while len(edges) < (n - 1) + extra:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b:
                edges.add((min(a, b), max(a, b)))
        st = _from_simplices(n, sorted(edges))
        for e in st.get_edge_list().to_vector():
            e.set_length(cmath.sqrt(complex(float(rng.uniform(0.4, 3.0)))))
            e.set_phase(float(rng.uniform(-math.pi, math.pi)))
        return st

    def test_eigenvectors_are_harmonic_and_random_vector_is_not(self):
        for seed in range(6):
            n = 4 + (seed % 3)  # 4, 5, 6 vertices
            with self.subTest(seed=seed, n=n):
                st = self._random_complex(seed, n)
                es = cob.EigenstateSynthesis(st)
                self.assertEqual(es.order(), n)
                L = _np_L(st)
                evals, evecs = np.linalg.eigh(L)
                for k in range(n):
                    v = evecs[:, k]
                    self.assertLess(es.residual(_cvec(v)), 1e-10)
                    self.assertAlmostEqual(es.rayleigh(_cvec(v)), evals[k],
                                           places=8)
                    Lv = np.asarray(es.apply(_cvec(v)))
                    np.testing.assert_allclose(Lv, evals[k] * v, atol=1e-8)
                    # L v genuinely parallel to v: |<v, Lv>| == ||Lv||.
                    self.assertAlmostEqual(abs(np.vdot(v, Lv)),
                                           np.linalg.norm(Lv), places=8)

                # A generic vector is not an eigenvector: r > 0, matching numpy,
                # and L u is not parallel to u.
                rng = np.random.default_rng(1000 + seed)
                u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
                u /= np.linalg.norm(u)
                r = es.residual(_cvec(u))
                self.assertGreater(r, 1e-6)
                self.assertAlmostEqual(r, _np_residual(L, u), places=10)
                Lu = np.asarray(es.apply(_cvec(u)))
                self.assertLess(abs(np.vdot(u, Lu)), np.linalg.norm(Lu) - 1e-9)


class ParameterOrderTest(unittest.TestCase):
    """weights() / phases() get-set round-trips, in the stable EdgeList order,
    with the operator reflecting that order. Extends the base round-trip test to
    the phase channel and to out-of-range phase values."""

    def test_weight_phase_order_matches_edgelist_and_operator(self):
        st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        es = cob.EigenstateSynthesis(st)
        m = es.num_edges()
        self.assertEqual(m, 5)

        # Distinct weights and phases including a negative and a > 2*pi value
        # (the phase must round-trip verbatim — no wrapping or clamping).
        w = [0.4 + 0.7 * i for i in range(m)]
        th = [-3.7, 0.0, 7.25, -0.4, 5.9]
        es.set_weights(w)
        es.set_phases(th)
        self.assertTrue(np.allclose(es.weights(), w))
        self.assertTrue(np.allclose(es.phases(), th))

        # The k-th parameter is the k-th EdgeList edge (the stable order).
        live = st.get_edge_list().to_vector()
        self.assertTrue(np.allclose([(e.get_length()**2).real for e in live], w))
        self.assertTrue(np.allclose([e.get_phase() for e in live], th))

        # And the operator (apply) is built in that same edge order: it agrees
        # with the numpy L assembled directly from the live edges.
        rng = np.random.default_rng(0)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        np.testing.assert_allclose(
            np.asarray(es.apply(_cvec(psi))), _np_L(st) @ psi, atol=1e-10)


class GeneralAmplitudeFloorOracleTest(unittest.TestCase):
    """The two-vertex floor w_min^2 (|c0|^2 - |c1|^2)^2 confirmed against an
    INDEPENDENT numpy grid+refine global-min oracle (not just the closed form),
    and reached by a scipy search over the C++ residual."""

    W_MIN, W_MAX = 0.1, 10.0

    def _two_vertex(self):
        st = _from_simplices(2, [(0, 1)])
        _set_uniform(st, 1.0, 0.0)
        return st

    def _cpp_floor(self, es, psi, seed, restarts=40):
        from scipy.optimize import minimize
        psi = _cvec(psi)
        rng = np.random.default_rng(seed)
        bounds = [(self.W_MIN, self.W_MAX), (-2.0 * math.pi, 2.0 * math.pi)]

        def objective(x):
            es.set_weights([x[0]])
            es.set_phases([x[1]])
            return es.residual(psi)

        best = np.inf
        for _ in range(restarts):
            x0 = [rng.uniform(self.W_MIN, self.W_MAX), rng.uniform(-math.pi, math.pi)]
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            best = min(best, float(res.fun))
        return best

    def _oracle_floor(self, st, psi):
        # Brute-force global min of the numpy residual over the single edge
        # (w, theta), then a local refine from the best grid point — fully
        # independent of the C++ code and of the analytic closed form.
        from scipy.optimize import minimize
        edge = st.get_edge_list().to_vector()[0]

        def f(x):
            edge.set_length(cmath.sqrt(complex(x[0])))
            edge.set_phase(x[1])
            return _np_residual(_np_L(st), psi)

        best, bx = np.inf, None
        for ww in np.linspace(self.W_MIN, self.W_MAX, 40):
            for th in np.linspace(-math.pi, math.pi, 40):
                val = f([ww, th])
                if val < best:
                    best, bx = val, [ww, th]
        ref = minimize(f, bx, method="L-BFGS-B",
                       bounds=[(self.W_MIN, self.W_MAX), (-2 * math.pi, 2 * math.pi)])
        return float(ref.fun)

    def test_floor_matches_global_min_oracle(self):
        for i, p0 in enumerate((0.6, 0.75, 0.85, 0.95)):
            with self.subTest(p0=p0):
                psi = np.array([math.sqrt(p0), math.sqrt(1.0 - p0)], dtype=complex)
                closed = self.W_MIN ** 2 * (2.0 * p0 - 1.0) ** 2

                oracle = self._oracle_floor(self._two_vertex(), psi)
                cpp = self._cpp_floor(cob.EigenstateSynthesis(self._two_vertex()),
                                      psi, seed=i + 1)

                # A positive floor (the obstruction motivating coning): it scales
                # as d^2 = (2 p0 - 1)^2, so it can be small for a mild imbalance,
                # but it is strictly nonzero (a balanced qubit reaches ~0 instead).
                self.assertGreater(oracle, 1e-5)
                # The brute-force global min equals the analytic closed form ...
                self.assertAlmostEqual(oracle, closed, delta=max(2e-3, 0.02 * closed))
                # ... and the scipy search over the C++ residual reaches it.
                self.assertAlmostEqual(cpp, oracle, delta=max(2e-3, 0.05 * closed))


if __name__ == "__main__":
    unittest.main()
