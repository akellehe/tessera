# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""§4b boundary-state synthesis — residual + optimizer on a fixed complex (#133).

EigenstateSynthesis scores how close the complex's current Hermitian edge weights
make a target qubit psi to being an eigenvector of the k=0 Hodge Laplacian
L = D - A (magnitude convention), via the eigenvalue-agnostic residual
r(psi) = ||(I - psi psi^dagger) L psi||^2, and reads/writes the edge magnitudes
{w_ij} and phases {theta_ij} so a search can perturb them. The non-convex,
multi-restart search lives here in the Python driver (scipy L-BFGS-B over the flat
[w; theta] vector), calling the C++ residual.

Acceptance (numpy/scipy oracle), on the §4b two-vertex single-edge complex:
  * a BALANCED qubit (e^{i*theta}, ±1)/√2 is recovered to r < 1e-10, and the
    Rayleigh quotient is the realized eigenvalue;
  * a GENERAL-amplitude qubit (|c0| != |c1|) cannot be a two-vertex eigenvector —
    the search floor on r stays bounded away from 0 (the motivation for #134's
    auxiliary vertices). The single edge has Laplacian
    L = [[|w|, -w e^{i*theta}], [-w e^{-i*theta}, |w|]] whose eigenvectors are the
    balanced (e^{i*theta}, ±1)/√2; over a box w in [w_min, w_max] the residual of
    an unbalanced qubit floors at the closed form w_min^2 (|c0|^2 - |c1|^2)^2 > 0;
  * residual == 0  <=>  L psi || psi, cross-checked against a direct numpy
    eigendecomposition (and on a richer 4-vertex complex too).
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism


# --------------------------------------------------------------------------- #
# Fixture builders (the hand-crafted-complex idiom shared with the Hodge tests)
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


def _two_vertex_edge():
    """The §4b seed of the inverse problem: two vertices, one edge (K_2)."""
    st = _from_simplices(2, [(0, 1)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _testbed():
    """Square 00-01-11-10-00 plus the entangling diagonal 00-11 (|V|=4, |E|=5)."""
    st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


# --------------------------------------------------------------------------- #
# numpy oracle + helpers
# --------------------------------------------------------------------------- #
def _cvec(v):
    """A plain list of python complex (pybind's std::vector<complex> caster)."""
    return [complex(z) for z in v]


def _np_L(st):
    """Independent D - A reference (magnitude convention), in the operator's
    stable sorted-vertex-id order — the same oracle the Hodge tests use."""
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
    """The spec residual r = ||(I - psi psi^dagger) L psi||^2 for unit psi."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    Lp = L @ psi
    lam = np.vdot(psi, Lp).real
    r = Lp - lam * psi
    return float(np.vdot(r, r).real)


def _synthesize(es, psi, n_restarts=40, seed=0, w_bounds=(0.1, 10.0)):
    """Non-convex multi-restart search (scipy L-BFGS-B) over the flat [w; theta]
    vector minimizing the C++ residual r(psi). Leaves the complex at the best
    parameters found and returns (best_r, best_x)."""
    from scipy.optimize import minimize

    m = es.num_edges()
    psi = _cvec(psi)
    rng = np.random.default_rng(seed)
    th_bounds = (-2.0 * math.pi, 2.0 * math.pi)
    bounds = [w_bounds] * m + [th_bounds] * m

    def objective(x):
        es.set_weights(x[:m].tolist())
        es.set_phases(x[m:].tolist())
        return es.residual(psi)

    best_r, best_x = np.inf, None
    for _ in range(n_restarts):
        w0 = rng.uniform(w_bounds[0], w_bounds[1], size=m)
        th0 = rng.uniform(-math.pi, math.pi, size=m)
        res = minimize(objective, np.concatenate([w0, th0]),
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best_r:
            best_r, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    objective(best_x)  # leave the complex realized at the optimum
    return best_r, best_x


# --------------------------------------------------------------------------- #
class EigenstateSynthesisStructureTest(unittest.TestCase):
    """Shape / parameter-access surface."""

    def test_order_and_edge_count(self):
        es = cob.EigenstateSynthesis(_two_vertex_edge())
        self.assertEqual(es.order(), 2)
        self.assertEqual(es.num_edges(), 1)

        es5 = cob.EigenstateSynthesis(_testbed())
        self.assertEqual(es5.order(), 4)
        self.assertEqual(es5.num_edges(), 5)

    def test_weights_phases_roundtrip(self):
        st = _testbed()
        es = cob.EigenstateSynthesis(st)
        w = [0.5, 1.5, 2.0, 0.25, 3.0]
        th = [0.1, -0.2, 0.3, -0.4, 0.5]
        es.set_weights(w)
        es.set_phases(th)
        self.assertTrue(np.allclose(es.weights(), w))
        self.assertTrue(np.allclose(es.phases(), th))
        # And the writes reach the underlying edges (the Laplacian sees them).
        self.assertTrue(np.allclose(np.array(es.weights()),
                                    [(e.get_length() * e.get_length()).real
                                     for e in st.get_edge_list().to_vector()]))

    def test_size_mismatch_raises(self):
        es = cob.EigenstateSynthesis(_two_vertex_edge())
        with self.assertRaises(Exception):
            es.residual([1.0 + 0j])            # needs length order() == 2
        with self.assertRaises(Exception):
            es.set_weights([1.0, 2.0])          # needs length num_edges() == 1


class ResidualParallelTest(unittest.TestCase):
    """residual == 0  <=>  L psi || psi, cross-checked vs numpy eigh."""

    def _check(self, st):
        es = cob.EigenstateSynthesis(st)
        L = _np_L(st)
        evals, evecs = np.linalg.eigh(L)
        n = L.shape[0]
        for k in range(n):
            v = evecs[:, k]
            # eigenvector => residual 0 (C++ matches the numpy oracle)
            self.assertLess(es.residual(_cvec(v)), 1e-18)
            self.assertAlmostEqual(es.residual(_cvec(v)), _np_residual(L, v),
                                   places=14)
            # ... and L v is genuinely parallel to v: |<v, Lv>| == ||Lv||
            Lv = np.asarray(es.apply(_cvec(v)))
            self.assertAlmostEqual(abs(np.vdot(v, Lv)), np.linalg.norm(Lv),
                                   places=10)
            self.assertTrue(np.allclose(Lv, evals[k] * v, atol=1e-10))
            # realized eigenvalue == Rayleigh quotient == numpy eigenvalue
            self.assertAlmostEqual(es.rayleigh(_cvec(v)), evals[k], places=10)

    def test_two_vertex(self):
        st = _two_vertex_edge()
        _set_uniform(st, 1.3, 0.5)  # arbitrary nonzero Hermitian weight
        self._check(st)
        # a generic NON-eigenvector => residual > 0 and L u not parallel to u
        es = cob.EigenstateSynthesis(st)
        u = np.array([0.6, 0.8j], dtype=complex)
        u /= np.linalg.norm(u)
        r = es.residual(_cvec(u))
        self.assertGreater(r, 1e-3)
        self.assertAlmostEqual(r, _np_residual(_np_L(st), u), places=12)

    def test_richer_complex(self):
        st = _testbed()
        # generic Hermitian weights so the spectrum is non-degenerate
        for i, e in enumerate(st.get_edge_list().to_vector()):
            e.set_length(cmath.sqrt(complex(0.7 + 0.3 * i)))
            e.set_phase(0.2 * (i + 1))
        self._check(st)


class BalancedRecoveryTest(unittest.TestCase):
    """A balanced qubit is recoverable to r < 1e-10 on the single edge; the
    realized eigenvalue is the Rayleigh quotient."""

    def test_balanced_recovered(self):
        for sign in (+1.0, -1.0):
            for alpha in (0.0, 0.7, -1.3):
                with self.subTest(sign=sign, alpha=alpha):
                    st = _two_vertex_edge()
                    es = cob.EigenstateSynthesis(st)
                    psi = np.array([np.exp(1j * alpha), sign]) / math.sqrt(2.0)
                    best_r, _ = _synthesize(es, psi, seed=abs(int(10 * alpha)) + 1)
                    self.assertLess(best_r, 1e-10)

                    # The Rayleigh quotient is the realized eigenvalue: cross-check
                    # against numpy eigh of the synthesized Laplacian.
                    lam = es.rayleigh(_cvec(psi))
                    L = _np_L(st)
                    evals, evecs = np.linalg.eigh(L)
                    overlaps = [abs(np.vdot(evecs[:, k], psi)) for k in range(2)]
                    k = int(np.argmax(overlaps))
                    self.assertAlmostEqual(overlaps[k], 1.0, places=5)
                    self.assertAlmostEqual(lam, evals[k], places=6)
                    # A balanced state can be realized as either branch of the
                    # single edge — the kernel mode (lambda = 0, theta = alpha) or
                    # the lambda = 2w mode (theta = alpha + pi), since
                    # (e^{i(alpha+pi)}, -1)/sqrt2 = -(e^{i*alpha}, +1)/sqrt2 is the
                    # same ray. The realized eigenvalue is whichever the search
                    # lands on; it must equal the Rayleigh quotient (checked above)
                    # and be one of the two analytic eigenvalues {0, 2w}.
                    w = es.weights()[0]
                    self.assertTrue(min(abs(lam - 0.0), abs(lam - 2.0 * w)) < 1e-6)


class GeneralAmplitudeFloorTest(unittest.TestCase):
    """An unbalanced qubit cannot be a two-vertex eigenvector: the residual floors
    bounded away from 0 — the motivation for auxiliary vertices (#134)."""

    def test_floor_bounded_away_from_zero(self):
        w_min = 0.1

        # Balanced sibling: reaches ~0 on the same single edge.
        es_bal = cob.EigenstateSynthesis(_two_vertex_edge())
        bal = np.array([1.0, 1.0]) / math.sqrt(2.0)
        r_bal, _ = _synthesize(es_bal, bal, seed=7, w_bounds=(w_min, 10.0))
        self.assertLess(r_bal, 1e-10)

        # Unbalanced qubit: |c0|^2 = 0.8, |c1|^2 = 0.2.
        a, b = math.sqrt(0.8), math.sqrt(0.2)
        es_gen = cob.EigenstateSynthesis(_two_vertex_edge())
        gen = np.array([a, b], dtype=complex)
        r_gen, _ = _synthesize(es_gen, gen, seed=8, w_bounds=(w_min, 10.0))

        # Closed-form floor over the box: min_{w,theta} w^2 (d^2 cos^2 + sin^2)
        #   = w_min^2 * d^2,  d = |c0|^2 - |c1|^2.
        d = a * a - b * b
        oracle_floor = w_min ** 2 * d ** 2
        self.assertAlmostEqual(r_gen, oracle_floor, delta=1e-4)
        self.assertGreater(r_gen, 1e-3)          # bounded away from 0
        # The contrast is the whole point: the balanced sibling reaches ~0 while
        # the unbalanced qubit cannot — it needs auxiliary scaffolding (#134).
        self.assertLess(r_bal, 1e-6 * r_gen)

    def test_floor_scales_with_imbalance(self):
        # The floor grows with the amplitude imbalance d = |c0|^2 - |c1|^2.
        w_min = 0.1
        prev = -1.0
        for p0 in (0.55, 0.7, 0.9):
            es = cob.EigenstateSynthesis(_two_vertex_edge())
            psi = np.array([math.sqrt(p0), math.sqrt(1.0 - p0)], dtype=complex)
            r, _ = _synthesize(es, psi, seed=int(100 * p0), w_bounds=(w_min, 10.0))
            d = 2.0 * p0 - 1.0
            floor = w_min ** 2 * d ** 2
            # The box minimum is the closed form; the search reaches it (it cannot
            # go below — that is the global min over w in [w_min, w_max], theta).
            self.assertLessEqual(abs(r - floor), 0.1 * floor + 2e-5)
            self.assertGreater(r, prev)  # grows with the amplitude imbalance
            prev = r


if __name__ == "__main__":
    unittest.main()
