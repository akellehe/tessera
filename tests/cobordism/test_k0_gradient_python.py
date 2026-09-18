"""The k = 0 analytic r_U gradient runs against the genuinely complex L_0 (#589).

The k = 0 metric operator is L_0 = D - A with D_ii = sum |l^2| and
A_ij = l^2 e^{i phase} — Hermitian COMPLEX (HodgeLaplacian::assemble consumes
the full complex l^2 and the U(1) phases), so the k >= 1 cores'
``laplacian(k).real()`` projection would be the wrong operator there. #582
guarded this with a throw; #589 replaces the guard with the correct complex
core (``periodGradientDegreeZero``): the four-entry per-edge dL_0
(dL_ii = dL_jj = d|w|/dw along the real axis = sign w, dL_ij = -e^{i phase}),
the complex Hermitian eigensplit, and the SVD pseudo-inverse fit with its
constant-rank (Golub–Pereyra) derivative — at k = 0 a globally gauge-flat
harmonic has zero period on every hole, so the period matrix is generically
rank-deficient and the k >= 1 normal-equations inverse would be singular.

Validation is the #461 recipe: the exact Euler identity — at k = 0 it is
``sum_e l^2_e dr_U/dl^2_e = +2 r_U`` (L_0 is homogeneous of degree +1 in l^2,
vs -r_U for the degree -1 metric L_k at k >= 1) — plus internal finite
differences along real perturbations.

A k = 0 register, concretely: holes are removed 1-cells (vertex pairs), the
periods are twisted differences of the gauge-flat vertex harmonics, and
carrying is possible exactly across connected components — so the fixture is
two disjoint triangles with (a) gauge-flat U(1) phases on one (a genuinely
complex kernel: any real-truncated operator gets it wrong) and (b) a balanced
pair of timelike (negative real) edges on the other (an alternating-sign
kernel exercising d|w|/dw = -1).
"""

import math

import numpy as np
import pytest

import tessera
import cmath

cob = tessera.cobordism


def _two_component_host(phases=True, signed=True):
    """Two disjoint triangles; b_0 = 2. Component A optionally carries
    gauge-flat phases (zero triangle holonomy — the kernel survives, twisted);
    component B optionally carries two timelike edges (balanced signed
    triangle — the kernel survives, alternating sign)."""
    st = tessera.Spacetime.fromVertexTuples(2, [[0, 1, 2], [3, 4, 5]], 1.0, 0.0)
    st.materializeFacets()
    edges = st.getEdgeList().toVector()
    by_pair = {}
    for e in edges:
        a, b = e.getSource().getId(), e.getTarget().getId()
        by_pair[(min(a, b), max(a, b))] = e
    for i, e in enumerate(edges):
        e.setLength(cmath.sqrt(complex(1.0 + 0.17 * (i % 4))))
    if signed:
        by_pair[(3, 4)].setLength(cmath.sqrt(complex(-1.3)))
        by_pair[(3, 5)].setLength(cmath.sqrt(complex(-1.1)))
        by_pair[(4, 5)].setLength(cmath.sqrt(complex(1.2)))
    if phases:
        # holonomy 0.7 + 0.3 - 1.0 = 0 around (0,1,2): gauge-flat, kernel kept
        ph = {(0, 1): 0.7, (1, 2): 0.3, (0, 2): 1.0}
        for (a, b), val in ph.items():
            e = by_pair[(a, b)]
            sign = 1.0 if e.getSource().getId() < e.getTarget().getId() else -1.0
            e.setPhase(sign * val)
    return st, by_pair


def _cc_edges(st):
    return [tuple(sorted(t))
            for t in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(1)]


def _fd_gradient(st, holes, target, by_pair, h=1e-6):
    fd = []
    for pair in _cc_edges(st):
        e = by_pair[pair]
        w0 = e.getLength()**2
        e.setLength(cmath.sqrt(complex(complex(w0.real + h, 0.0))))
        rp = cob.EigenstateSynthesis(st, 0).residualForPeriods(holes, target)
        e.setLength(cmath.sqrt(complex(complex(w0.real - h, 0.0))))
        rm = cob.EigenstateSynthesis(st, 0).residualForPeriods(holes, target)
        e.setLength(cmath.sqrt(complex(w0)))
        fd.append((rp - rm) / (2 * h))
    return np.asarray(fd)


# Three cross-component holes against a 2-dim carried space: overdetermined,
# so a generic target is non-realizable (r_U > 0) — the gradient regime.
_HOLES = [[0, 3], [1, 4], [2, 5]]
_TARGET = [complex(1.0, 0.0), complex(-0.5, 0.3), complex(0.2, -0.8)]


class TestK0Gradient:
    def test_k0_gradient_matches_fd_complex_twisted_and_signed(self):
        # Full regime: complex (phased) kernel on A, alternating-sign kernel on
        # B (two timelike edges — d|w|/dw = -1 genuinely exercised).
        st, by_pair = _two_component_host(phases=True, signed=True)
        es = cob.EigenstateSynthesis(st, 0)
        r0 = es.residualForPeriods(_HOLES, _TARGET)
        assert r0 > 1e-3, "fixture must be non-realizable (overdetermined)"
        g = np.asarray(es.residualForPeriodsGradient(_HOLES, _TARGET))
        fd = _fd_gradient(st, _HOLES, _TARGET, by_pair)
        assert np.max(np.abs(g - fd)) < 1e-6

    def test_k0_gradient_matches_fd_rank_deficient(self):
        # No phases, all-positive weights: the kernel is the two plain component
        # constants and the GLOBAL constant has zero period on every hole — the
        # period matrix is column-rank-deficient, the regime where the SVD
        # pseudo-inverse (and its constant-rank derivative) is load-bearing.
        st, by_pair = _two_component_host(phases=False, signed=False)
        es = cob.EigenstateSynthesis(st, 0)
        r0 = es.residualForPeriods(_HOLES, _TARGET)
        assert r0 > 1e-3
        g = np.asarray(es.residualForPeriodsGradient(_HOLES, _TARGET))
        fd = _fd_gradient(st, _HOLES, _TARGET, by_pair)
        assert np.max(np.abs(g - fd)) < 1e-6

    @pytest.mark.parametrize("phases,signed", [(True, True), (False, False),
                                               (True, False), (False, True)])
    def test_k0_euler_identity_plus_two_ru(self, phases, signed):
        # L_0(s l^2) = s L_0(l^2) for s > 0 (degree +1), so the exact k = 0
        # Euler identity is sum_e l^2_e dr_U/dl^2_e = +2 r_U (machine
        # precision; k >= 1's metric L_k is degree -1, giving -r_U there).
        st, by_pair = _two_component_host(phases=phases, signed=signed)
        es = cob.EigenstateSynthesis(st, 0)
        r0 = es.residualForPeriods(_HOLES, _TARGET)
        g = np.asarray(es.residualForPeriodsGradient(_HOLES, _TARGET))
        l2 = np.asarray([(by_pair[p].getLength()**2).real
                         for p in _cc_edges(st)])
        assert math.isclose(float(np.dot(l2, g)), 2.0 * r0,
                            rel_tol=1e-10, abs_tol=1e-12)

    def test_k0_realizable_target_sits_at_the_zero(self):
        # Two holes against the 2-dim carried space: realizable, r_U ~ 0 and
        # the gradient of a floor-zero smooth functional vanishes there.
        st, by_pair = _two_component_host(phases=True, signed=True)
        es = cob.EigenstateSynthesis(st, 0)
        holes = [[0, 3], [1, 4]]
        target = [complex(1.0, 0.0), complex(-0.5, 0.3)]
        assert es.residualForPeriods(holes, target) < 1e-20
        g = np.asarray(es.residualForPeriodsGradient(holes, target))
        assert np.max(np.abs(g)) < 1e-10


class TestOtherDegreesUnchanged:
    def test_k1_gradients_still_work(self):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _holed_surface import holed_surface
        st, es, holes, P = holed_surface(degree=1)
        target = [complex(z) for z in P[0]]
        g = np.asarray(es.residualForPeriodsGradient(holes, target))
        assert g.shape[0] > 0 and np.all(np.isfinite(g))
        g2 = np.asarray(es.periodGapForPeriodsGradient(holes, target))
        assert g2.shape[0] > 0 and np.all(np.isfinite(g2))

    def test_loop_cores_state_their_degree_1_contract(self):
        # The signed edge-loop machinery (r_U loop core, r_psi gap core) is
        # degree-1 by construction — a loop period reads a 1-cell cochain. On
        # any other degree the call is a layout mismatch, reported as the
        # structural contract (an API arity error, not a physics veto): the
        # degree-routing hole API (residualForPeriodsGradient) is the entry
        # point for every degree, including the complex k = 0 core above.
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _holed_surface import holed_surface
        st, _es1, holes, P = holed_surface(degree=1)
        es0 = cob.EigenstateSynthesis(st)  # default degree k = 0
        target = [complex(z) for z in P[0]]
        with pytest.raises(RuntimeError, match="degree"):
            es0.periodGapForPeriodsGradient(holes, target)
