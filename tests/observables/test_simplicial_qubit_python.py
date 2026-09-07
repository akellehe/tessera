# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The simplicial qubit (#955), docs/design/simplicial_qubit_spec.md: the
section-12 reference cases and assertions, the section-2 validations, the
section-5 flags and Delaunay pass, the section-13 degeneration behaviour, and
the section-14 API."""
import cmath
import math
import warnings

import numpy as np
import pytest

import tessera
from tessera import observables as obs

SimplicialQubit = obs.SimplicialQubit

# Spec section 12: (name, tau, expected Bloch vector).
REFERENCE = [
    ("square", 1j, (0.0, 1.0, 0.0)),
    ("rectangle r=2", 2j, (0.0, 4.0 / 5.0, -3.0 / 5.0)),
    ("shear s=0.3", 0.3 + 1j, (0.6 / 2.09, 2.0 / 2.09, -0.09 / 2.09)),
    ("hexagonal", cmath.exp(1j * math.pi / 3), (0.5, math.sqrt(3) / 2, 0.0)),
]


def flat(tau, nx=4, ny=4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimplicialQubit.flat_torus(tau, nx, ny)


def rebuilt(q, lengths=None, faces=None, cycle_A=None, cycle_B=None, edges=None, vertices=None,
            **kwargs):
    """The same qubit through the section-2 constructor, with substitutions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimplicialQubit(
            list(q.vertices()) if vertices is None else vertices,
            list(q.edges()) if edges is None else edges,
            list(q.faces()) if faces is None else faces,
            list(q.lengths()) if lengths is None else lengths,
            list(q.cycle_A()) if cycle_A is None else cycle_A,
            list(q.cycle_B()) if cycle_B is None else cycle_B,
            **kwargs)


# ---------------------------------------------------------------------------
# Section 12: reference test cases (exact on flat tori)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,tau,bloch", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_flat_torus_reference_table(name, tau, bloch):
    q = flat(tau)
    assert abs(q.tau() - tau) < 1e-10
    assert np.allclose(q.bloch(), bloch, atol=1e-10)
    assert abs(np.linalg.norm(q.bloch()) - 1.0) < 1e-12
    assert q.j_residual() < 1e-10
    assert not q.marking_swapped()
    P_A, P_B = q.periods()
    assert abs(P_B / P_A - tau) < 1e-10
    psi = q.state()
    expected = np.array([1.0, tau]) / math.sqrt(1 + abs(tau) ** 2)
    assert np.allclose(psi, expected)
    rho = q.density_matrix()
    r = q.bloch()
    sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    assert np.allclose(rho, 0.5 * (np.eye(2) + sum(ri * si for ri, si in zip(r, sigma))))
    assert np.allclose(rho, np.outer(psi, psi.conj()))


def test_dim_h_is_two_and_the_basis_is_orthonormal():
    for tau in (1j, 0.3 + 1.2j, cmath.exp(1j * math.pi / 3)):
        q = flat(tau)
        H = q.harmonic_basis()
        assert H.shape == (len(q.edges()), 2)
        assert np.allclose(H.T @ H, np.eye(2), atol=1e-12)
        J = q.complex_structure()
        assert J.shape == (2, 2)
        assert np.linalg.norm(J @ J + np.eye(2)) == pytest.approx(q.j_residual())
        assert q.holomorphic_form().shape == (len(q.edges()),)


def test_tau_is_invariant_under_refinement_at_fixed_geometry():
    tau = 0.3 + 1.2j
    for nx, ny in [(3, 3), (4, 6), (8, 8)]:
        q = flat(tau, nx, ny)
        assert abs(q.tau() - tau) < 1e-9, (nx, ny, q.tau())
        assert q.j_residual() < 1e-9


def test_tau_is_invariant_under_uniform_scaling():
    q = flat(0.2 + 0.9j)
    scaled = rebuilt(q, lengths=[2.7 * l for l in q.lengths()])
    assert abs(scaled.tau() - q.tau()) < 1e-10
    # The Whitney L2 pairings of 1-forms are dimensionless, so G and R are
    # scale-invariant up to the orthogonal freedom of the null-space basis:
    # compare their O(2) invariants.
    G, Gs = q.gram(), scaled.gram()
    assert np.linalg.det(Gs) == pytest.approx(np.linalg.det(G), rel=1e-10)
    assert np.trace(Gs) == pytest.approx(np.trace(G), rel=1e-10)
    assert abs(scaled.rotation_pairing()[0, 1]) == pytest.approx(abs(q.rotation_pairing()[0, 1]), rel=1e-10)
    assert scaled.j_residual() == pytest.approx(q.j_residual(), abs=1e-12)


def test_modular_transformations():
    tau = 0.35 + 1.1j
    q = flat(tau)
    A, B = list(q.cycle_A()), list(q.cycle_B())
    minus_A = [(e, -s) for (e, s) in A]
    # A' = B, B' = -A  ->  -1/tau
    q_s = rebuilt(q, cycle_A=B, cycle_B=minus_A)
    assert abs(q_s.tau() - (-1.0 / tau)) < 1e-9
    # B' = A + B  ->  tau + 1
    q_t = rebuilt(q, cycle_A=A, cycle_B=A + B)
    assert abs(q_t.tau() - (tau + 1.0)) < 1e-9


def _conformal(n, amplitude=0.3):
    """The square torus with lengths scaled by exp(phi) at each edge midpoint,
    phi = amplitude sin(2 pi x) cos(2 pi y): conformally flat, conformal
    structure still tau = i, but not flat, so J is only approximately a
    complex structure."""
    q = flat(1j, n, n)

    def reduce(delta):
        return -1 if delta == n - 1 else (1 if delta == -(n - 1) else delta)

    lengths = []
    for (u, v), length in zip(q.edges(), q.lengths()):
        iu, ju = divmod(u, n)
        iv, jv = divmod(v, n)
        di, dj = reduce(iv - iu), reduce(jv - ju)
        x, y = (iu + 0.5 * di) / n, (ju + 0.5 * dj) / n
        lengths.append(length * math.exp(amplitude * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)))
    return rebuilt(q, lengths=lengths)


def test_j_residual_decreases_monotonically_under_uniform_refinement():
    residuals, errors = [], []
    for n in (4, 8, 16):
        q = _conformal(n)
        residuals.append(q.j_residual())
        errors.append(abs(q.tau() - 1j))
    assert residuals[0] > 1e-6
    assert residuals[0] > residuals[1] > residuals[2]
    assert errors[0] > errors[1] > errors[2]


# ---------------------------------------------------------------------------
# Sections 3, 4, 7: incidence, per-face geometry, barycentric gradients
# ---------------------------------------------------------------------------

def test_incidence_matrices_and_per_face_geometry():
    q = flat(0.3 + 1j, 4, 5)
    nV, nE, nF = len(q.vertices()), len(q.edges()), len(q.faces())
    assert (nV, nE, nF) == (20, 60, 40) and nV - nE + nF == 0
    d0, d1 = q.d0(), q.d1()
    assert d0.shape == (nE, nV) and d1.shape == (nF, nE)
    assert np.all(d1 @ d0 == 0)
    assert np.all(np.abs(d1).sum(axis=1) == 3) and np.all(np.abs(d0).sum(axis=1) == 2)
    # Angles sum to pi; Heron areas agree with the container's own Simplex.area().
    assert np.allclose(q.angles().sum(axis=1), math.pi)
    by_set = {}
    for s in q.spacetime().getSimplices():
        ids = tuple(sorted(v.getId() for v in s.getVertices()))
        if len(ids) == 3:
            by_set[ids] = s.area().real
    for face, area in zip(q.faces(), q.areas()):
        assert area == pytest.approx(by_set[tuple(sorted(face))], rel=1e-12)
    # Local layout: p_i = 0, p_j = (c, 0), p_k above the edge.
    layout = q.layout()
    assert np.allclose(layout[:, 0:2], 0.0) and np.allclose(layout[:, 3], 0.0) and np.all(layout[:, 5] > 0)
    # grad_lambda_i + grad_lambda_j + grad_lambda_k == 0.
    g = q.barycentric_gradients()
    assert np.allclose(g[:, 0:2] + g[:, 2:4] + g[:, 4:6], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 5: cotangent weights, flags, and the intrinsic Delaunay pass
# ---------------------------------------------------------------------------

def test_cotangent_weights_and_delaunay_flags():
    # Square torus: axis edges see 45 + 45 degrees (w = 1), diagonals 90 + 90 (w = 0).
    q = flat(1j)
    assert set(np.round(q.weights(), 12)) == {0.0, 1.0}
    assert q.non_delaunay_edges() == [] and q.negative_weight_edges() == []
    # The hexagonal lattice cut along the long diagonal: 30-30-120 triangles,
    # the diagonal opposite 120 + 120 > 180 degrees, negative weights, flagged.
    with pytest.warns(UserWarning, match="Delaunay"):
        h = SimplicialQubit.flat_torus(cmath.exp(1j * math.pi / 3), 4, 4)
    assert len(h.non_delaunay_edges()) == 16 and len(h.negative_weight_edges()) == 16
    assert any("Delaunay" in w for w in h.warnings())


def test_intrinsic_delaunay_pass_repairs_the_hexagonal_torus():
    tau = cmath.exp(1j * math.pi / 3)
    h = flat(tau)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = h.intrinsic_delaunay()
    assert d.delaunay_flip_count() == 16
    assert d.non_delaunay_edges() == [] and d.negative_weight_edges() == []
    assert np.allclose(d.lengths(), 0.25)            # every triangle equilateral
    assert np.allclose(d.angles(), math.pi / 3)
    assert abs(d.tau() - tau) < 1e-10                # the intrinsic geometry is unchanged
    assert abs(h.tau() - tau) < 1e-10
    assert len(d.cycle_A()) >= len(h.cycle_A()) and len(d.edges()) == len(h.edges())
    # Already Delaunay: nothing to do.
    s = flat(1j)
    assert s.intrinsic_delaunay().delaunay_flip_count() == 0


# ---------------------------------------------------------------------------
# Section 2: validation on load
# ---------------------------------------------------------------------------

def test_validation_on_load():
    q = flat(1j)
    faces = list(q.faces())
    with pytest.raises(ValueError, match="belongs to 1 faces"):
        rebuilt(q, faces=faces[:-1])
    with pytest.raises(ValueError, match="inconsistent"):
        rebuilt(q, faces=[faces[0][::-1]] + faces[1:])
    with pytest.raises(ValueError, match="triangle inequality"):
        rebuilt(q, lengths=[10.0] + list(q.lengths())[1:])
    with pytest.raises(ValueError, match="real and positive"):
        rebuilt(q, lengths=[-1.0] + list(q.lengths())[1:])
    with pytest.raises(ValueError, match="not closed"):
        rebuilt(q, cycle_A=list(q.cycle_A())[:-1])
    with pytest.raises(ValueError, match="not independent"):
        rebuilt(q, cycle_B=list(q.cycle_A()))
    with pytest.raises(ValueError, match="0 .. nV-1"):
        rebuilt(q, vertices=list(q.vertices())[1:] + [99])
    with pytest.raises(ValueError, match="not in E"):
        rebuilt(q, edges=list(q.edges())[1:], lengths=list(q.lengths())[1:])
    # The 2-sphere: every edge in two faces, chi = 2.
    sphere = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edges = sorted({tuple(sorted((f[a], f[(a + 1) % 3]))) for f in sphere for a in range(3)})
    with pytest.raises(ValueError, match="Euler characteristic"):
        SimplicialQubit([0, 1, 2, 3], edges, sphere, [1.0] * 6, [(0, 1)], [(1, 1)])
    with pytest.raises(ValueError):
        SimplicialQubit.flat_torus(-1j, 4, 4)
    with pytest.raises(ValueError):
        SimplicialQubit.flat_torus(1j, 2, 4)


# ---------------------------------------------------------------------------
# Section 13: degeneration behaviour
# ---------------------------------------------------------------------------

def test_pinching_cycle_warns_and_moves_the_state_to_a_pole():
    q = flat(1j)
    thin = flat(0.05j)
    with pytest.warns(UserWarning, match="near-degenerate"):
        p = SimplicialQubit(list(thin.vertices()), list(thin.edges()), list(thin.faces()),
                            list(thin.lengths()), list(thin.cycle_A()), list(thin.cycle_B()),
                            degeneracy_threshold=10.0)
    assert p.near_degenerate() and p.condition_m1() > 10.0
    assert abs(p.tau() - 0.05j) < 1e-9
    assert p.bloch()[2] > 0.99                        # towards |0>
    assert math.isfinite(obs.fubini_study_distance(p, q))
    assert obs.weil_petersson_distance(p, q) > obs.weil_petersson_distance(flat(0.5j), q)
    assert abs(np.linalg.norm(p.bloch()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Section 11: the two metrics
# ---------------------------------------------------------------------------

def test_distances():
    a, b = flat(1j), flat(2j)
    assert obs.fubini_study_distance(a, a) == 0.0
    assert obs.weil_petersson_distance(a, a) == 0.0
    assert obs.weil_petersson_distance(a, b) == pytest.approx(math.log(2))
    assert obs.fubini_study_distance(a, b) == obs.fubini_study_distance(b, a)
    assert obs.fubini_study_distance(a, b) == pytest.approx(math.acos(3 / math.sqrt(10)))


# ---------------------------------------------------------------------------
# The Spacetime constructor
# ---------------------------------------------------------------------------

def test_reading_the_spacetime_directly_gives_the_same_state():
    tau = 0.3 + 1.2j
    q = flat(tau)
    st = q.spacetime()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B()))
        flipped = SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B()), reversed=True)
    assert list(direct.edges()) == list(q.edges())
    assert abs(direct.tau() - tau) < 1e-10 and abs(flipped.tau() - tau) < 1e-10
    # Exactly one of the two orientations agrees with the marking's A.B = +1;
    # the other is caught by the Im tau < 0 rule of section 9.
    assert sum(any("conjugate" in w for w in x.warnings()) for x in (direct, flipped)) == 1
    # A complex length is read (section 16, test_simplicial_qubit_complex_python.py);
    # a real non-positive one is still refused by section 2.
    edges = list(st.getEdgeList().toVector())
    original = edges[0].getLength()
    edges[0].setLength(complex(original) * (1 + 0.1j))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B()))
    assert not off.on_real_locus() and direct.on_real_locus()
    edges[0].setLength(-abs(complex(original)))
    with pytest.raises(ValueError, match="real and positive"):
        SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B()))
    edges[0].setLength(original)


# ---------------------------------------------------------------------------
# Qubit cobordism spec D2: the analytic derivative of tau and the
# intersection number
# ---------------------------------------------------------------------------

def _squared_lengths(q):
    return np.asarray(q.lengths(), dtype=complex) ** 2


def _euler_defect(q):
    """|sum_e z_e dtau/dz_e| against the scale of the terms: tau is invariant
    under a common scale of the lengths, so the sum is an identity in the
    derivative (zero to rounding), not a tolerance."""
    z = _squared_lengths(q)
    d = np.asarray(q.tau_derivative())
    return abs(np.sum(z * d)) / (np.abs(z).max() * np.abs(d).max())


@pytest.mark.parametrize("tau", [1j, 0.3 + 1.1j, -0.2 + 0.8j, 2j])
def test_tau_derivative_is_scale_free_on_flat_tori(tau):
    # Both charts of the derivative: |tau| > 1 (sigma = 1/tau) and |tau| < 1.
    assert _euler_defect(flat(tau, 3, 3)) < 1e-13
    assert _euler_defect(flat(tau, 4, 4)) < 1e-13


def test_tau_derivative_is_exact_on_the_flat_family():
    """tau(flat_torus(tau0)) == tau0, so the chain rule through the lattice's
    squared lengths must give dtau/dRe(tau0) = 1 and dtau/dIm(tau0) = i. The
    squared lengths are quadratic in (Re tau0, Im tau0), so their central
    differences are exact to rounding; only the analytic derivative of tau
    is under test."""
    tau0, h = 0.3 + 1.1j, 1e-4
    for n in (3, 4):
        d = np.asarray(flat(tau0, n, n).tau_derivative())
        dz_dx = (_squared_lengths(flat(tau0 + h, n, n)) - _squared_lengths(flat(tau0 - h, n, n))) / (2 * h)
        dz_dy = (_squared_lengths(flat(tau0 + 1j * h, n, n)) - _squared_lengths(flat(tau0 - 1j * h, n, n))) / (2 * h)
        assert abs(np.sum(d * dz_dx) - 1.0) < 1e-9
        assert abs(np.sum(d * dz_dy) - 1j) < 1e-9


def _finite_difference(q, e, h=1e-6):
    """The central difference of tau in the squared length of edge e (an
    approximation of order h^2, about 1e-9 here), the section-2 constructor
    rebuilt at each point."""
    values = []
    for sign in (+1, -1):
        lengths = list(q.lengths())
        lengths[e] = complex(np.sqrt(lengths[e] ** 2 + sign * h))
        values.append(rebuilt(q, lengths=lengths).tau())
    return (values[0] - values[1]) / (2 * h)


def test_tau_derivative_matches_finite_differences_off_the_flat_locus():
    """On a conformally deformed (non-flat) torus every stage of the
    construction moves with the lengths -- the harmonic frames, the areas,
    the layouts, the cotangent weights -- and the analytic derivative agrees
    with the finite difference to the difference's own accuracy (~1e-8)."""
    q = _conformal(4, amplitude=0.2)
    assert _euler_defect(q) < 1e-13
    d = np.asarray(q.tau_derivative())
    rng = np.random.default_rng(0)
    for e in rng.choice(len(q.edges()), 5, replace=False):
        fd = _finite_difference(q, int(e))
        assert abs(fd - d[e]) <= 1e-7 * max(abs(d[e]), 1.0)


def test_tau_derivative_is_holomorphic_off_the_real_locus():
    """Section 16: with complex lengths the derivative is the derivative of
    the continued branch, holomorphic in z, so a step i*h moves tau by i
    times the step h does; and it stays scale-free."""
    q = flat(0.3 + 1.1j, 3, 3)
    rng = np.random.default_rng(2)
    lengths = [complex(l) * (1.0 + 0.06 * rng.uniform(-1, 1) + 0.08j * rng.uniform(-1, 1)) for l in q.lengths()]
    qz = rebuilt(q, lengths=lengths)
    assert not qz.on_real_locus()
    assert _euler_defect(qz) < 1e-13
    d = np.asarray(qz.tau_derivative())
    for e in (1, 7, 20):
        real_step = _finite_difference(qz, e)
        values = []
        for sign in (+1, -1):
            perturbed = list(qz.lengths())
            perturbed[e] = complex(np.sqrt(perturbed[e] ** 2 + sign * 1e-6j))
            values.append(rebuilt(qz, lengths=perturbed).tau())
        imaginary_step = (values[0] - values[1]) / (2e-6)
        assert abs(real_step - d[e]) <= 1e-7 * max(abs(d[e]), 1.0)
        assert abs(imaginary_step - 1j * d[e]) <= 1e-7 * max(abs(d[e]), 1.0)


def test_tau_derivative_is_gauge_invariant():
    """A pure gauge on the links (section 16) leaves tau invariant, and so
    its derivative: the twisted frames, their duals and the transported
    periods all carry the same base-point factor, which cancels."""
    q = _conformal(3, amplitude=0.2)
    plain = np.asarray(q.tau_derivative())
    spacetime = q.spacetime()
    rng = np.random.default_rng(5)
    gauge = {int(v.getId()): rng.uniform(-np.pi, np.pi) for v in spacetime.getVertexList().toVector()}
    for edge in spacetime.getEdgeList().toVector():
        edge.setPhase(gauge[int(edge.getTarget().getId())] - gauge[int(edge.getSource().getId())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gauged = SimplicialQubit(spacetime, list(q.cycle_A()), list(q.cycle_B()), q.intersection_number() < 0)
    assert not gauged.trivial_connection()
    assert abs(gauged.tau() - q.tau()) < 1e-12
    assert np.abs(np.asarray(gauged.tau_derivative()) - plain).max() < 1e-11 * np.abs(plain).max()


def test_intersection_number_fixes_the_orientation():
    """A . B = +1 on the flat torus (spec section 12 builds it so); the
    Spacetime, which stores no orientation, reads +1 in exactly one of its
    two orientations, and swapping the marking flips the sign."""
    q = flat(0.3 + 1.1j, 3, 3)
    assert abs(q.intersection_number() - 1.0) < 1e-12
    spacetime = q.spacetime()
    A, B = list(q.cycle_A()), list(q.cycle_B())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signs = [SimplicialQubit(spacetime, A, B, reversed).intersection_number() for reversed in (False, True)]
        swapped = SimplicialQubit(spacetime, B, A, False).intersection_number()
    assert sorted(round(s) for s in signs) == [-1, 1]
    assert abs(abs(swapped) - 1.0) < 1e-12 and round(swapped) == -round(signs[0])
