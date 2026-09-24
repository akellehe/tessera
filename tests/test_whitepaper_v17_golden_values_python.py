# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The numbers the whitepaper "Recursive Spectral Fibers" (v17) states, held
as golden values against the library.

Every test cites the line of
``reports/theory-docs/whitepaper/recursive_spectral_fibers_whitepaper_v17.md``
in the notes repository that states the value, written ``WP v17 line N``. The
values are exact statements of the paper (spectra, identities, closed forms);
each is recomputed here from the library object that implements it, and where
the paper's statement is an identity (the Feshbach factorization, the Drazin
inverse, the Ward identity) the identity is checked against an independent
dense computation rather than against the object's own certificate alone.

Terms used below:

* ``mu`` is the monopole number of the tetrahedron's connection: the total
  outward face flux divided by 2 pi (WP v17 line 504).
* The *T-average* of an edge operator X is (1/|T|) sum_g D_1(g)^{-1} X D_1(g)
  over the twelve rotations T = A_4 of the tetrahedron, with D_1(g) the
  projective action on edge cochains (WP v17 line 506).
* A *band* is a set of equal eigenvalues; the *first-order split* of a band
  under a perturbation dL is the spectrum of dL compressed to the band by its
  spectral projector.
"""
import cmath
import itertools
import math

import numpy as np
import pytest

import tessera as T
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

SQRT3 = math.sqrt(3.0)


def _sorted_real(matrix):
    return np.sort(np.linalg.eigvals(np.asarray(matrix)).real)


def _regular_tetrahedron(squared_length=8.0):
    """A single regular tetrahedron with every squared edge length equal."""
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(squared_length)))
    return spacetime


def _whitney_hodge(spacetime):
    return cob.HodgeLaplacian(spacetime,
                              cob.HodgeLaplacian.defaultWeightConvention(),
                              cob.HodgeMetricSource.WhitneyPencil)


def _edge_records(spacetime):
    return [(int(e.getSource().getId()), int(e.getTarget().getId()))
            for e in spacetime.getEdgeList().toVector()]


def _band_split(operator, derivative, value, tolerance=1e-8):
    """The first-order split of the band of ``operator`` at ``value`` under
    ``derivative``: the eigenvalues of the derivative compressed to the band
    by the band's right eigenvectors and the matching rows of their inverse."""
    values, vectors = np.linalg.eig(operator)
    index = [k for k in range(len(values)) if abs(values[k] - value) < tolerance]
    left = np.linalg.inv(vectors)[index, :]
    compressed = left @ derivative @ vectors[:, index]
    return np.sort(np.linalg.eigvals(compressed).real), np.trace(compressed)


def _commuting_rotation_pair(group):
    """Two distinct commuting order-two rotations (double transpositions)."""
    involutions = [list(g) for g in group
                   if list(g) != [0, 1, 2, 3]
                   and all(g[g[i]] == i for i in range(4))]
    return involutions[0], involutions[1]


def _group_commutator(representation):
    a, b = representation
    return a @ b @ np.linalg.inv(a) @ np.linalg.inv(b)


# ------------------------------------------------ Section 3: the tetrahedron


def test_whitney_edge_spectrum_of_the_regular_tetrahedron():
    """WP v17 line 145: with the Whitney weights and trivial connection the
    degree-one Hodge Laplacian of the regular tetrahedron of edge length a has
    the exact triplet at 40/a^2 and the coexact triplet at 80/a^2; at
    a^2 = 8 these are 5 and 10."""
    hodge = _whitney_hodge(_regular_tetrahedron(8.0))
    values = _sorted_real(np.asarray(hodge.laplacian(1)).reshape(6, 6))
    np.testing.assert_allclose(values, [5.0] * 3 + [10.0] * 3, atol=1e-10)


@pytest.mark.parametrize("mu,expected", [
    (0, [0.0, 4.0, 4.0, 4.0]),
    (2, [2.0, 2.0, 2.0, 6.0]),
    (1, [3.0 - SQRT3] * 2 + [3.0 + SQRT3] * 2),
    (-1, [3.0 - SQRT3] * 2 + [3.0 + SQRT3] * 2),
])
def test_monopole_vertex_spectra(mu, expected):
    """WP v17 line 504: the combinatorial twisted vertex Laplacian of the
    tetrahedron has spectrum {0, 4^(3)} at mu = 0, {2^(3), 6} at mu = 2, and
    two doublets {3 - sqrt 3 ^(2), 3 + sqrt 3 ^(2)} at mu = +-1."""
    support = obs.MonopoleSupport.tetrahedron(mu)
    np.testing.assert_allclose(_sorted_real(support.vertexLaplacian()),
                               expected, atol=1e-10)


@pytest.mark.parametrize("mu,phase", [(0, 1.0), (2, 1.0), (1, -1.0),
                                      (-1, -1.0)])
@pytest.mark.parametrize("degree", [0, 1])
def test_the_two_c2_rotations_commute_at_even_and_anticommute_at_odd_flux(
        mu, phase, degree):
    """WP v17 line 504: two commuting C_2 rotations a, b commute as operators
    at mu = 0 and mu = 2, and anticommute at mu = +-1,
    D(a) D(b) D(a)^{-1} D(b)^{-1} = -1 (the quaternion relation of 2T). The
    commutator is formed here from the representation matrices on vertex
    (degree 0) and edge (degree 1) cochains, not read from the cocycle."""
    support = obs.MonopoleSupport.tetrahedron(mu)
    a, b = _commuting_rotation_pair(obs.MonopoleSupport.tetrahedralRotations())
    assert [a[b[i]] for i in range(4)] == [b[a[i]] for i in range(4)]
    represent = (support.vertexRepresentation if degree == 0
                 else support.edgeRepresentation)
    commutator = _group_commutator([np.asarray(represent(a)),
                                    np.asarray(represent(b))])
    size = commutator.shape[0]
    np.testing.assert_allclose(commutator, phase * np.eye(size), atol=1e-12)


def test_the_t_averaged_twisted_edge_laplacian():
    """WP v17 line 506: under the unit monopole the T-averaged twisted edge
    Laplacian has spectrum {4 - 2/sqrt 3 ^(2), 4^(2), 4 + 2/sqrt 3 ^(2)}, the
    genuine j = 1/2 doublet (the coexact one) at 4; the same construction
    returns 4 I at zero flux."""
    group = obs.MonopoleSupport.tetrahedralRotations()
    unit = obs.MonopoleSupport.tetrahedron(1)
    averaged = unit.rotationAveragedEdgeOperator(unit.edgeLaplacian(), group)
    np.testing.assert_allclose(
        _sorted_real(averaged),
        [4 - 2 / SQRT3] * 2 + [4.0] * 2 + [4 + 2 / SQRT3] * 2, atol=1e-12)
    read = unit.spinRead(group)
    doublet = read.bands[read.doublet_index]
    assert read.half_integer_doublet
    assert doublet.dimension == 2 and doublet.spinor_doublet and doublet.coexact
    assert doublet.eigenvalue == pytest.approx(4.0, abs=1e-12)
    flat = obs.MonopoleSupport.tetrahedron(0)
    np.testing.assert_allclose(
        flat.rotationAveragedEdgeOperator(flat.edgeLaplacian(), group),
        4.0 * np.eye(6), atol=1e-12)


# ------------------------------------------- Section 7: the Whitney forces


@pytest.mark.parametrize("band,expected", [
    (5.0, [-1.0 / 8.0, 0.0, 1.0 / 16.0]),
    (10.0, [-1.0 / 16.0, -1.0 / 32.0, 1.0 / 32.0]),
])
def test_single_edge_splits_of_the_regular_tetrahedron(band, expected):
    """WP v17 line 263: on the regular tetrahedron of squared edge length 8,
    under the Whitney metric, a single-edge perturbation splits the exact band
    (lambda = 5) with derivatives {-lambda/8, 0, +lambda/16} and the coexact
    band (lambda = 10) with {-lambda/16, -lambda/32, +lambda/32}, and the band
    trace is the uniform dilation -lambda/16 per edge. The derivative is
    dL_1/dz_e, the exact analytic gradient in one edge's squared length."""
    spacetime = _regular_tetrahedron(8.0)
    hodge = _whitney_hodge(spacetime)
    operator = np.asarray(hodge.laplacian(1)).reshape(6, 6)
    for a, b in _edge_records(spacetime):
        derivative = np.asarray(hodge.laplacianGradient(1, a, b)).reshape(6, 6)
        split, trace = _band_split(operator, derivative, band)
        np.testing.assert_allclose(split / band, expected, atol=1e-12)
        assert trace.real / band == pytest.approx(-1.0 / 16.0, abs=1e-12)
        assert abs(trace.imag) < 1e-12


def test_the_whitney_operator_is_homogeneous_of_degree_minus_one():
    """WP v17 line 263: the Whitney Hodge Laplacian is homogeneous of degree
    -1 in the squared lengths, so every eigenvalue obeys
    sum_e z_e d(lambda)/d(z_e) = -lambda. Checked on a tetrahedron with
    generic complex squared lengths, where the six eigenvalues are simple and
    each derivative is w^T dL v / w^T v with w, v its left and right
    eigenvectors."""
    spacetime = _regular_tetrahedron(8.0)
    rng = np.random.default_rng(1)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(8.0 + rng.normal() + 0.3j * rng.normal()))
    hodge = _whitney_hodge(spacetime)
    operator = np.asarray(hodge.laplacian(1)).reshape(6, 6)
    values, right = np.linalg.eig(operator)
    left = np.linalg.inv(right)
    euler = np.zeros(6, dtype=complex)
    for edge, (a, b) in zip(spacetime.getEdgeList().toVector(),
                            _edge_records(spacetime)):
        z = complex(edge.getLength()) ** 2
        derivative = np.asarray(hodge.laplacianGradient(1, a, b)).reshape(6, 6)
        euler += z * np.diag(left @ derivative @ right)
    np.testing.assert_allclose(euler, -values, rtol=1e-12, atol=1e-12)


# ----------------------------------------------- Section 3 and 7: holonomy


def _cell_host(links):
    """The three-sheeted host of `baryon_poles` with every sheet the regular
    tetrahedron of squared length 8 carrying the given six links."""
    return bp.build_host(cell={"squared_lengths": [8.0] * 6,
                               "links": list(links)})


def _phase_stiffness(spacetime, beta, holonomy):
    hessian = np.asarray(cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, beta, holonomy=holonomy)).holonomy_hessian())
    return np.sort(np.linalg.eigvalsh(-hessian.reshape(18, 18).real))


@pytest.mark.parametrize("holonomy", ["wilson", "villain"])
@pytest.mark.parametrize("beta", [0.7, 2.0])
def test_the_bare_connection_stiffness_is_4_beta_at_trivial_holonomy(
        holonomy, beta):
    """WP v17 lines 173 and 291: at trivial holonomy the quadratic expansion
    of the holonomy term is (1/2) beta dphi^T L_1^up dphi for both the Villain
    and the Wilson form, which is 4 beta on the coexact block of the
    tetrahedron and zero on the exact (pure-gauge) block. On three sheets
    there are nine of each."""
    values = _phase_stiffness(_cell_host([1.0] * 6), beta, holonomy)
    np.testing.assert_allclose(values[:9], 0.0, atol=1e-10)
    np.testing.assert_allclose(values[9:], 4.0 * beta, rtol=1e-10)


def _villain_quarter_turn_curvature(beta, holonomy=1j, terms=200):
    """beta_V (<m^2>_F - <m>_F^2) with the weights e^{-m^2/(2 beta)} F^m / W,
    summed directly, and beta_V = beta / <m^2>_beta."""
    m = np.arange(-terms, terms + 1, dtype=float)
    base = np.exp(-m ** 2 / (2.0 * beta))
    second_moment = np.sum(m ** 2 * base) / np.sum(base)
    weights = base * holonomy ** m
    w = np.sum(weights)
    mean = np.sum(m * weights) / w
    variance = np.sum(m ** 2 * weights) / w - mean ** 2
    return (beta / second_moment) * variance


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_the_villain_stiffness_at_a_quarter_turn(beta):
    """WP v17 line 173: at a quarter-turn face holonomy F = +-i the Wilson
    curvature beta cos(theta) vanishes, while the Villain curvature
    beta_V (<m^2>_F - <m>_F^2) does not. On the unit-monopole host every
    face carries F = +-i: the Wilson phase stiffness is zero everywhere, and
    the Villain one is four times that curvature on the nine coexact
    directions."""
    host = bp.build_host()
    wilson = _phase_stiffness(host, beta, "wilson")
    np.testing.assert_allclose(wilson, 0.0, atol=1e-12)
    curvature = _villain_quarter_turn_curvature(beta)
    assert abs(curvature.imag) < 1e-12 and curvature.real > 0.4 * beta
    villain = _phase_stiffness(host, beta, "villain")
    np.testing.assert_allclose(villain[:9], 0.0, atol=1e-10)
    np.testing.assert_allclose(villain[9:], 4.0 * curvature.real, rtol=1e-10)


def test_the_matched_villain_weight_tends_to_one():
    """WP v17 lines 171 and 173: beta_V = beta / <m^2>_beta, and by Poisson
    summation beta_V -> 1 as beta -> infinity."""
    for beta in (0.5, 2.0):
        character = cob.VillainCharacter(beta)
        m = np.arange(-400, 401, dtype=float)
        weights = np.exp(-m ** 2 / (2.0 * beta))
        assert character.matched_weight == pytest.approx(
            beta / (np.sum(m ** 2 * weights) / np.sum(weights)), rel=1e-12)
    assert cob.VillainCharacter(60.0).matched_weight == pytest.approx(
        1.0, abs=1e-12)


def test_the_ward_identity_on_pure_gauge_directions():
    """WP v17 line 289: at omega = 0, D - Pi(0) vanishes identically on a
    pure-gauge direction, while the paramagnetic term alone does not. Read on
    the unit-monopole host with its three lowest modes of h_1 occupied, D by a
    Cauchy rule along each pure-gauge direction and Pi(0) by
    `DressedFluctuation.paramagnetic`."""
    spacetime = bp.build_host()
    config = bp.default_config([0.5], [1.0])
    bare = cob.JointAction(spacetime, bp.action_declaration(spacetime, 0.5, 1.0))
    declaration = bp.action_declaration(spacetime, 0.5, 1.0)
    declaration.covariance = bare.occupation_projector(3)
    carrier = bp.matrix(cob.JointAction(spacetime, declaration)
                        .carrier_operator())
    couplings = bp.fluctuation_couplings(spacetime, True)
    ward = bp.ward_read(spacetime, carrier, couplings,
                        bp.gauge_directions(spacetime, True), config)
    assert ward["directions"] == 9
    assert ward["residual"] < 1e-12
    assert ward["paramagnetic_alone"] > 1e-2


# ------------------------------------------ Section 4: Feshbach and Drazin


def test_the_feshbach_determinant_factorization():
    """WP v17 line 193: for lambda outside spec A_II,
    det(A - lambda I) = det(A_II - lambda I) det F_B(lambda), with
    F_B(lambda) = A_BB - lambda I - A_BI (A_II - lambda I)^{-1} A_IB; and
    lambda in spec A exactly when 0 in spec F_B(lambda)."""
    rng = np.random.default_rng(11)
    n, interface = 7, [0, 1, 2]
    interior = [3, 4, 5, 6]
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = x + x.T
    identity = np.eye(n, dtype=complex)
    lam = 0.3 + 0.2j
    read = ch.PencilSchur.feshbach(A, identity, lam, interface)
    P = A - lam * identity
    F = P[np.ix_(interface, interface)] - P[np.ix_(interface, interior)] @ \
        np.linalg.solve(P[np.ix_(interior, interior)],
                        P[np.ix_(interior, interface)])
    np.testing.assert_allclose(np.asarray(read.response), F, atol=1e-12)
    det_p = np.linalg.det(P)
    assert abs(det_p - np.linalg.det(P[np.ix_(interior, interior)])
               * np.linalg.det(F)) < 1e-12 * abs(det_p)
    assert abs(read.pencilDeterminant - read.interiorDeterminant
               * read.responseDeterminant) < 1e-12 * abs(det_p)
    assert read.determinantResidual < 1e-12
    eigenvalue = np.linalg.eigvals(A)[0]
    at_pole = ch.PencilSchur.feshbach(A, identity, eigenvalue, interface)
    singular = np.linalg.svd(np.asarray(at_pole.response), compute_uv=False)
    assert singular[-1] < 1e-10 * singular[0]


def _index_one_singular(seed=4, n=6):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    J = np.diag([0, 0, 1.5, -2 + 0.5j, 3, 0.7j]).astype(complex)
    return S @ J @ np.linalg.inv(S), rng


def _drazin(A, radius=1e-10):
    read = ch.PencilSchur.feshbach(A, np.zeros_like(A), 0j, [], 1e-12, radius)
    assert read.interiorSingular
    return np.asarray(read.interiorInverse), np.asarray(read.nullProjector), \
        read


def test_the_drazin_inverse_identities():
    """WP v17 line 190: with Pi_0 the Riesz projector onto the generalized
    null space, A^D = (A + Pi_0)^{-1} (I - Pi_0), with range and null
    projectors I - Pi_0 and Pi_0. A^D satisfies A^D A A^D = A^D,
    A A^D = A^D A = I - Pi_0, and, at index one, A A^D A = A."""
    A, _ = _index_one_singular()
    D, null, read = _drazin(A)
    n = A.shape[0]
    identity = np.eye(n)
    scale = np.linalg.norm(D)
    assert np.linalg.norm(D @ A @ D - D) < 1e-12 * scale
    assert np.linalg.norm(A @ D - D @ A) < 1e-12 * scale
    assert np.linalg.norm(A @ D - (identity - null)) < 1e-12
    assert np.linalg.norm(A @ D @ A - A) < 1e-12 * np.linalg.norm(A)
    np.testing.assert_allclose(D, np.linalg.solve(A + null, identity - null),
                               atol=1e-12 * scale)
    np.testing.assert_allclose(np.asarray(read.rangeProjector),
                               identity - null, atol=1e-12)
    assert np.trace(null).real == pytest.approx(2.0, abs=1e-10)
    assert read.interiorRank == n - 2


def test_the_drazin_inverse_is_similarity_covariant():
    """WP v17 line 190: the Drazin inverse is covariant under every
    similarity, (S^{-1} A S)^D = S^{-1} A^D S."""
    A, rng = _index_one_singular()
    D, _, _ = _drazin(A)
    S = rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)
    moved, _, _ = _drazin(np.linalg.solve(S, A @ S))
    np.testing.assert_allclose(moved, np.linalg.solve(S, D @ S),
                               atol=1e-10 * np.linalg.norm(D))


def test_the_disc_window_is_the_real_window_in_the_hermitian_regime():
    """WP v17 line 193: the surrogates retain the modes inside the declared
    disc |theta - c| <= rho; in the Hermitian regime the spectrum is real and
    the disc meets the real axis in [c - rho, c + rho], so the retained
    fixed-interface modes are exactly the interior eigenvalues in that
    interval, and every reported eigenvalue of the window is real and lies in
    it."""
    rng = np.random.default_rng(3)
    nb, ni = 3, 7
    n = nb + ni
    x = rng.normal(size=(n, n))
    A = x + x.T
    q, _ = np.linalg.qr(rng.normal(size=(ni, ni)))
    spectrum = np.array([-3.0, -1.2, 0.4, 0.9, 1.4, 2.6, 5.0])
    A[nb:, nb:] = q @ np.diag(spectrum) @ q.T
    A[:nb, nb:] *= 0.05
    A[nb:, :nb] = A[:nb, nb:].T
    centre, radius = 1.0, 0.5
    read = ch.PencilSchur.craigBamptonSurrogate(
        A.astype(complex), np.eye(n, dtype=complex), list(range(nb)),
        complex(centre), radius, radius)
    inside = spectrum[np.abs(spectrum - centre) <= radius]
    assert read.retainedModes == len(inside) == 2
    interior = np.asarray(read.interiorEigenvalues)
    np.testing.assert_allclose(np.sort(interior.real), np.sort(spectrum),
                               atol=1e-12)
    assert np.max(np.abs(interior.imag)) < 1e-12
    window = [complex(read.eigenvalues[k]) for k in read.windowIndices]
    assert window
    for value in window:
        assert abs(value.imag) < 1e-10
        assert centre - radius <= value.real <= centre + radius


# ---------------------------------------------- Section 15: the grown cell


def _generic_tetrahedron(links):
    """The squared lengths of a generic Euclidean tetrahedron in the
    ascending edge order, and the given links keyed by ascending edge.
    `recursion.level_rule_shift` reads such a tetrahedron as a grown cell: its
    vertex fibers are the exact chains of the twisted coboundary, their dual
    partners those of the inverse connection, and its transports the links."""
    rng = np.random.default_rng(4)
    points = rng.normal(size=(4, 3))
    pairs = list(itertools.combinations(range(4), 2))
    s = [complex(np.sum((points[j] - points[i]) ** 2)) for i, j in pairs]
    return s, dict(zip(pairs, links))


def test_the_grown_cell_rule_returns_level_zero_at_every_pure_gauge():
    """WP v17 lines 666 and 672: at level zero the rule returns the squared
    lengths exactly at every pure-gauge connection, unitary or complex, and
    the row-sum defect vanishes there; face holonomies shift the lengths."""
    rng = np.random.default_rng(6)
    for unitary in (True, False):
        theta = rng.normal(size=4) + (0.0 if unitary else
                                      0.3j * rng.normal(size=4))
        g = np.exp(1j * theta)
        links = [g[j] / g[i] for i, j in itertools.combinations(range(4), 2)]
        s, link = _generic_tetrahedron(links)
        z = dict(zip(itertools.combinations(range(4), 2), s))
        read = R.level_rule_shift([[0, 1, 2, 3]], z, link)[0]
        assert read["relative_shift"] < 1e-10
        assert read["row_sum_defect"] < 1e-12
    curved = {e: cmath.exp(1j * rng.normal()) for e in z}
    assert R.level_rule_shift([[0, 1, 2, 3]], z, curved)[0][
        "relative_shift"] > 1e-3


def test_the_phase_rule_returns_the_level_zero_connection():
    """WP v17 line 674: the connection on a grown edge is det M_vw; at level
    zero the fibers have rank one and the transport along an edge is
    multiplication by U_e, so the rule returns the level-zero connection
    exactly."""
    rng = np.random.default_rng(7)
    for _ in range(4):
        u = cmath.exp(1j * (rng.normal() + 0.2j * rng.normal()))
        assert ch.GrownCellRule.transportConnection(np.array([[u]])) == \
            pytest.approx(u, rel=1e-15)
    block = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    assert ch.GrownCellRule.transportConnection(block) == pytest.approx(
        np.linalg.det(block), rel=1e-13)


def test_the_inherited_pairing_is_gauge_and_frame_invariant():
    """WP v17 lines 656 and 676: a microscopic gauge transformation multiplies
    the numerator of g_vw and U_vw = det M_vw by the same factor, and a change
    of frame Y_v -> Y_v g_v with the dual renormalized to
    det((Y_v^vee)^T Y_v) = 1 moves U_vw by det(g_v)^{-1} det(g_w); every
    entry of g is unchanged."""
    rng = np.random.default_rng(13)
    n, r = 12, 2
    G = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    frames = [rng.normal(size=(n, r)) + 1j * rng.normal(size=(n, r))
              for _ in range(4)]
    duals = [rng.normal(size=(n, r)) + 1j * rng.normal(size=(n, r))
             for _ in range(4)]
    connection = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))

    def pairing(frames, duals, connection):
        normalized = [np.asarray(ch.GrownCellRule.normalizeDualFrame(Y, D))
                      for Y, D in zip(frames, duals)]
        return np.asarray(ch.GrownCellRule.gaugeInvariantPairing(
            normalized, [G @ Y for Y in frames], connection))

    base = pairing(frames, duals, connection)
    g = [rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r))
         for _ in range(4)]
    dets = np.array([np.linalg.det(x) for x in g])
    moved = pairing([Y @ x for Y, x in zip(frames, g)],
                    [D @ (3.0 * np.eye(r)) for D in duals],
                    connection * np.outer(1.0 / dets, dets))
    np.testing.assert_allclose(moved, base, rtol=1e-9)


def test_the_two_dimensional_scale_is_not_determined():
    """WP v17 line 678: at d = 2 the block does not determine the scale, so a
    two-dimensional cell's lengths are known only up to one scale: the
    inversion says so, returns no squared lengths, and the triangle's block is
    the same at every scale."""
    triangle = cob.ChainComplex.fromTopCells([[0, 1, 2]])
    s = [1.0 + 0.0j, 2.0 + 0.0j, 1.5 + 0.0j]
    block = np.asarray(ch.WhitneyMass.topSimplexBlocks(triangle, s, 1)[0].block)
    scaled = np.asarray(ch.WhitneyMass.topSimplexBlocks(
        triangle, [4.0 * z for z in s], 1)[0].block)
    np.testing.assert_allclose(scaled, block, rtol=1e-12)
    read = ch.GrownCellRule.invertWhitneyBlock(block)
    assert read.dimension == 2 and not read.scaleDetermined
    assert read.squaredLengths == []
    ratio = np.array(read.scaledSquaredLengths) / np.array(s)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


def test_the_three_dimensional_scale_is_14400_over_the_determinant():
    """WP v17 line 664: in three dimensions the scale C of the pairing is
    14400 / det(g / C), with no square root taken."""
    tetrahedron = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    rng = np.random.default_rng(2)
    points = rng.normal(size=(4, 3))
    s = [complex(np.sum((points[j] - points[i]) ** 2))
         for i, j in itertools.combinations(range(4), 2)]
    block = np.asarray(ch.WhitneyMass.topSimplexBlocks(tetrahedron, s, 1)[0]
                       .block)
    read = ch.GrownCellRule.invertWhitneyBlock(block)
    assert read.scaleDetermined
    scaled_metric = np.asarray(read.scaledMetric)
    assert read.scale == pytest.approx(14400.0 / np.linalg.det(scaled_metric),
                                       rel=1e-10)
    np.testing.assert_allclose(read.squaredLengths, s, rtol=1e-10)


# --------------------------------------- Section 11.1: held monopole sectors


def test_held_monopole_sectors_are_the_same_before_and_after():
    """WP v17 line 508: in controlled synthesis the declared monopole sectors
    are held during every relaxation, and the monopole numbers before and
    after each relaxation agree. The fan of two unit-monopole tetrahedra on
    three sheets is relaxed with every sector held."""
    config = R.default_config(tetrahedra=2)
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    before = R.monopole_numbers(cells, links)
    report = R.relax_level(spacetime, config,
                           R.held_sectors(cells, before, count))
    assert report["converged"]
    assert before == [1, 1]
    for _, sheet_links in R.sheet_fields(spacetime, count):
        assert R.monopole_numbers(cells, sheet_links) == before
    assert report["sector_monopole_numbers"] == before * R.SHEETS
    assert report["held_modulus_drift"] < 1e-12


# ------------------------------------------------ Section 10: the detector


def _flavoured_monopole_frames():
    """A constructed flavour doubling of the unit-monopole T-averaged edge
    operator: h-bar (x) I_2 (x) I_3, in two frames that differ by a
    rotation-invariant deformation, with the rotations and sheets lifted."""
    support = obs.MonopoleSupport.tetrahedron(1)
    group = obs.MonopoleSupport.tetrahedralRotations()
    base = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    actions = [np.asarray(support.edgeRepresentation(g)) for g in group]
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    x = x + x.conj().T
    deformation = sum(d @ x @ np.linalg.inv(d) for d in actions) / len(actions)

    def lift(operator):
        return np.kron(np.kron(operator, np.eye(2)), np.eye(3))

    declaration = obs.IsospinDoubletDeclaration()
    declaration.operator_name = "constructed flavour doubling"
    n = 36
    declaration.sheet_of_cell = [i % 3 for i in range(n)]
    declaration.base_cell_of_cell = [i // 3 for i in range(n)]
    declaration.symmetry = [lift(d) for d in actions]
    declaration.symmetry_name = "T (projective)"
    declaration.spinorial = True
    declaration.frames = [
        obs.IsospinFrame("frame 0", lift(base)),
        obs.IsospinFrame("frame 1", lift(base + 0.05 * deformation))]
    # a second resolution of the same cluster: three far-away cells per sheet
    # added, the coarse cochains prolonged by inclusion
    extra = 9
    fine = np.zeros((n + extra, n + extra), dtype=complex)
    fine[:n, :n] = lift(base)
    fine[n:, n:] = 50.0 * np.eye(extra)
    resolution = obs.IsospinResolution()
    resolution.label = "padded"
    resolution.operator = fine
    resolution.prolongation = np.vstack([np.eye(n), np.zeros((extra, n))])
    resolution.sheet_of_cell = declaration.sheet_of_cell + [
        i % 3 for i in range(extra)]
    resolution.base_cell_of_cell = declaration.base_cell_of_cell + [
        12 + i // 3 for i in range(extra)]
    resolution.symmetry = [
        np.block([[d, np.zeros((n, extra))],
                  [np.zeros((extra, n)), np.eye(extra)]])
        for d in declaration.symmetry]
    declaration.resolutions = [resolution]
    return declaration


def test_the_isospin_detector_finds_a_constructed_doublet():
    """WP v17 line 464: an isospin doublet is an unlabeled two-dimensional
    band that neither the rotations nor the sheets act on. On a constructed
    flavour doubling, read in two frames and at two resolutions, the detector
    finds it: every band carries a doublet candidate, and the emergence and
    coherent-transport conditions pass."""
    read = obs.IsospinDoublet.observe(_flavoured_monopole_frames())
    assert len(read.candidates) == 3
    assert read.conditions[0].status == obs.QuarkConditionStatus.Passed
    assert read.conditions[1].status == obs.QuarkConditionStatus.Passed
    assert not read.no_isospin_doublet


def test_the_isospin_detector_finds_none_on_the_monopole_host():
    """WP v17 lines 464 and 737 (falsifier 8): on the three-sheeted
    unit-monopole host the bands are colour (the sheets) times spin (the
    doublets 2, 2', 2''), so no unlabeled two-dimensional flavour band
    emerges, on h_1 or on its T-average."""
    from tessera.drivers import isospin_doublet as iso
    reads = iso.observe_host(iso.declared_carrier())
    for key in ("covariant", "t_averaged"):
        assert reads[key]["candidates"] == []
        assert not reads[key]["doublet_observed"]
        assert reads[key]["falsifier_8_no_isospin_doublet"]
        assert reads[key]["conditions"][0]["failing"] == [
            "two-dimensional-flavour-band"]


# --------------------------------------------- Section 10: quark conditions


def test_quark_conditions_names_all_seven_conditions():
    """WP v17 lines 448 to 460: a quark candidate satisfies seven derived
    conditions: a persistent cluster (1), its colour-spin fiber (2), a stable
    anchor atlas (3), odd occupation (4), full-rank colour transport (5), an
    oriented lineage (6) and a stable spectral fingerprint (7). The verdict
    names each, requires evidence for each, and certifies only when all seven
    pass."""
    names = obs.QuarkConditions.condition_names()
    assert names == ["persistent-cluster", "color-spin-fiber", "anchor-atlas",
                     "odd-occupation", "color-transport", "lineage",
                     "fingerprint"]
    for number in range(1, 8):
        assert obs.QuarkConditions.statement(number)
        assert obs.QuarkConditions.required_evidence(number)
    held = [[obs.QuarkConditionEvidence(name, True, "held")
             for name in obs.QuarkConditions.required_evidence(n)]
            for n in range(1, 8)]
    verdict = obs.QuarkConditions.evaluate(held)
    assert verdict.certified
    assert [c.number for c in verdict.conditions] == list(range(1, 8))
    assert [c.name for c in verdict.conditions] == names
    assert verdict.failed == [] and verdict.not_evaluable == []
    held[3] = [obs.QuarkConditionEvidence("odd-occupation-parity", False,
                                          "even")]
    refused = obs.QuarkConditions.evaluate(held)
    assert not refused.certified
    assert refused.conditions[3].status == obs.QuarkConditionStatus.Failed
    assert refused.conditions[3].failing == ["odd-occupation-parity"]
