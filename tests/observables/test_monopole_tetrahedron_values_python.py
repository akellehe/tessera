# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The unit-monopole tetrahedron and the sharp spin read, as the recursion
driver reaches them, against the values whitepaper v17 states.

The per-cell read of `tessera.drivers.recursion` hands every host cell to
`tessera.drivers.baryon_poles`, which reads the spin content through
`MonopoleSupport` (`spinRead`, `rotationAveragedEdgeOperator`,
`edgeRepresentation`, `cocycle`, `monopoleNumber`, `u1Part`) and certifies the
three-quark spin with `SharpSpin` (`doubletSpinMatrices`, `determinant`,
`applyTotalSpinSquared`, `read`). Every expected value below is quoted from
whitepaper v17 §11.1 (the "Proposition (spin from flux)", line 504, and the
edge-mode paragraph, line 506), or is exact arithmetic on a constructed
state whose spin is known by construction.
"""
import itertools
import math

import numpy as np
import pytest

from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

MonopoleSupport = obs.MonopoleSupport
SharpSpin = obs.SharpSpin
SQRT3 = math.sqrt(3.0)
#: The three C_2 rotations of the tetrahedron (double transpositions).
C2 = ([1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0])


def _spectrum(matrix):
    return np.sort(np.linalg.eigvalsh(np.asarray(matrix)))


# ------------------------------------------------------- vertex spectra


@pytest.mark.parametrize("mu, expected", [
    (0, [0.0, 4.0, 4.0, 4.0]),
    (1, [3 - SQRT3, 3 - SQRT3, 3 + SQRT3, 3 + SQRT3]),
    (-1, [3 - SQRT3, 3 - SQRT3, 3 + SQRT3, 3 + SQRT3]),
    (2, [2.0, 2.0, 2.0, 6.0]),
])
def test_the_vertex_spectrum_at_each_monopole_number(mu, expected):
    """WP v17 line 504: at mu = 0 the combinatorial vertex spectrum is
    {0, 4^(3)}, at mu = 2 it is {2^(3), 6}, and at mu = +-1 the vertex modes
    split into two doublets {3 - sqrt 3 ^(2), 3 + sqrt 3 ^(2)}."""
    np.testing.assert_allclose(
        _spectrum(MonopoleSupport.tetrahedron(mu).vertexLaplacian()),
        expected, atol=1e-12)


@pytest.mark.parametrize("mu", [0, 1, -1, 2])
def test_every_outward_face_holonomy_is_a_quarter_turn_times_mu(mu):
    """WP v17 line 504: for the tetrahedron the outward face holonomies are
    exp(2 pi i mu / 4), and the monopole number read back from them is mu,
    odd exactly for mu = +-1."""
    read = MonopoleSupport.tetrahedron(mu).monopoleNumber()
    for holonomy in read.face_holonomies:
        assert abs(holonomy - np.exp(2j * np.pi * mu / 4)) < 1e-14
    assert read.monopole_number == mu
    assert read.odd == (mu % 2 != 0)
    assert read.total_flux == pytest.approx(2 * np.pi * mu, abs=1e-12)
    assert read.bundle


# ------------------------------------------- C_2 rotations and the cocycle


@pytest.mark.parametrize("mu, sign", [(0, 1.0), (2, 1.0), (1, -1.0),
                                      (-1, -1.0)])
def test_the_c2_rotations_commute_at_even_and_anticommute_at_odd_flux(mu,
                                                                      sign):
    """WP v17 line 504: at mu = 0 and 2 the two commuting C_2 rotations
    commute as operators; at mu = +-1 they anticommute,
    D(a) D(b) D(a)^-1 D(b)^-1 = -1, the quaternion relation of 2T. Checked
    for every pair of the three C_2 rotations on the vertex modes."""
    support = MonopoleSupport.tetrahedron(mu)
    for a, b in itertools.combinations(C2, 2):
        da = np.asarray(support.vertexRepresentation(a))
        db = np.asarray(support.vertexRepresentation(b))
        commutator = da @ db @ np.linalg.inv(da) @ np.linalg.inv(db)
        assert np.max(np.abs(commutator - sign * np.eye(4))) < 1e-13


@pytest.mark.parametrize("mu, nontrivial, phase", [
    (0, False, 1.0), (1, True, -1.0), (-1, True, -1.0), (2, False, 1.0)])
def test_the_cocycle_class_is_nontrivial_exactly_at_odd_flux(mu, nontrivial,
                                                             phase):
    """WP v17 line 504: the cocycle varpi of D_k(g) D_k(h) = varpi D_k(gh) has
    a nontrivial class in H^2(T; U(1)) iff mu is odd; the witnessing
    commutator phase of a commuting pair is -1 there and 1 otherwise, and on
    the twelve rotations the product is scalar to rounding."""
    read = MonopoleSupport.tetrahedron(mu).cocycle(
        MonopoleSupport.tetrahedralRotations())
    assert read.group_order == 12
    assert read.nontrivial == nontrivial
    assert abs(read.commutator_phase - phase) < 1e-12
    assert read.max_commutator_deviation == pytest.approx(
        2.0 if nontrivial else 0.0, abs=1e-12)
    assert read.scalar_residual < 1e-12
    assert read.certificate.holds()


def test_the_rotation_group_has_twelve_elements_and_three_c2s():
    rotations = [list(r) for r in MonopoleSupport.tetrahedralRotations()]
    assert len(rotations) == 12
    assert sorted(r for r in rotations
                  if sum(1 for i, x in enumerate(r) if i != x) == 4) == \
        sorted(C2)


# ----------------------------------------------------- the edge modes


def test_the_t_averaged_edge_operator_under_the_unit_monopole():
    """WP v17 line 506: the T-averaged twisted edge Laplacian of the unit
    monopole has spectrum {4 - 2/sqrt 3 ^(2), 4^(2), 4 + 2/sqrt 3 ^(2)}."""
    support = MonopoleSupport.tetrahedron(1)
    averaged = support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), MonopoleSupport.tetrahedralRotations())
    expected = [4 - 2 / SQRT3] * 2 + [4.0] * 2 + [4 + 2 / SQRT3] * 2
    np.testing.assert_allclose(_spectrum(averaged), expected, atol=1e-12)


def test_the_same_construction_returns_four_times_the_identity_at_zero_flux():
    """WP v17 line 506: "the same construction returns 4I at zero flux"."""
    support = MonopoleSupport.tetrahedron(0)
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), MonopoleSupport.tetrahedralRotations()))
    assert np.max(np.abs(averaged - 4.0 * np.eye(6))) < 1e-12


def test_the_six_edge_modes_are_two_plus_two_prime_plus_two_double_prime():
    """WP v17 line 506: under the unit monopole the six edge modes decompose
    as 2 + 2' + 2'', three rank-two spinor doublets; the coexact one is the
    genuine j = 1/2 doublet and sits at 4, and the three are distinguished by
    the Z_3 = 2T / Q_8 character, which takes the three values 0, 1, 2
    (`baryon_poles.aligned_doublet_frame`)."""
    support = MonopoleSupport.tetrahedron(1)
    group = MonopoleSupport.tetrahedralRotations()
    read = support.spinRead(group)
    assert [b.dimension for b in read.bands] == [2, 2, 2]
    assert all(b.spinor_doublet for b in read.bands)
    assert all(b.irreducibility_score == pytest.approx(1.0, abs=1e-12)
               for b in read.bands)
    assert read.half_integer_doublet
    doublet = read.bands[read.doublet_index]
    assert doublet.coexact and doublet.coexact_residual < 1e-12
    assert doublet.eigenvalue == pytest.approx(4.0, abs=1e-12)
    assert [b.coexact for b in read.bands].count(True) == 1
    alignment = bp.aligned_doublet_frame(support, group)
    assert sorted(alignment["trialities"]) == [0, 1, 2]
    assert alignment["intertwining_residual"] < 1e-12
    np.testing.assert_allclose(alignment["averaged_eigenvalues"],
                               [4 - 2 / SQRT3] * 2 + [4.0] * 2
                               + [4 + 2 / SQRT3] * 2, atol=1e-12)


def test_the_edge_action_intertwines_the_twisted_coboundary():
    """WP v17 line 506: D_1 intertwines delta_0^U, D_1(g) delta D_0(g)^-1 =
    delta, on every rotation of the unit-monopole tetrahedron."""
    support = MonopoleSupport.tetrahedron(1)
    delta = np.asarray(support.twistedCoboundary())
    for g in MonopoleSupport.tetrahedralRotations():
        d1 = np.asarray(support.edgeRepresentation(g))
        d0 = np.asarray(support.vertexRepresentation(g))
        assert np.max(np.abs(d1 @ delta - delta @ d0)) < 1e-13


def test_the_even_monopole_carries_no_half_integer_doublet():
    """At mu = 2 the class is trivial (WP v17 line 504), so no band is a
    spinor doublet and no j = 1/2 doublet is read."""
    support = MonopoleSupport.tetrahedron(2)
    read = support.spinRead(MonopoleSupport.tetrahedralRotations())
    assert not read.half_integer_doublet
    assert read.doublet_index == len(read.bands)
    assert not any(b.spinor_doublet for b in read.bands)


# ------------------------------------------------------------ refusals


def test_a_connection_off_the_unit_circle_is_refused_and_u1_part_repairs_it():
    fixture = MonopoleSupport.tetrahedron(1)
    scaled = [2.0 * u for u in fixture.connection]
    with pytest.raises(ValueError, match="has modulus 2"):
        MonopoleSupport(4, fixture.edges, fixture.faces, scaled)
    repaired = MonopoleSupport(4, fixture.edges, fixture.faces,
                               MonopoleSupport.u1Part(scaled))
    assert repaired.monopoleNumber().monopole_number == 1
    with pytest.raises(ValueError, match="is zero and has no U\\(1\\) part"):
        MonopoleSupport.u1Part([0.0] + scaled[1:])


def test_structural_refusals_of_the_support_are_named():
    fixture = MonopoleSupport.tetrahedron(1)
    connection = list(fixture.connection)
    with pytest.raises(ValueError, match="6 edges were declared but 5 "
                                         "connection values"):
        MonopoleSupport(4, fixture.edges, fixture.faces, connection[:5])
    with pytest.raises(ValueError, match="is stored as \\(1, 0\\)"):
        MonopoleSupport(4, [[1, 0]] + [list(e) for e in fixture.edges[1:]],
                        fixture.faces, connection)
    with pytest.raises(ValueError, match="no declared edge joins vertices "
                                         "0 and 0"):
        fixture.transport(0, 0)
    with pytest.raises(ValueError, match="a rotation is a permutation of the "
                                         "4 vertices; received 3 images"):
        fixture.edgeRepresentation([0, 1, 2])
    with pytest.raises(ValueError, match="is not in the supplied set"):
        fixture.cocycle([[0, 1, 2, 3], [1, 2, 0, 3]])


# ------------------------------------------------------------ SharpSpin


def _spin():
    return SharpSpin.doubletSpinMatrices(3)


def test_the_spin_matrices_are_three_spin_half_carriers():
    """`doubletSpinMatrices(3)` is I_3 (x) sigma_a / 2 in the mode order
    2 c + s, so J_z = diag(1/2, -1/2) on every carrier and the one-particle
    J^2 is 3/4 on every mode."""
    jx, jy, jz = (np.asarray(j) for j in _spin())
    assert np.max(np.abs(jz - np.kron(np.eye(3), np.diag([0.5, -0.5])))) \
        == 0.0
    one_particle = jx @ jx + jy @ jy + jz @ jz
    assert np.max(np.abs(one_particle - 0.75 * np.eye(6))) < 1e-15


def test_three_aligned_spins_are_a_sharp_spin_three_halves():
    """|up, up, up> over the three carriers (modes 0, 2, 4) is the top state
    of j = 3/2: J^2 psi = 15/4 psi exactly, so the read at 15/4 is sharp with
    zero residuals, one determinant, and expectation 15/4."""
    state = SharpSpin.determinant([0, 2, 4], 6)
    read = SharpSpin.read(_spin(), state, state, 3.75, 1e-12)
    assert read.sharp
    assert read.right_residual < 1e-15 and read.left_residual < 1e-15
    assert abs(read.expectation - 3.75) < 1e-14
    assert read.determinant_count == 1


def test_three_aligned_spins_are_not_spin_half_by_exactly_three():
    """At the spin-1/2 target the same state has (J^2 - 3/4) psi = 3 psi, so
    both relative residuals are exactly 3 and the read is not sharp."""
    state = SharpSpin.determinant([0, 2, 4], 6)
    read = SharpSpin.read(_spin(), state, state, 0.75, 1e-9)
    assert not read.sharp
    assert read.right_residual == pytest.approx(3.0, rel=1e-14)
    assert read.left_residual == pytest.approx(3.0, rel=1e-14)


def test_a_constructed_spin_half_superposition_is_sharp():
    """(|up_0 down_1> - |down_0 up_1>) |up_2> couples carriers 0 and 1 to
    spin 0 and leaves carrier 2 as the spin: a j = 1/2 eigenstate of two
    determinants, sharp at 3/4 with zero residuals."""
    state = SharpSpin.determinantSuperposition([[0, 3, 4], [1, 2, 4]],
                                               [1.0, -1.0], 6)
    read = SharpSpin.read(_spin(), state, state, 0.75, 1e-12)
    assert read.sharp
    assert read.right_residual < 1e-15
    assert abs(read.expectation - 0.75) < 1e-14
    assert read.determinant_count == 2


def test_the_symmetric_combination_is_spin_three_halves_not_half():
    """(|up_0 down_1> + |down_0 up_1>) |up_2> is the pair's triplet m = 0
    state times an up spin, |1, 0> |1/2, 1/2> = sqrt(2/3) |3/2, 1/2> -
    sqrt(1/3) |1/2, 1/2> by the Clebsch-Gordan table: a mixture of j = 3/2 and
    j = 1/2, so it is sharp at neither 3/4 nor 15/4, and its expectation is
    (2/3)(15/4) + (1/3)(3/4) = 11/4."""
    state = SharpSpin.determinantSuperposition([[0, 3, 4], [1, 2, 4]],
                                               [1.0, 1.0], 6)
    for target in (0.75, 3.75):
        assert not SharpSpin.read(_spin(), state, state, target,
                                  1e-9).sharp
    read = SharpSpin.read(_spin(), state, state, 0.75, 1e-9)
    assert abs(read.expectation - 2.75) < 1e-14


def test_the_applied_and_dense_total_spin_agree_on_six_modes():
    """`applyTotalSpinSquared` equals the dense `totalSpinSquaredMatrix` on a
    fixed state of the 64-dimensional Fock space, whose spectrum is the
    exterior algebra's: J^2 eigenvalues 0, 3/4, 2, 15/4 only."""
    dense = np.asarray(SharpSpin.totalSpinSquaredMatrix(_spin()))
    values = np.unique(np.round(np.linalg.eigvalsh(dense), 12))
    np.testing.assert_allclose(values, [0.0, 0.75, 2.0, 3.75], atol=1e-12)
    state = np.arange(64, dtype=complex) + 1j
    applied = np.asarray(SharpSpin.applyTotalSpinSquared(_spin(), state))
    assert np.max(np.abs(applied - dense @ state)) < 1e-12


def test_the_sharp_spin_refusals_are_named():
    spin = _spin()
    state = SharpSpin.determinant([0, 2, 4], 6)
    with pytest.raises(ValueError, match="neither state may be the zero "
                                         "vector"):
        SharpSpin.read(spin, 0 * state, state)
    with pytest.raises(ValueError, match="the right state has dimension 64 "
                                         "and the left state"):
        SharpSpin.read(spin, state, state[:32])
    with pytest.raises(ValueError, match="must be positive"):
        SharpSpin.read(spin, state, state, 0.75, 0.0)
    with pytest.raises(ValueError, match="mode 2 is occupied twice"):
        SharpSpin.determinant([2, 2], 6)
    with pytest.raises(ValueError, match="mode 6 is outside the 6-mode "
                                         "space"):
        SharpSpin.determinant([6], 6)
    with pytest.raises(ValueError, match="the mode count must lie between "
                                         "one and 24; received 25"):
        SharpSpin.determinant([0], 25)
    with pytest.raises(ValueError, match="2 determinants were listed but 1 "
                                         "amplitudes"):
        SharpSpin.determinantSuperposition([[0], [1]], [1.0], 6)
    with pytest.raises(ValueError, match="at least one spin-one-half "
                                         "carrier"):
        SharpSpin.doubletSpinMatrices(0)
    with pytest.raises(ValueError, match="13 carriers need 26 modes, above "
                                         "the limit of 24"):
        SharpSpin.doubletSpinMatrices(13)
    with pytest.raises(ValueError, match="dense Fock operator is materialized "
                                         "only"):
        SharpSpin.totalSpinSquaredMatrix(SharpSpin.doubletSpinMatrices(9))
    with pytest.raises(ValueError, match="which is not 2\\^M"):
        SharpSpin.applyTotalSpinSquared(spin, np.ones(63, dtype=complex))
