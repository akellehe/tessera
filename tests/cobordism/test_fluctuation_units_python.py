# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The fluctuation reads do not depend on the unit their data is expressed in.

``DressedFluctuation`` and ``PencilSchur.feshbach`` take a carrier operator
and a stiffness whose size is the unit of the level they are read on: on a
level whose squared lengths are of order 1e9 the carrier is of order 1e-9.
Every comparison these reads make is therefore relative to the size of the
data it is made on, and none has a floor in the unit of the carrier or of the
stiffness. A read has no value only where a denominator is exactly zero or a
matrix has no finite inverse; a matrix that is singular at the declared
tolerance and has an inverse is inverted as computed, and the read says so.

Terms used below:

* the *polarization* Pi(w) is the paramagnetic term of the dressed stiffness,
  a sum over the pairs of an occupied mode m and an empty mode n of
  Delta [(O_a)_mn (O_b)_nm + (O_a)_nm (O_b)_mn] / (Delta^2 - w^2), with
  Delta the particle-hole energy lambda_n - lambda_m;
* a *collective mode* is a frequency w at which the dressed stiffness
  A + D - Pi(w) is singular;
* the *Drazin inverse* A^D of a matrix A inverts A on the invariant subspace
  complementary to its generalized null space and is zero on that null space;
  ``PencilSchur.feshbach`` forms it as (A + u Pi_0)^-1 (I - Pi_0) with Pi_0
  the Riesz projector onto the null space and u the spectral radius of A;
* *defective* (of a carrier) and *singular* (of a stiffness) are marks: a
  pivot of the LU decomposition at or below the declared tolerance times the
  largest pivot.
"""
import itertools

import numpy as np
import pytest

import tessera as T

cob = T.cobordism
PS = T.chainhodge.PencilSchur

#: The units a carrier or a stiffness is expressed in: a level whose squared
#: lengths are of order 1e9, one of order one, and one of order 1e-9.
UNITS = (1e-9, 1.0, 1e9)


def _flat(matrix):
    return [complex(value) for value in np.asarray(matrix).reshape(-1)]


def _relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(a)))


def _declaration(carrier, couplings, second=None, stiffness=None, occupied=1,
                 tolerance=1e-15, broadening=0.0):
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = int(np.asarray(carrier).shape[0])
    declaration.carrier = _flat(carrier)
    declaration.couplings = [_flat(operator) for operator in couplings]
    if second is not None:
        declaration.second_derivatives = [_flat(operator)
                                          for operator in second]
    if stiffness is not None:
        declaration.bare_stiffness = _flat(stiffness)
    declaration.occupied_modes = occupied
    declaration.tolerance = tolerance
    declaration.continuum_broadening = broadening
    return declaration


def _symmetric_fixture(order=4, fluctuations=2, seed=2):
    """A real symmetric carrier with simple eigenvalues and real symmetric
    couplings."""
    rng = np.random.default_rng(seed)
    carrier = rng.normal(size=(order, order))
    carrier = carrier + carrier.T
    couplings = []
    for _ in range(fluctuations):
        coupling = rng.normal(size=(order, order))
        couplings.append(coupling + coupling.T)
    return carrier.astype(complex), [c.astype(complex) for c in couplings]


def _reference_polarization(carrier, couplings, occupied, frequency=0.0):
    """Pi(w) from one dense eigendecomposition, the left frame the inverse
    of the right one, both orders of the two matrix elements."""
    values, right = np.linalg.eig(np.asarray(carrier, dtype=complex))
    order = sorted(range(len(values)),
                   key=lambda index: (values[index].real, values[index].imag))
    values, right = values[order], right[:, order]
    left = np.linalg.inv(right)
    currents = [left @ np.asarray(o, dtype=complex) @ right for o in couplings]
    count = len(couplings)
    polarization = np.zeros((count, count), dtype=complex)
    for m in range(occupied):
        for n in range(occupied, len(values)):
            gap = values[n] - values[m]
            forward = np.outer([c[m, n] for c in currents],
                               [c[n, m] for c in currents])
            polarization += gap / (gap * gap - frequency * frequency) * (
                forward + forward.T)
    return polarization


# -------------------------------------------------------- the polarization


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("frequency", [0.0, 0.4, 0.3 + 0.2j])
def test_the_polarization_carries_the_unit_of_the_carrier(unit, frequency):
    """Every term of Pi is O O / Delta, so Pi of the carrier c h with the
    couplings c O_a at the frequency c w is c Pi of h and O_a at w, for a
    carrier of order 1e-9 as for one of order one."""
    carrier, couplings = _symmetric_fixture()
    reference = _reference_polarization(carrier, couplings, 2, frequency)
    fluctuation = cob.DressedFluctuation(_declaration(
        unit * carrier, [unit * o for o in couplings], occupied=2))
    polarization = np.asarray(
        fluctuation.paramagnetic(unit * frequency)).reshape(2, 2)
    assert _relative(unit * reference, polarization) < 1e-12


def test_the_polarization_has_no_value_only_at_an_exact_pole():
    """At a particle-hole energy the denominator is zero and the polarization
    has no value. One unit in the last place away from it the denominator is
    not zero: the polarization has a value, the dense reference's, and the
    proximity of the frequency to the pole is reported."""
    carrier = np.diag([0.0, 2.0]).astype(complex)
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    fluctuation = cob.DressedFluctuation(
        _declaration(carrier, [coupling], occupied=1))
    with pytest.raises(ValueError, match="has a pole and no value"):
        fluctuation.paramagnetic(2.0)
    assert fluctuation.pole_proximity(2.0) == 0.0

    beside = float(np.nextafter(2.0, 3.0))
    value = fluctuation.paramagnetic(beside)[0]
    expected = 2.0 * 2.0 / (2.0 * 2.0 - beside * beside)
    assert np.isfinite(value)
    assert value == pytest.approx(expected, rel=1e-12)
    assert 0.0 < fluctuation.pole_proximity(beside) < 1e-15
    # the static polarization of a state with a gap sits a whole unit from
    # every pole, whatever the unit of the carrier is
    for unit in UNITS:
        scaled = cob.DressedFluctuation(_declaration(
            unit * carrier, [unit * coupling], occupied=1))
        assert scaled.pole_proximity(0.0) == 1.0
        assert scaled.smallest_relative_gap() == 1.0


def test_the_smallest_gap_is_reported_in_the_unit_of_the_carrier():
    """The smallest particle-hole energy over the largest eigenvalue
    modulus: 1/5 for the levels 0 | 1, 5 with one occupied mode, and not a
    number when the state has no particle-hole pair."""
    carrier = np.diag([5.0, 0.0, 1.0]).astype(complex)
    coupling = np.ones((3, 3), dtype=complex)
    for unit in UNITS:
        fluctuation = cob.DressedFluctuation(_declaration(
            unit * carrier, [unit * coupling], occupied=1))
        assert fluctuation.smallest_relative_gap() == pytest.approx(
            0.2, rel=1e-14)
    full = cob.DressedFluctuation(_declaration(carrier, [coupling],
                                               occupied=3))
    assert np.isnan(full.smallest_relative_gap())
    assert np.isnan(full.pole_proximity(0.0))


# ------------------------------------------- the modes of decoupled blocks


def _three_sheets(interleaved):
    """A carrier of three equal blocks that no entry couples, with couplings
    of the same structure, and one block of each."""
    rng = np.random.default_rng(11)
    sheet = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    sheet = sheet + sheet.T
    coupling = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    coupling = coupling + coupling.T
    if interleaved:
        # index 3 k + t is coordinate k of block t
        lift = lambda block: np.kron(block, np.eye(3))  # noqa: E731
    else:
        lift = lambda block: np.kron(np.eye(3), block)  # noqa: E731
    return sheet, coupling, lift


@pytest.mark.parametrize("interleaved", [False, True])
def test_equal_decoupled_blocks_keep_their_degeneracy_exactly(interleaved):
    """Each decoupled block of the carrier is decomposed on its own, so the
    three copies of an eigenvalue are equal bit for bit, whatever order the
    blocks' coordinates are listed in. With one copy of the lowest level
    occupied and two empty, two particle-hole energies are zero exactly: the
    state's smallest gap is zero and its static polarization has no value.
    With the whole level occupied the polarization is three times one
    block's."""
    sheet, coupling, lift = _three_sheets(interleaved)
    carrier, lifted = lift(sheet), lift(coupling)
    whole = cob.DressedFluctuation(_declaration(carrier, [lifted],
                                                occupied=3))
    values = whole.carrier_eigenvalues()
    for level in range(4):
        assert values[3 * level] == values[3 * level + 1] \
            == values[3 * level + 2]
    assert not whole.carrier_defective()
    one = cob.DressedFluctuation(_declaration(sheet, [coupling], occupied=1))
    assert _relative(3.0 * np.asarray(one.paramagnetic(0.0)),
                     np.asarray(whole.paramagnetic(0.0))) < 1e-12
    assert whole.smallest_relative_gap() == pytest.approx(
        one.smallest_relative_gap(), rel=1e-12)

    split = cob.DressedFluctuation(_declaration(carrier, [lifted],
                                                occupied=1))
    assert split.smallest_relative_gap() == 0.0
    assert split.pole_proximity(0.0) == 0.0
    with pytest.raises(ValueError, match="has a pole and no value"):
        split.paramagnetic(0.0)


def test_a_defective_carrier_is_read_with_its_frame_as_computed():
    """A Jordan block beside a simple eigenvalue: the two computed modes of
    the block are parallel to rounding, so the frame of right modes is
    singular at the declared tolerance. The instance is built, says the
    carrier is defective with the reciprocal condition of the frame, and
    makes its reads with the left frame as computed. A diagonalizable
    carrier is not marked."""
    jordan = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
                      dtype=complex)
    coupling = np.array([[0.6, 0.3, 1.3], [0.3, -0.8, 1.4], [1.3, 1.4, 0.2]],
                        dtype=complex)
    fluctuation = cob.DressedFluctuation(
        _declaration(jordan, [coupling], occupied=2))
    assert fluctuation.carrier_defective()
    assert fluctuation.mode_frame_reciprocal_condition() < 1e-14
    np.testing.assert_allclose(fluctuation.carrier_eigenvalues(),
                               [1.0, 1.0, 3.0], atol=1e-12)
    assert np.all(np.isfinite(np.asarray(fluctuation.paramagnetic(0.0))))

    carrier, couplings = _symmetric_fixture()
    regular = cob.DressedFluctuation(_declaration(carrier, couplings,
                                                  occupied=2))
    assert not regular.carrier_defective()
    assert regular.mode_frame_reciprocal_condition() > 1e-3


# ------------------------------------------------------ the collective modes


def _closed_form(unit=1.0, ratio=None, tolerance=1e-8):
    """One particle-hole pair and one retained fluctuation, whose poles are
    w^2 = Delta^2 (1 - pi / (a + d)) with pi = Pi(0). ``ratio`` sets
    pi / a for a system without a diamagnetic term."""
    gap, stiffness = 2.5, 1.4
    carrier = np.diag([0.0, gap]).astype(complex)
    if ratio is None:
        element = 0.8
        second = [np.diag([0.6, 0.0]).astype(complex)]
    else:
        element = np.sqrt(ratio * gap * stiffness / 2.0)
        second = None
    coupling = np.array([[0.0, element], [element, 0.0]], dtype=complex)
    fluctuation = cob.DressedFluctuation(_declaration(
        unit * carrier, [unit * coupling],
        None if second is None else [unit * s for s in second],
        unit * np.array([[stiffness]], dtype=complex), occupied=1,
        tolerance=tolerance))
    return fluctuation, gap, stiffness


@pytest.mark.parametrize("unit", UNITS)
def test_the_collective_modes_carry_the_unit_of_the_carrier(unit):
    """The poles of the carrier c h with the couplings, the second
    derivatives and the bare stiffness times c are c times the poles of h:
    the closed form w^2 = Delta^2 (1 - pi / (a + d)) at every unit."""
    fluctuation, gap, stiffness = _closed_form(unit)
    scalar = fluctuation.diamagnetic()[0] / unit + stiffness
    polarization = fluctuation.paramagnetic(0.0)[0] / unit
    expected = gap * gap * (1.0 - polarization / scalar)
    modes = fluctuation.collective_modes()
    assert len(modes) == 2
    for mode in modes:
        assert (mode.frequency / unit) ** 2 == pytest.approx(expected,
                                                             rel=1e-9)
        assert mode.residual < 1e-10
        assert mode.continuum_distance / unit == pytest.approx(
            abs(abs(mode.frequency / unit) - gap), rel=1e-6)
    assert modes[0].frequency == pytest.approx(-modes[1].frequency,
                                               rel=1e-10)


@pytest.mark.parametrize("unit", UNITS)
def test_every_pole_of_a_tetrahedron_is_a_zero_at_every_unit(unit):
    """The pole search on the tetrahedron's covariant operator with its
    phase couplings, at three units of the operator: as many modes at each,
    and every one a zero of the dressed stiffness."""
    ch = T.chainhodge
    rng = np.random.default_rng(9)
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    edges = K.numSimplices(1)
    squared = [complex(1.0 + 0.1 * rng.uniform(-1, 1)) for _ in range(edges)]
    links = [complex(np.exp(1j * rng.uniform(-np.pi, np.pi)))
             for _ in range(edges)]
    covariant = ch.CovariantChainHodge(ch.ChainHodge(K, squared),
                                       ch.Connection(K, links), 7, False)
    boundary = np.asarray(K.boundaryMatrix(2), dtype=float).reshape(
        edges, K.numSimplices(2))
    carrier = np.asarray(covariant.covariantOperator(1))
    couplings = [np.asarray(covariant.covariantOperatorPhaseDerivative(1, a))
                 for a in range(edges)]
    second = [np.asarray(covariant.covariantOperatorPhaseHessian(1, a, b))
              for a in range(edges) for b in range(a, edges)]
    stiffness = 0.5 * (boundary @ boundary.T).astype(complex)

    def modes_at(c):
        fluctuation = cob.DressedFluctuation(_declaration(
            c * carrier, [c * o for o in couplings], [c * s for s in second],
            c * stiffness, occupied=3, tolerance=1e-8))
        return fluctuation, fluctuation.collective_modes()

    _, reference = modes_at(1.0)
    fluctuation, modes = modes_at(unit)
    assert len(reference) > 0 and len(modes) == len(reference)
    for mode in modes:
        dressed = np.asarray(fluctuation.dressed_stiffness(
            mode.frequency)).reshape(edges, edges)
        singular = np.linalg.svd(dressed, compute_uv=False)
        assert singular[-1] < 1e-6 * singular[0]


def test_a_pole_beside_a_particle_hole_energy_is_reported_with_its_distance():
    """With pi / a = 2e-6 the closed form puts the pole at
    Delta sqrt(1 - 2e-6), 2.5e-6 from the particle-hole energy 2.5. At the
    declared tolerance 1e-4 that is within the tolerance of the energy. The
    pole is a zero of the dressed stiffness and is reported, with its
    distance from the energy."""
    fluctuation, gap, _ = _closed_form(ratio=2e-6, tolerance=1e-4)
    expected = gap * np.sqrt(1.0 - 2e-6)
    modes = fluctuation.collective_modes()
    assert len(modes) == 2
    for mode in modes:
        assert abs(mode.frequency) == pytest.approx(expected, rel=1e-8)
        assert mode.continuum_distance == pytest.approx(gap - expected,
                                                        rel=1e-4)
        assert mode.continuum_distance < 1e-4 * gap
        assert mode.residual < 1e-7


# ------------------------------------------------------- the elimination


def _elimination_fixture(seed=17):
    rng = np.random.default_rng(seed)
    carrier = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    couplings = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
                 for _ in range(2)]
    return carrier, couplings


def _reference_quartic(couplings, response, particles=2):
    """-1/2 sum_ab response_ab J_a J_b on the ``particles``-particle space
    of the whole carrier, from the elementary creation and annihilation
    matrices of the exterior algebra."""
    rank = couplings[0].shape[0]
    states = []
    for count in range(rank + 1):
        states.extend(itertools.combinations(range(rank), count))
    position = {state: index for index, state in enumerate(states)}
    size = len(states)
    creation = [np.zeros((size, size), dtype=complex) for _ in range(rank)]
    annihilation = [np.zeros((size, size), dtype=complex)
                    for _ in range(rank)]
    for state in states:
        for mode in range(rank):
            sign = (-1.0) ** sum(1 for member in state if member < mode)
            if mode in state:
                target = tuple(m for m in state if m != mode)
                annihilation[mode][position[target], position[state]] = sign
            else:
                target = tuple(sorted(state + (mode,)))
                creation[mode][position[target], position[state]] = sign
    currents = []
    for coupling in couplings:
        total = np.zeros((size, size), dtype=complex)
        for i in range(rank):
            for j in range(rank):
                total += coupling[i, j] * (creation[i] @ annihilation[j])
        currents.append(total)
    quartic = np.zeros((size, size), dtype=complex)
    for a, b in itertools.product(range(len(couplings)), repeat=2):
        quartic -= 0.5 * response[a, b] * (currents[a] @ currents[b])
    sector = [position[state] for state in states if len(state) == particles]
    return quartic[np.ix_(sector, sector)]


@pytest.mark.parametrize("unit", UNITS)
def test_the_elimination_carries_the_unit_of_the_stiffness(unit):
    """-1/2 J^T (c A)^-1 J is 1/c of -1/2 J^T A^-1 J: the mark of a
    singular stiffness is taken against the stiffness's own largest pivot,
    and a regular stiffness of any unit is not marked."""
    carrier, couplings = _elimination_fixture()
    stiffness = np.array([[2.0, 0.3 + 0.1j], [0.3 + 0.1j, 1.5]],
                         dtype=complex)
    read = cob.DressedFluctuation(_declaration(
        carrier, couplings, stiffness=unit * stiffness,
        occupied=2)).effective_action([], [], 2)
    reference = _reference_quartic(couplings, np.linalg.inv(stiffness))
    quartic = np.asarray(read.quartic).reshape(6, 6)
    assert _relative(reference, unit * quartic) < 1e-12
    assert not read.stiffness_singular
    # the reciprocal condition number in the 1-norm
    expected = 1.0 / (np.linalg.norm(stiffness, 1)
                      * np.linalg.norm(np.linalg.inv(stiffness), 1))
    assert read.stiffness_reciprocal_condition == pytest.approx(expected,
                                                                rel=0.5)


def test_a_stiffness_singular_at_the_tolerance_is_inverted_and_marked():
    """A stiffness whose second pivot is 1e-17 of its first is singular at
    the declared tolerance 1e-15 and has an inverse: the elimination is made
    with it, says the stiffness is singular and reports its reciprocal
    condition. A stiffness with an exactly zero pivot has no inverse, and
    the elimination has no value."""
    carrier, couplings = _elimination_fixture()
    stiffness = np.diag([1.0, 1e-17]).astype(complex)
    read = cob.DressedFluctuation(_declaration(
        carrier, couplings, stiffness=stiffness,
        occupied=2)).effective_action([], [], 2)
    assert read.stiffness_singular
    assert read.stiffness_reciprocal_condition == pytest.approx(1e-17,
                                                                rel=1e-6)
    reference = _reference_quartic(couplings, np.diag([1.0, 1e17]))
    assert _relative(reference,
                     np.asarray(read.quartic).reshape(6, 6)) < 1e-12

    without = cob.DressedFluctuation(_declaration(
        carrier, couplings, stiffness=np.diag([1.0, 0.0]).astype(complex),
        occupied=2))
    with pytest.raises(ValueError, match="no finite inverse"):
        without.effective_action([], [], 2)


def test_an_elimination_with_no_retained_fluctuation_is_the_one_body_term():
    """With no coupling and no stiffness nothing is integrated out: the
    quartic and its two parts are zero, the effective action is the
    one-body term, and no stiffness is inverted, so it has no condition
    number. With a coupling and no stiffness there is nothing to invert."""
    carrier, couplings = _elimination_fixture()
    read = cob.DressedFluctuation(_declaration(
        carrier, [], occupied=2)).effective_action([], [], 2)
    assert read.dimension == 6
    for part in (read.quartic, read.induced_one_body,
                 read.normal_ordered_quartic):
        assert not np.any(np.asarray(part))
    np.testing.assert_array_equal(np.asarray(read.effective_action),
                                  np.asarray(read.one_body))
    assert np.isnan(read.stiffness_conditioning)
    assert np.isnan(read.stiffness_reciprocal_condition)
    assert not read.stiffness_singular
    assert read.certificate.holds()

    with pytest.raises(ValueError, match="none is declared"):
        cob.DressedFluctuation(_declaration(
            carrier, couplings, occupied=2)).effective_action([], [], 2)


def test_no_cap_on_the_many_body_dimension_is_declared_by_default():
    """The dimension of the particle space is capped only by a cap the
    caller declares: the default is none, and a declared cap that the
    dimension exceeds ends the read by name."""
    signature = cob.DressedFluctuation.effective_action.__doc__.splitlines()[0]
    assert "dimension_cap" in signature and "= None" in signature
    carrier, couplings = _elimination_fixture()
    fluctuation = cob.DressedFluctuation(_declaration(
        carrier, couplings, stiffness=np.eye(2, dtype=complex), occupied=2))
    assert fluctuation.effective_action([], [], 2, None).dimension == 6
    assert fluctuation.effective_action([], [], 2, 6).dimension == 6
    with pytest.raises(ValueError, match="above the declared cap of 5"):
        fluctuation.effective_action([], [], 2, 5)


# ------------------------------------------------------ the Drazin inverse


def _singular_symmetric():
    """A complex symmetric matrix of order six with a null space of
    dimension two and index one, Q diag(d) Q^T with Q real orthogonal, and
    its Drazin inverse Q diag(1 / d on the nonzero d) Q^T."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    d = np.array([0, 0, 1.5 + 0.2j, -0.7, 2.0 - 1j, 0.3 + 0.4j])
    inverse = np.array([0, 0] + [1.0 / value for value in d[2:]])
    return q @ np.diag(d) @ q.T, q @ np.diag(inverse) @ q.T


def _drazin(matrix):
    return PS.feshbach(matrix, np.zeros_like(matrix), 0j, [], 1e-15, 1e-15)


@pytest.mark.parametrize("unit", [1e-12, 1e-9, 1e-3, 1.0, 1e3, 1e9, 1e12])
def test_the_drazin_inverse_carries_the_unit_of_the_matrix(unit):
    """(c A)^D = A^D / c, with the three defining identities
    A A^D A = A (index one), A^D A A^D = A^D and A A^D = A^D A, at every
    unit: the Riesz projector is added to the matrix in the matrix's own
    unit, its spectral radius."""
    matrix, exact = _singular_symmetric()
    scaled = unit * matrix
    read = _drazin(scaled)
    assert read.interiorSingular and read.interiorRank == 4
    drazin = np.asarray(read.interiorInverse)
    assert _relative(exact / unit, drazin) < 1e-13
    assert _relative(scaled, scaled @ drazin @ scaled) < 1e-13
    assert _relative(drazin, drazin @ scaled @ drazin) < 1e-13
    assert np.linalg.norm(scaled @ drazin - drazin @ scaled) < 1e-13
    assert read.drazinUnit == pytest.approx(
        unit * max(abs(np.linalg.eigvals(matrix))), rel=1e-12)


def test_the_drazin_inverse_of_a_matrix_without_a_spectral_radius_is_zero():
    """A nilpotent block and the zero block have every eigenvalue zero: the
    generalized null space is the whole space, the Drazin inverse is zero,
    and the unit the projector is added in is the block's norm, or one for
    the zero block. Away from a resonance no Drazin inverse is formed."""
    nilpotent = np.array([[0.0, 3.0], [0.0, 0.0]], dtype=complex)
    read = _drazin(nilpotent)
    assert read.interiorSingular and read.interiorRank == 0
    assert not np.any(np.asarray(read.interiorInverse))
    assert read.drazinUnit == pytest.approx(3.0, rel=1e-15)

    read = _drazin(np.zeros((3, 3), dtype=complex))
    assert read.interiorSingular and read.interiorRank == 0
    assert not np.any(np.asarray(read.interiorInverse))
    assert read.drazinUnit == 1.0

    regular = _drazin(np.diag([1.0, 2.0]).astype(complex))
    assert not regular.interiorSingular
    assert np.isnan(regular.drazinUnit)


# ---------------------------------------------- the surrogate's two metrics


def _surrogate(metric_entry, rank_tolerance=1e-12):
    """A diagonal pencil of order four with the interface coordinate 0, the
    interior metric diag(1, 1, ``metric_entry``)."""
    stiffness = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    stiffness[0, 1] = stiffness[1, 0] = 0.25
    metric = np.diag([1.0, 1.0, 1.0, metric_entry]).astype(complex)
    return PS.craigBamptonSurrogate(stiffness, metric, [0], 2.0 + 0j, 1.5,
                                    1e6, 0j, 1e-6, rank_tolerance)


def test_a_surrogate_with_a_singular_metric_is_returned_uncertified():
    """An interior metric whose smallest pivot is 1e-14 of its largest is
    singular at the rank tolerance 1e-12 and has an inverse: the surrogate
    is returned with its numbers, says the metric is singular with its
    reciprocal condition, and does not certify. A metric with an exactly
    zero pivot has no inverse, and the fixed-interface pencil no spectrum.
    A regular metric is not marked."""
    regular = _surrogate(1.0)
    assert not regular.interiorMetricSingular
    assert not regular.reducedMetricSingular
    assert regular.interiorMetricReciprocalCondition == pytest.approx(1.0)

    read = _surrogate(1e-14)
    assert read.interiorMetricSingular
    assert read.interiorMetricReciprocalCondition == pytest.approx(1e-14,
                                                                   rel=1e-6)
    assert not read.certified
    assert "interior chain metric M_II is singular" in read.refusal
    assert len(read.interiorEigenvalues) == 3
    assert np.all(np.isfinite(np.asarray(read.interiorEigenvalues)))

    with pytest.raises(RuntimeError, match="no finite inverse"):
        _surrogate(0.0)
