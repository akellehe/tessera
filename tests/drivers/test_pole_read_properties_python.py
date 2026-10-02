# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Properties of the reads made on a cell after its solve
(`tessera.drivers.baryon_poles`), each stated and checked at the declared
tolerances.

Terms used below:

* a *cell* is one tetrahedron given by its six squared lengths and its six
  links on the ascending orientation of its edges, in the order of
  `MonopoleSupport.edges`, the mapping `build_host` and `build_base` take;
* the *fixture* is the declared symmetric host cell: squared lengths 8 and
  the unit-monopole links of `monopole_support()`;
* a *gauge copy* of a cell has the links g_a U_ab / g_b for a number g_a per
  vertex and the same squared lengths; it is the same configuration
  (whitepaper v18, section 3), of unit modulus when every |g_a| is one;
* a *relabeled* cell has its vertices renamed by a permutation, its edge
  fields carried with them;
* the *unit* of an operator is a common factor of all its entries: the
  covariant operator h_1 of a cell whose squared lengths are multiplied by c
  is h_1 / c, so a property that changes with the factor depends on the unit
  the lengths are measured in.

A test that states a property the code does not have is a strict expected
failure whose reason names the defect.
"""

import cmath
import itertools

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

PAIRS = tuple(itertools.combinations(range(4), 2))

#: The host cell (0, 1, 2, 3) of tick 1 of the recursion run of 2026-10-01
#: (`v18-multicobordism-5t`): squared lengths of order 1e9, links off the
#: unit circle, and no rotation of the tetrahedron as a symmetry.
DILATED_CELL = {
    "squared_lengths": [
        2842442056.134229 - 91405.50229423534j,
        3035544551.267494 - 562355.1291879081j,
        4480356235.888196 + 852532.8843812894j,
        5157071901.545203 - 338028.86587001313j,
        5657491805.671762 + 879501.240034435j,
        6154624323.67364 + 432819.4583564187j],
    "links": [
        -0.04225458178179704 + 0.1097954824191668j,
        -0.12203545645129985 + 0.04494497620869077j,
        -0.10648073217619775 - 0.2029597947019897j,
        -0.5011746868080266 + 0.9456600494109102j,
        0.9422698455536058 + 1.7779568125833112j,
        -1.11479931570793 - 1.590042265101136j],
}

#: The host cell (0, 1, 3, 4) of tick 0 of the same run: squared lengths 8
#: and links of unit modulus whose face holonomies are 1, omega, omega^-1
#: and omega, omega a primitive cube root of one.
CUBE_ROOT_CELL = {
    "squared_lengths": [8.000000000000002 + 0j] * 6,
    "links": [
        1 + 2.7509237127789626e-16j,
        1 + 1.8073963429006066e-16j,
        -0.5000000000000002 + 0.8660254037844385j,
        1 - 1.337633413904918e-16j,
        -0.5000000000000002 - 0.8660254037844385j,
        1 - 1.0373931348654725e-16j],
}


def _fixture_cell(edge_squared=8.0):
    support = bp.monopole_support()
    assert [tuple(e) for e in support.edges] == list(PAIRS)
    return {"squared_lengths": [complex(edge_squared)] * 6,
            "links": [complex(support.transport(a, b)) for a, b in PAIRS]}


def _gauged(cell, gauge):
    return {"squared_lengths": list(cell["squared_lengths"]),
            "links": [gauge[a] * u / gauge[b]
                      for (a, b), u in zip(PAIRS, cell["links"])]}


def _relabeled(cell, permutation):
    """The cell with vertex v renamed permutation[v]."""
    squared, links = {}, {}
    for (a, b), z, u in zip(PAIRS, cell["squared_lengths"], cell["links"]):
        image = (permutation[a], permutation[b])
        key = (min(image), max(image))
        squared[key] = z
        links[key] = u if image[0] < image[1] else 1.0 / u
    return {"squared_lengths": [squared[e] for e in PAIRS],
            "links": [links[e] for e in PAIRS]}


def _scaled(cell, factor):
    return {"squared_lengths": [factor * z for z in cell["squared_lengths"]],
            "links": list(cell["links"])}


def _unit_gauge(seed):
    angles = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 4)
    return [cmath.exp(1j * a) for a in angles]


def _carrier(cell, sheets=1):
    """h_1 of the cell under the declared action at kappa = beta = 1, on one
    sheet (`build_base`) or on the three-sheeted host (`build_host`)."""
    spacetime = (bp.build_base(cell=cell) if sheets == 1
                 else bp.build_host(cell=cell))
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    return bp.matrix(action.carrier_operator())


def _spectrum(operator):
    return np.sort_complex(np.linalg.eigvals(operator))


def _relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(a)))


def _spin_frame_of(cell):
    """The spin frame the driver reads a cell in, with the cell's host and
    its sheets' supports."""
    spacetime = bp.build_host(cell=cell)
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports)
    return spacetime, supports, symmetry, bp.spin_frame(supports, symmetry)


def _averaged_spectrum(cell):
    """The spectrum of the T-averaged h_1 of a cell in the frame the driver
    reads it in: its eigenvalues are the one-particle energies the
    quasi-free poles are sums of."""
    _, _, _, frame = _spin_frame_of(cell)
    return _spectrum(bp.rotation_averaged(_carrier(cell, bp.SHEETS),
                                          frame["actions"]))


# ----------------------------------------------- the Drazin elimination


def _singular_symmetric(seed=3):
    """A complex symmetric matrix of order six with a null space of
    dimension two and index one: Q diag(d) Q^T with Q real orthogonal."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    d = np.array([0, 0, 1.5 + 0.2j, -0.7, 2.0 - 1j, 0.3 + 0.4j])
    return q @ np.diag(d) @ q.T


def test_the_drazin_inverse_satisfies_its_defining_identities():
    """A^D A A^D = A^D, A A^D = A^D A and, the index being one,
    A A^D A = A; the reduced coordinates rebuild A^D."""
    a = _singular_symmetric()
    drazin, basis, reduced, record = bp.drazin_elimination(a, None, 1e-15)
    assert record["null_dimension"] == 2
    assert record["eliminated_dimension"] == 4
    assert basis.shape == (6, 4) and reduced.shape == (4, 4)
    assert _relative(drazin, drazin @ a @ drazin) < 1e-13
    assert np.linalg.norm(a @ drazin - drazin @ a) < 1e-13
    assert _relative(a, a @ drazin @ a) < 1e-13
    assert record["drazin_identity_residual"] < 1e-13
    assert record["reduction_residual"] < 1e-13
    assert record["projector_idempotency"] < 1e-13


@pytest.mark.parametrize("unit", [1e-9, 1e9])
def test_the_null_space_of_a_stiffness_does_not_depend_on_its_unit(unit):
    """The null space is decided against the stiffness's own spectral
    radius, and the reduced coordinates carry the unit:
    R (R^T (c A) R)^-1 R^T = A^D / c."""
    a = _singular_symmetric()
    reference, _, _, _ = bp.drazin_elimination(a, None, 1e-15)
    _, basis, reduced, record = bp.drazin_elimination(unit * a, None, 1e-15)
    assert record["null_dimension"] == 2
    assert record["projector_idempotency"] < 1e-13
    rebuilt = basis @ np.linalg.solve(reduced, basis.T)
    assert _relative(reference, unit * rebuilt) < 1e-12


@pytest.mark.xfail(strict=True, reason=(
    "PencilSchur::feshbach forms the Drazin inverse as "
    "(P_II + Pi_0)^-1 (I - Pi_0): the projector is added to P_II without "
    "its unit, so the solve is conditioned by the unit of the stiffness. "
    "For a stiffness of size 1e-9 the inverse agrees with A^D / c to "
    "5.5e-8 and A A^D A - A is 2.2e-8 of A; for 1e9 the agreement is "
    "6.2e-7. The driver's induced displacement, one-body shift and "
    "constant are formed with this inverse"))
@pytest.mark.parametrize("unit", [1e-9, 1e9])
def test_the_drazin_inverse_carries_the_unit_of_the_stiffness(unit):
    """(c A)^D = A^D / c."""
    a = _singular_symmetric()
    reference, _, _, _ = bp.drazin_elimination(a, None, 1e-15)
    drazin, _, _, record = bp.drazin_elimination(unit * a, None, 1e-15)
    assert _relative(reference, unit * drazin) < 1e-12
    assert record["reduction_residual"] < 1e-12


def test_the_drazin_inverse_is_zero_on_a_nilpotent_block():
    """A stiffness with a complex symmetric nilpotent block N (N^2 = 0, N
    not zero) beside an invertible block B has index two: its Drazin inverse
    is zero on the whole generalized null space, the block of N, and B^-1 on
    the other. A^D A A^D = A^D holds; A A^D A = A does not, and the record's
    residual is the size of N in A."""
    rng = np.random.default_rng(3)
    nilpotent = np.array([[1, 1j], [1j, -1]], dtype=complex)
    block = np.array([[2.0, 0.5], [0.5, 1.0 + 1j]])
    core = np.zeros((4, 4), dtype=complex)
    core[:2, :2], core[2:, 2:] = nilpotent, block
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a = q @ core @ q.T
    expected = np.zeros((4, 4), dtype=complex)
    expected[2:, 2:] = np.linalg.inv(block)
    expected = q @ expected @ q.T
    drazin, _, _, record = bp.drazin_elimination(a, None, 1e-6)
    assert record["null_dimension"] == 2
    assert _relative(expected, drazin) < 1e-6
    assert _relative(drazin, drazin @ a @ drazin) < 1e-6
    measured = _relative(a, a @ drazin @ a)
    assert measured == pytest.approx(
        np.linalg.norm(nilpotent) / np.linalg.norm(core), rel=1e-6)
    assert record["drazin_identity_residual"] == pytest.approx(measured,
                                                               rel=1e-6)


def _wedge_lift(operator):
    """dGamma(X) on the two-particle space of a three-mode fiber, from the
    tensor form: X (x) 1 + 1 (x) X restricted to the antisymmetric vectors
    (e_i (x) e_j - e_j (x) e_i) / sqrt(2), i < j, in lexicographic order.
    An oracle independent of the library's occupation-number bookkeeping."""
    n = operator.shape[0]
    identity = np.eye(n)
    pairs = list(itertools.combinations(range(n), 2))
    basis = np.zeros((n * n, len(pairs)))
    for column, (i, j) in enumerate(pairs):
        basis[i * n + j, column] = 1.0 / np.sqrt(2.0)
        basis[j * n + i, column] = -1.0 / np.sqrt(2.0)
    return basis.T @ (np.kron(operator, identity)
                      + np.kron(identity, operator)) @ basis


def test_the_elimination_on_a_small_fiber_is_the_tensor_form():
    """Two particles in three modes with two fluctuations of nonsingular
    stiffness A: the eliminated operator is
    dGamma(h) - 1/2 sum_ab (A^-1)_ab dGamma(O_a) dGamma(O_b), with dGamma
    read from the antisymmetrized tensor product."""
    rng = np.random.default_rng(8)

    def random():
        return rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    carrier, couplings = random(), [random(), random()]
    stiffness = np.array([[2.0 + 0.3j, 0.4], [0.4, 1.5 - 0.2j]])
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = 3
    declaration.carrier = list(carrier.reshape(-1))
    declaration.couplings = [list(o.reshape(-1)) for o in couplings]
    declaration.bare_stiffness = list(stiffness.reshape(-1))
    declaration.occupied_modes = 2
    declaration.tolerance = 1e-15
    read = cob.DressedFluctuation(declaration).effective_action([], [], 2)
    assert int(read.dimension) == 3
    assert [list(b) for b in read.basis] == [[0, 1], [0, 2], [1, 2]]
    response = np.linalg.inv(stiffness)
    lifted = [_wedge_lift(o) for o in couplings]
    expected = _wedge_lift(carrier) - 0.5 * sum(
        response[a, b] * lifted[a] @ lifted[b]
        for a in range(2) for b in range(2))
    assert _relative(_wedge_lift(carrier),
                     np.asarray(read.one_body).reshape(3, 3)) < 1e-13
    assert _relative(expected,
                     np.asarray(read.effective_action).reshape(3, 3)) < 1e-13


# --------------------------------------------------- the poles of a sector


def _known_block(jordan):
    """An operator of order seven with an invariant subspace of dimension
    four on which its eigenvalues are 1 + i/2 twice, -2 and 3 - i, the
    double one a Jordan block of size two or diagonalizable; the operator
    and a basis of the subspace that is not its eigenbasis."""
    rng = np.random.default_rng(4)
    vectors = (np.eye(7) + 0.3 * rng.normal(size=(7, 7))
               + 0.2j * rng.normal(size=(7, 7)))
    core = np.diag([1.0 + 0.5j, 1.0 + 0.5j, -2.0, 3.0 - 1j, 5.0, 6.0,
                    7.0]).astype(complex)
    if jordan:
        core[0, 1] = 1.0
    operator = vectors @ core @ np.linalg.inv(vectors)
    sector = vectors[:, :4] @ (np.eye(4) + 0.1 * rng.normal(size=(4, 4)))
    return operator, sector


@pytest.mark.parametrize("jordan", [False, True])
@pytest.mark.parametrize("unit", [1.0, 1e-9, 1e9])
def test_the_poles_of_a_sector_are_the_eigenvalues_of_its_block(jordan,
                                                                unit):
    """`sector_poles` on an operator with a known invariant subspace: the
    compression leaks nothing, the poles are the block's eigenvalues, each
    a root of det(block - s), with multiplicities that add up to the
    sector's dimension, and they carry the unit of the operator: every
    decision of the read is relative to the block's scale."""
    operator, sector = _known_block(jordan)
    block, leakage, read = bp.sector_poles(unit * operator, sector, None)
    poles = np.array([complex(p) for p in read.poles]) / unit
    assert leakage < 1e-12
    assert list(read.failed_certificates) == []
    assert sum(int(m) for m in read.multiplicity) == sector.shape[1]
    expected = {-2.0 + 0j: 1, 1.0 + 0.5j: 2, 3.0 - 1j: 1}
    if jordan:
        # the two computed eigenvalues of a Jordan block of size two differ
        # by the square root of the rounding, and are two poles at the
        # declared pole tolerance 1e-15
        near = [p for p in poles if abs(p - (1.0 + 0.5j)) < 1e-6]
        assert len(near) in (1, 2)
        assert sum(int(m) for p, m in zip(poles, read.multiplicity)
                   if abs(p - (1.0 + 0.5j)) < 1e-6) == 2
    else:
        for value, count in expected.items():
            found = [int(m) for p, m in zip(poles, read.multiplicity)
                     if abs(p - value) < 1e-10]
            assert sum(found) == count
    for pole in poles:
        assert abs(np.linalg.det(block / unit - pole * np.eye(4))) < 1e-6


def _spin_block(states):
    """The block of J^2 on a set of states, in their basis, and how much of
    the image leaves their span."""
    basis = bp.occupation_basis()
    spins = bp.edge_spin_matrices()
    focks = np.column_stack([bp.to_fock(states[:, k], basis)
                             for k in range(states.shape[1])])
    images = np.column_stack([
        np.asarray(obs.SharpSpin.applyTotalSpinSquared(spins, focks[:, k]))
        for k in range(states.shape[1])])
    block = np.linalg.solve(focks.T @ focks, focks.T @ images)
    return block, float(np.linalg.norm(images - focks @ block))


def _half_sector():
    """The two colour-singlet states of spin 1/2 of the carrier content
    (0, 1, 2), with their bilinear dual: the block of J^2 on the content's
    singlet states is the polynomial projector (15/4 - B) / 3 onto them."""
    states, _ = bp.singlet_states((0, 1, 2))
    block, _ = _spin_block(states)
    projector = (bp.SPIN_THREE_HALVES * np.eye(len(block)) - block) / 3.0
    left, singular, _ = np.linalg.svd(projector)
    assert np.sum(singular > 0.5) == 2
    sector = states @ left[:, :2]
    return sector, bp.left_inverse(sector)


def _certificate_of(block):
    """The certificates `sector_entry` reads for the lowest pole of an
    operator whose compressed block on the spin-1/2 sector of `_half_sector`
    is ``block``: those of the pole's eigenspace, from the block's spectral
    projector onto it."""
    sector, dual = _half_sector()
    images = bp.sector_spin_images(sector, dual, bp.occupation_basis(),
                                   bp.edge_spin_matrices())
    _, _, read = bp.sector_poles(sector @ block @ dual, sector)
    (projector,) = bp.pole_projectors(read, 2)
    return bp.pole_certificates(projector, read.multiplicity[0],
                                bp.SPIN_HALF, sector, dual, images)


def test_the_certificates_of_a_double_pole_on_a_scalar_block():
    certificate = _certificate_of(2.0 * np.eye(2, dtype=complex))
    assert certificate["spin_lift_sharp"]
    assert certificate["spin_lift_expectation"] == pytest.approx(
        bp.SPIN_HALF, abs=1e-12)
    assert certificate["colour_casimir_residual"] < 1e-12


@pytest.mark.parametrize("entry", [(0, 1), (1, 0)])
def test_the_certificates_of_a_double_pole_pair_a_state_with_its_partner(
        entry):
    """The expectation of J^2 on the eigenspace of a double pole of the
    spin-1/2 sector is 3/4 on a block that is scalar up to one entry of
    1e-16, where an eigensolver returns two eigenvectors that are parallel
    to rounding: the certificate is read on the eigenspace, from the block's
    spectral projector, and no state is paired with a vector that is not its
    partner."""
    block = 2.0 * np.eye(2, dtype=complex)
    block[entry] = 1e-16
    certificate = _certificate_of(block)
    assert certificate["eigenspace_dimension"] == 2
    assert certificate["spin_lift_sharp"]
    assert certificate["spin_lift_expectation"] == pytest.approx(
        bp.SPIN_HALF, abs=1e-9)


# ------------------------------------- the polarization and the Ward read


def _polarization(unit):
    rng = np.random.default_rng(2)
    carrier = rng.normal(size=(4, 4))
    carrier = carrier + carrier.T
    couplings = [rng.normal(size=(4, 4)) for _ in range(2)]
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = 4
    declaration.carrier = list((unit * carrier).astype(complex).reshape(-1))
    declaration.couplings = [
        list((unit * (o + o.T)).astype(complex).reshape(-1))
        for o in couplings]
    declaration.occupied_modes = 2
    declaration.tolerance = 1e-15
    return np.asarray(cob.DressedFluctuation(declaration).paramagnetic(
        0j)).reshape(2, 2)


@pytest.mark.parametrize("unit", [1e3, 1e-3])
def test_the_polarization_carries_the_unit_of_the_carrier(unit):
    """Pi(0) of the carrier c h with the couplings c O_a is c Pi(0) of h and
    O_a: every term is O O / gap."""
    assert _relative(unit * _polarization(1.0), _polarization(unit)) < 1e-12


@pytest.mark.xfail(strict=True, reason=(
    "DressedFluctuation::paramagnetic refuses a particle-hole pair whose "
    "gap squared is at or below tolerance * max(1, gap^2 + omega^2): the "
    "floor of one is in the unit of the carrier, so a carrier of size 1e-8 "
    "or less (a cell whose squared lengths are 1e8 or more) has no "
    "polarization at the declared tolerance 1e-15, and its Ward read is "
    "unmeasured (48 of the 50 reads of tick 1 of the run of 2026-10-01)"))
def test_the_polarization_of_a_small_carrier_has_a_value():
    assert _relative(1e-9 * _polarization(1.0), _polarization(1e-9)) < 1e-12


def _ward_read(cell):
    spacetime = bp.build_host(cell=cell)
    config = bp.default_config([1.0], [1.0])
    carrier = _carrier(cell, bp.SHEETS)
    return bp.ward_read(spacetime, carrier,
                        bp.fluctuation_couplings(spacetime, True),
                        bp.gauge_directions(spacetime, True), config)


def test_the_ward_identity_holds_on_the_fixture():
    read = _ward_read(_fixture_cell())
    assert read["directions"] == 9
    assert read["residual"] < 1e-12
    assert read["paramagnetic_alone"] > 1e-2


@pytest.mark.xfail(strict=True, reason=(
    "the Ward read of a cell whose squared lengths are of order 1e9 is "
    "unmeasured: DressedFluctuation::paramagnetic compares the "
    "particle-hole gaps with a floor in the unit of the carrier (see "
    "test_the_polarization_of_a_small_carrier_has_a_value)"))
def test_the_ward_identity_is_read_on_a_dilated_cell():
    """The identity (D - Pi(0)) g = 0 is homogeneous in the carrier, so it
    is read on the fixture with its squared lengths multiplied by 1e9 as it
    is on the fixture."""
    read = _ward_read(_scaled(_fixture_cell(), 1e9))
    assert "unmeasured" not in read
    assert read["residual"] < 1e-12


# ------------------------------------------------------ ties and ratios


def test_two_poles_tie_at_the_declared_tolerance():
    assert bp._tied(2.0 + 0j, 2.0 + 0j)
    assert not bp._tied(2.0 + 0j, 2.0 + 1e-7 + 0j)


def test_ties_do_not_depend_on_the_unit_of_the_poles():
    for unit in (1e9, 1e-9):
        assert not bp._tied(unit * (2.0 + 0j), unit * (2.0 + 1e-7 + 0j))


def _record(content, doublet_content, sectors):
    out = {}
    for key, (value, irreps) in sectors.items():
        entry = {"lowest_pole": value, "poles": [value], "multiplicity": [1]}
        out[key] = {"quasi_free": entry, "with_quartic": dict(entry),
                    "restriction_to_2T": irreps,
                    "nucleon_reading": "2" in irreps,
                    "delta_reading": sorted(irreps) == ["2'", "2''"]}
    return {"content": list(content), "doublet_reads": [
        {"doublet_content": list(doublet_content), "sectors": out}]}


def _records():
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)
    return [
        _record([3, 0, 0], (0, 2, 1), {three: (-9.0 + 1j, ["2'", "2''"])}),
        _record([2, 1, 0], (1, 1, 1), {half: (-7.0 + 0j, ["2'"]),
                                       three: (-6.0 + 2j, ["2''", "2"])}),
        _record([1, 1, 1], (1, 1, 1), {half: (-5.0 + 0j, ["2"]),
                                       three: (-4.0 + 0j, ["2'", "2''"])}),
    ]


def test_the_ratios_name_the_pairs_whose_poles_they_divide():
    """By the 2T reading the nucleon is the lowest pole (smallest real
    part) of the sectors that restrict to a 2 and the Delta the lowest of
    those that restrict to 2' + 2''; by the spin of the lift they are the
    lowest spin-1/2 and spin-3/2 poles. The ratio is the quotient of the
    two poles named, as complex numbers, and the pairs are named with it."""
    out = bp.ratios(_records())["quasi_free"]
    reading = out["by_2T_reading"]
    assert reading["nucleon_content"] == [2, 1, 0]
    assert reading["nucleon_pole"] == -6.0 + 2j
    assert reading["nucleon_spin_j_j_plus_1"] == pytest.approx(3.75)
    assert reading["delta_content"] == [3, 0, 0]
    assert reading["delta_pole"] == -9.0 + 1j
    assert reading["pole_ratio"] == (-6.0 + 2j) / (-9.0 + 1j)
    assert reading["modulus_ratio"] == pytest.approx(
        abs(-6.0 + 2j) / abs(-9.0 + 1j))
    assert reading["real_part_ratio"] == pytest.approx(6.0 / 9.0)
    lift = out["by_spin_lift"]
    assert lift["nucleon_content"] == [2, 1, 0]
    assert lift["nucleon_pole"] == -7.0 + 0j
    assert lift["delta_content"] == [3, 0, 0]
    assert lift["delta_pole"] == -9.0 + 1j


def test_the_ratios_do_not_depend_on_the_order_of_the_contents():
    records = _records()
    reference = bp.ratios(records)
    for order in itertools.permutations(range(3)):
        out = bp.ratios([records[k] for k in order])
        for column in bp.COLUMNS:
            for pairing in ("by_2T_reading", "by_spin_lift"):
                for key in ("nucleon_pole", "delta_pole", "nucleon_content",
                            "delta_content", "pole_ratio"):
                    assert out[column][pairing][key] == \
                        reference[column][pairing][key]


def test_a_content_without_a_read_supplies_no_pole():
    """A content with no value has no doublet read; the ratios are those of
    the other contents, and with no nucleon reading left there is no ratio
    by the 2T reading."""
    records = _records()
    failed = {"content": [0, 0, 3], "failed": "no value",
              "doublet_reads": []}
    assert bp.ratios(records + [failed]) == bp.ratios(records)
    only_delta = [records[0], failed]
    out = bp.ratios(only_delta)["quasi_free"]
    assert out["by_2T_reading"] is None
    assert len(bp.pole_rows(only_delta)) == 2


# ------------------------------------------ singlet states and their spin


def _sheet_permutation(sigma):
    """Lambda^3 of the permutation of the sheets on the modes b * 3 + t,
    and its sign."""
    one = np.zeros((bp.SHEETS * bp.BASE_EDGES, bp.SHEETS * bp.BASE_EDGES))
    for b in range(bp.BASE_EDGES):
        for t in range(bp.SHEETS):
            one[b * bp.SHEETS + sigma[t], b * bp.SHEETS + t] = 1.0
    return bp.third_exterior_power(one), bp._permutation_sign(list(sigma))


@pytest.fixture(scope="module")
def sheet_permutations():
    return [_sheet_permutation(sigma)
            for sigma in ((1, 0, 2), (0, 2, 1), (1, 2, 0))]


#: The dimension of the spin-1/2 and of the spin-3/2 part of the
#: colour-singlet states of a content, by the multiset of its occupations:
#: three quarks in one doublet are a quartet, two in one doublet (a triplet)
#: and one in another are a doublet and a quartet, and one in each doublet
#: are two doublets and a quartet.
SPIN_DIMENSIONS = {(0, 0, 3): (0, 4), (0, 1, 2): (2, 4), (1, 1, 1): (4, 4)}


@pytest.mark.parametrize("content", bp.contents())
def test_a_singlet_state_takes_the_sign_of_a_permutation_of_the_sheets(
        content, sheet_permutations):
    states, _ = bp.singlet_states(content)
    for lifted, sign in sheet_permutations:
        assert np.linalg.norm(lifted @ states - sign * states) < 1e-12


@pytest.mark.parametrize("content", bp.contents())
def test_the_spin_of_the_singlet_states_is_one_half_or_three_halves(content):
    """J^2 maps the colour-singlet states of a content into their span, and
    its block B there satisfies (B - 3/4)(B - 15/4) = 0: its only
    eigenvalues are 3/4 and 15/4, with the multiplicities of the
    content."""
    states, _ = bp.singlet_states(content)
    half, three = SPIN_DIMENSIONS[tuple(sorted(content))]
    assert states.shape[1] == half + three
    block, leakage = _spin_block(states)
    identity = np.eye(len(block))
    assert leakage < 1e-12
    assert np.linalg.norm((block - bp.SPIN_HALF * identity)
                          @ (block - bp.SPIN_THREE_HALVES * identity)) < 1e-12
    assert np.trace(block).real == pytest.approx(
        bp.SPIN_HALF * half + bp.SPIN_THREE_HALVES * three, abs=1e-12)


def test_the_spin_sectors_account_for_every_singlet_state():
    for content in bp.contents():
        states, _ = bp.singlet_states(content)
        sectors, _ = bp.spin_sectors(states)
        half, three = SPIN_DIMENSIONS[tuple(sorted(content))]
        measured = (
            sectors[bp.SPIN_HALF].shape[1] if bp.SPIN_HALF in sectors else 0,
            sectors[bp.SPIN_THREE_HALVES].shape[1]
            if bp.SPIN_THREE_HALVES in sectors else 0)
        assert measured == (half, three), content


# ------------------------------------------------------- the T-average


@pytest.fixture(scope="module")
def fixture_frame():
    """The actions and the aligned frame of the fixture, the frame a cell
    without the tetrahedral group is read in."""
    host = bp.monopole_support()
    actions = bp.rotation_action([host] * bp.SHEETS)
    alignment = bp.aligned_doublet_frame(host, bp.rotation_group())
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    return actions, frame, np.linalg.inv(frame)


def test_the_rotation_average_is_the_projection_onto_the_commutant(
        fixture_frame):
    """The T-average of any operator commutes with every D_1(g), and
    averaging twice is averaging once."""
    actions, _, _ = fixture_frame
    rng = np.random.default_rng(6)
    operator = rng.normal(size=(18, 18)) + 1j * rng.normal(size=(18, 18))
    averaged = bp.rotation_averaged(operator, actions)
    assert _relative(averaged, bp.rotation_averaged(averaged, actions)) < 1e-13
    for action in actions:
        assert np.linalg.norm(action @ averaged - averaged @ action) \
            < 1e-13 * np.linalg.norm(averaged)


def test_the_many_body_average_of_a_one_body_operator_is_its_average(
        fixture_frame):
    """The average of dGamma(X) over the diagonal rotation action is dGamma
    of the T-averaged X, and the many-body average is a projection."""
    actions, frame, dual = fixture_frame
    rng = np.random.default_rng(7)
    operator = rng.normal(size=(18, 18)) + 1j * rng.normal(size=(18, 18))
    lifted = bp.second_quantized(dual @ operator @ frame)
    averaged = bp.rotation_averaged_many_body(lifted, actions, frame, dual)
    expected = bp.second_quantized(
        dual @ bp.rotation_averaged(operator, actions) @ frame)
    assert _relative(expected, averaged) < 1e-12
    assert _relative(averaged, bp.rotation_averaged_many_body(
        averaged, actions, frame, dual)) < 1e-12


def test_the_aligned_frame_is_the_same_on_every_call():
    host = bp.monopole_support()
    first = bp.aligned_doublet_frame(host, bp.rotation_group())
    second = bp.aligned_doublet_frame(host, bp.rotation_group())
    assert np.array_equal(first["frame"], second["frame"])
    assert first["trialities"] == second["trialities"]
    assert first["reference_carrier"] == second["reference_carrier"]


# ------------------------------- the unit, the gauge and the vertex labels


@pytest.mark.parametrize("factor", [1e9, 1e-6])
@pytest.mark.parametrize("cell", ["fixture", "dilated"])
def test_h1_carries_the_inverse_unit_of_the_squared_lengths(cell, factor):
    cell = _fixture_cell() if cell == "fixture" else DILATED_CELL
    assert _relative(_carrier(cell),
                     factor * _carrier(_scaled(cell, factor))) < 1e-12


@pytest.mark.parametrize("cell", ["fixture", "cube root", "dilated"])
def test_the_spectrum_of_h1_does_not_depend_on_the_gauge(cell):
    """Whitepaper v18, section 3: a gauge transformation, of unit modulus
    or not, is a similarity of h_1."""
    cell = {"fixture": _fixture_cell(), "cube root": CUBE_ROOT_CELL,
            "dilated": DILATED_CELL}[cell]
    reference = _spectrum(_carrier(cell))
    scale = np.max(np.abs(reference))
    unit = _spectrum(_carrier(_gauged(cell, _unit_gauge(21))))
    assert np.max(np.abs(unit - reference)) < 1e-12 * scale
    moduli = np.exp(np.random.default_rng(22).normal(size=4) * 0.5)
    whole = [m * g for m, g in zip(moduli, _unit_gauge(23))]
    other = _spectrum(_carrier(_gauged(cell, whole)))
    assert np.max(np.abs(other - reference)) < 1e-11 * scale


def test_the_spectrum_of_h1_of_the_fixture_does_not_depend_on_the_labels():
    reference = _spectrum(_carrier(_fixture_cell()))
    for permutation in itertools.permutations(range(4)):
        other = _spectrum(_carrier(_relabeled(_fixture_cell(),
                                              list(permutation))))
        assert np.max(np.abs(other - reference)) < 1e-12


@pytest.mark.xfail(strict=True, reason=(
    "h_1 dresses the mass matrix with the link between the lowest vertices "
    "of two cells (rsf_whitney_integration_spec, Definition 1: "
    "(M_k^U)_st = (M_k)_st U_{b(s) b(t)}, b(s) = min s). On a connection "
    "with curvature whose cell is not symmetric, renaming the vertices "
    "changes which vertex is lowest, and that is not a gauge "
    "transformation: the spectrum of h_1 of the tick-1 host cell changes "
    "by 5.6e-3, 1.5e-2 and 0.43 of its size under the three relabelings "
    "below, where quark condition 7 (whitepaper v18, section 10) asks for "
    "a fingerprint that is stable under vertex relabeling"))
def test_the_spectrum_of_h1_does_not_depend_on_the_labels_of_a_generic_cell():
    reference = _spectrum(_carrier(DILATED_CELL))
    scale = np.max(np.abs(reference))
    for permutation in ((1, 0, 2, 3), (1, 2, 0, 3), (3, 2, 1, 0)):
        other = _spectrum(_carrier(_relabeled(DILATED_CELL,
                                              list(permutation))))
        assert np.max(np.abs(other - reference)) < 1e-10 * scale


# ----------------------------------------------- the frame of the spin read


def test_the_fixture_is_read_in_its_own_frame():
    _, _, symmetry, frame = _spin_frame_of(_fixture_cell())
    assert symmetry["tetrahedral"]
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_CELL
    assert frame["flags"] == []


@pytest.mark.xfail(strict=True, reason=(
    "a gauge copy of unit modulus of the fixture has the twelve rotations "
    "as symmetries at the declared tolerance and is read in the fixture's "
    "frame all the same: MonopoleSupport::spinRead grades the bands of the "
    "rotation-averaged edge operator at 1e-15 (their degeneracy as an "
    "absolute gap, and the invariance, irreducibility and coexactness "
    "residuals of each band), the residuals of the copy are a few 1e-16 to "
    "1e-15, and no band is named the j = 1/2 doublet (fix(drivers): the "
    "spin and band reads group computed eigenvalues at 1e-15, below the "
    "eigensolver's rounding, "
    "https://github.com/akellehe/tessera/issues/1362)"))
def test_a_gauge_copy_of_the_fixture_is_read_in_its_own_frame():
    _, _, symmetry, frame = _spin_frame_of(
        _gauged(_fixture_cell(), _unit_gauge(11)))
    assert symmetry["tetrahedral"]
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_CELL


@pytest.mark.xfail(strict=True, reason=(
    "a cell without the tetrahedral group is read with the projective "
    "action D_1(g) of the fixture, which is built on the fixture's links "
    "in the fixture's gauge. A gauge transformation of the cell is a "
    "similarity of h_1 that the fixture's action does not follow, so the "
    "T-averaged operator, and with it every pole, depends on the gauge of "
    "the cell's links (theory(drivers): what read a relaxed cell has when "
    "its rotation group is a proper subgroup of T, "
    "https://github.com/akellehe/tessera/issues/1298)"))
@pytest.mark.parametrize("cell", ["cube root", "dilated"])
def test_the_averaged_spectrum_does_not_depend_on_the_gauge(cell):
    """The quasi-free poles are sums of three eigenvalues of the T-averaged
    h_1; whitepaper v18, section 3, says the spectral gates are gauge
    invariant."""
    cell = {"cube root": CUBE_ROOT_CELL, "dilated": DILATED_CELL}[cell]
    reference = _averaged_spectrum(cell)
    other = _averaged_spectrum(_gauged(cell, _unit_gauge(21)))
    assert np.max(np.abs(other - reference)) \
        < 1e-10 * np.max(np.abs(reference))


def test_the_reference_carrier_of_the_fixture_is_its_coexact_doublet():
    support = bp.monopole_support()
    alignment = bp.aligned_doublet_frame(support, bp.rotation_group())
    reference = alignment["reference_carrier"]
    block = alignment["frame"][:, 2 * reference:2 * reference + 2]
    coexact = np.asarray(support.coexactProjector(1e-12))
    assert np.linalg.norm(coexact @ block - block) \
        < 1e-10 * np.linalg.norm(block)


# ----------------------------------------------------- the bands of h_1


def test_the_bands_of_the_three_sheeted_fixture_have_rank_three():
    bands = bp.band_projectors(_carrier(_fixture_cell(), bp.SHEETS), 1e-15)
    assert [len(values) for values, _ in bands] == [3] * 6


def _three_levels(unit):
    """An operator with the eigenvalues 1, 1 + 1e-7 and 2 times ``unit``,
    in a fixed non-orthogonal eigenbasis."""
    rng = np.random.default_rng(9)
    vectors = np.eye(3) + 0.2 * rng.normal(size=(3, 3))
    return unit * (vectors @ np.diag([1.0, 1.0 + 1e-7, 2.0])
                   @ np.linalg.inv(vectors))


def test_three_separate_eigenvalues_are_three_bands():
    assert [len(values) for values, _ in
            bp.band_projectors(_three_levels(1.0), 1e-15)] == [1, 1, 1]
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [1.0]
    read = cob.BandFollower(declaration).read(
        list(_three_levels(1.0).astype(complex).reshape(-1)))
    assert list(read.ranks) == [1, 1, 1]


@pytest.mark.xfail(strict=True, reason=(
    "band_projectors takes its bands from BandFollower::read, which groups "
    "two eigenvalues when they differ by at most bandTolerance * "
    "max(1, |eigenvalue|): the floor of one is in the unit of the operator, "
    "so on an operator of size 1e-9 (h_1 of a cell whose squared lengths "
    "are 1e9) eigenvalues one part in 1e7 apart are one band at the band "
    "tolerance 1e-15 (fix(cobordism): the Regge sheet starts again at every "
    "scored point, and the multipliers and band grouping depend on the unit "
    "of length, https://github.com/akellehe/tessera/issues/1385)"))
def test_the_bands_do_not_depend_on_the_unit_of_the_operator():
    assert [len(values) for values, _ in
            bp.band_projectors(_three_levels(1e-9), 1e-15)] == [1, 1, 1]


@pytest.mark.xfail(strict=True, reason=(
    "BandFollower::read groups two eigenvalues when they differ by at most "
    "bandTolerance * max(1, |eigenvalue|): the floor of one is in the unit "
    "of the operator, so on an operator of size 1e-9 (h_1 of a cell whose "
    "squared lengths are 1e9) eigenvalues one part in 1e7 apart are one "
    "band at the band tolerance 1e-15 (fix(cobordism): the Regge sheet "
    "starts again at every scored point, and the multipliers and band "
    "grouping depend on the unit of length, "
    "https://github.com/akellehe/tessera/issues/1385)"))
def test_the_band_follower_does_not_depend_on_the_unit_of_the_operator():
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [1.0]
    read = cob.BandFollower(declaration).read(
        list(_three_levels(1e-9).astype(complex).reshape(-1)))
    assert list(read.ranks) == [1, 1, 1]


# ------------------------------------------- the measured quark evidences


def test_the_sheets_are_read_whatever_the_vertex_ids():
    """Three tetrahedra whose vertex ids interleave are three sheets, each
    a copy of the first under the ascending correspondence."""
    spacetime = T.Spacetime.fromVertexTuples(
        3, [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]], 1.0, 0.0)
    sheets = bp.sheets_of(spacetime)
    assert sheets["count"] == 3
    assert sheets["components"] == [[0, 3, 6, 9], [1, 4, 7, 10],
                                    [2, 5, 8, 11]]
    assert sheets["cells"] == [[4, 6, 4, 1]] * 3
    assert sheets["copies"]
    assert bp.sheet_count_evidence(spacetime).held


def test_the_occupations_of_the_sheets_add_up_to_the_quarks():
    """The covariance of the three lowest modes of the three-sheeted
    fixture is one mode on every sheet, the occupations add up to tr Gamma,
    and a gauge copy of the covariance has the same occupations."""
    host = bp.build_host()
    action = cob.JointAction(host, bp.action_declaration(host, 1.0, 1.0))
    gamma = bp.matrix(action.occupation_projector(3))
    sheets = bp.sheets_of(host)
    occupations = bp.sheet_occupations(host, gamma, sheets)
    assert sum(occupations) == pytest.approx(np.trace(gamma), abs=1e-12)
    assert sum(occupations).real == pytest.approx(3.0, abs=1e-12)
    assert max(abs(n - 1.0) for n in occupations) < 1e-12
    weights = np.exp(np.random.default_rng(10).normal(size=18)
                     + 1j * np.random.default_rng(11).normal(size=18))
    copy = np.diag(1.0 / weights) @ gamma @ np.diag(weights)
    assert np.allclose(bp.sheet_occupations(host, copy, sheets),
                       occupations, atol=1e-12)


def test_the_base_band_evidence_of_a_cell_is_read_on_the_cell():
    spacetime, supports, _, frame = _spin_frame_of(DILATED_CELL)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    own = supports[0][0].spinRead(bp.rotation_group(), 1e-15, 1e-15)
    assert not own.half_integer_doublet
    quark = bp.quark_conditions(spacetime, frame["alignments"],
                                {"refusal": "not read"}, 0.0, None)
    (two,) = [c for c in quark["conditions"] if c["number"] == 2]
    held = {e["name"]: e["held"] for e in two["evidence"]}
    assert not held["base-band-sector"]


def test_every_sheet_of_the_fixture_carries_one_quark():
    host = bp.build_host()
    action = cob.JointAction(host, bp.action_declaration(host, 1.0, 1.0))
    config = bp.default_config([1.0], [1.0])
    for content in bp.contents():
        read = cob.BandFollower(bp.mean_field_declaration(
            content, config)).read(action.carrier_operator())
        gamma = bp.matrix(read.covariance)
        occupations = bp.sheet_occupations(host, gamma, bp.sheets_of(host))
        assert max(abs(n - 1.0) for n in occupations) < 1e-12, content
        assert bp.occupation_parity_evidence(host, gamma, 1e-12).held


# ---------------------------------------------------- reads of the driver


def _zero_solve_config(content, cell, **declared):
    """The config of one content read at the cell as it stands: no update
    of the solve is taken and no Pachner move is tried."""
    config = bp.default_config(
        [1.0], [1.0], selected_contents=[content],
        limits={"iteration_limit": 0}, solve={"pachner_moves": False},
        **declared)
    config["host_cell"] = cell
    return config


def test_the_lengths_only_elimination_is_read_through_the_driver():
    config = _zero_solve_config((1, 1, 1), CUBE_ROOT_CELL,
                                elimination="lengths")
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == []


@pytest.fixture(scope="module")
def cube_root_operators():
    """The driver's steps on the cube-root cell for the content (1, 1, 1),
    from the solve (no update) to the eliminated operator, in the driver's
    order (`evaluate_content`): dGamma(h_1 + shift) and the eliminated
    operator on the three-particle space of the 18 modes of the frame."""
    content = (1, 1, 1)
    config = _zero_solve_config(content, CUBE_ROOT_CELL)
    spacetime, action, _, _ = bp.relax_content(content, 1.0, 1.0, config)
    carrier = bp.matrix(action.carrier_operator())
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    frame_read = bp.spin_frame(supports,
                               bp.cell_symmetry(spacetime, supports))
    frame = bp._micro_frame(frame_read["alignments"])
    dual = np.linalg.inv(frame)
    fluctuations = bp.eliminate_fluctuations(spacetime, action, carrier,
                                             1.0, 1.0, config)
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = bp.SHEETS * bp.BASE_EDGES
    declaration.carrier = list((carrier + fluctuations["shift"]).reshape(-1))
    declaration.couplings = [list(o.reshape(-1))
                             for o in fluctuations["reduced_couplings"]]
    declaration.bare_stiffness = list(
        fluctuations["reduced_stiffness"].reshape(-1))
    declaration.occupied_modes = 3
    declaration.tolerance = bp.declared_tolerance(config,
                                                  "fluctuation_tolerance")
    read = cob.DressedFluctuation(declaration).effective_action(
        list(frame.reshape(-1)), list(dual.reshape(-1)), 3)
    dimension = int(read.dimension)
    return {
        "bands": list(cob.BandFollower(bp.mean_field_declaration(
            content, config)).read(action.carrier_operator()).ranks),
        "one_body": np.asarray(read.one_body).reshape(dimension, dimension),
        "eliminated": np.asarray(read.effective_action).reshape(dimension,
                                                                dimension),
        "record": fluctuations["record"],
    }


def test_the_eliminated_operator_commutes_with_the_sheet_permutations(
        cube_root_operators, sheet_permutations):
    """The sheets are copies of one base cell and the content's covariance
    is the same on each, so a permutation of the sheets commutes with the
    one-body operator and with the eliminated operator of Section 7."""
    assert cube_root_operators["bands"][:3] == [3, 3, 3]
    for name in ("one_body", "eliminated"):
        operator = cube_root_operators[name]
        for lifted, _ in sheet_permutations:
            assert np.linalg.norm(lifted @ operator - operator @ lifted) \
                < 1e-12 * np.linalg.norm(operator), name


def test_the_elimination_of_the_cube_root_cell_is_certified(
        cube_root_operators):
    """Nine coexact phases are eliminated, the lengths and the pure-gauge
    phases are the null space, and the Ward identity is read."""
    record = cube_root_operators["record"]
    assert record["drazin"]["null_dimension"] == 27
    assert record["drazin"]["eliminated_dimension"] == 9
    assert record["drazin"]["drazin_identity_residual"] < 1e-12
    assert record["drazin"]["reduction_residual"] < 1e-12
    assert record["expectation_force_check"] < 1e-12
    assert "unmeasured" not in record["ward_identity"]
    assert record["ward_identity"]["residual"] < 1e-11
