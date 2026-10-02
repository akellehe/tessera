# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The units, the declared tolerances and the declared limits of the isospin
and spinor reads.

Terms used below:

* a *declared tolerance* is a field or an argument the caller can set, which
  defaults to 1e-15;
* a *floor in the unit of the data* is a comparison of a quantity that
  carries the operator's unit with a fixed number (a tolerance times
  ``max(1, |h|)``, or an absolute tolerance): the read then changes when the
  operator is expressed in another unit. Every comparison here is taken
  relative to the operator's own scale, so a read of ``c * h`` is the read of
  ``h`` for every positive ``c``;
* a *declared limit* is a size the caller may declare an object must not
  exceed. None is declared as built, and a declared limit that is reached is
  named in the error.

The isospin fixture is the flavour doubling of the unit-monopole
tetrahedron's averaged edge Laplacian (`test_isospin_doublet_python`), with a
splitting ``eps * sigma_z`` of the two flavours added inside each band. Its
reads are made at tolerances of 1e-9, above the rounding of its twelve-fold
eigenvalues, so that what each test varies is the one field it is about.
"""
import itertools

import numpy as np
import pytest

import tessera
from tessera.drivers import baryon_poles as bp
from tests.observables import test_isospin_doublet_python as iso

obs = tessera.observables
MonopoleSupport = obs.MonopoleSupport
SharpSpin = obs.SharpSpin

Passed = obs.QuarkConditionStatus.Passed
Failed = obs.QuarkConditionStatus.Failed

#: The splitting of the two flavours inside a band, in units of the operator.
EPS = 1e-7
SCALES = (1e-9, 1.0, 1e9)


def loose(**fields):
    """The detector's configuration with the tolerances of the reads the
    fixture's rounding decides at 1e-9, the eigenvalues within 1e-5 of the
    operator's scale grouped into one band, and ``fields`` set on top."""
    config = obs.IsospinDoubletConfig()
    for name in ("projector_tolerance", "invariance_tolerance",
                 "commutant_tolerance", "isotypic_tolerance",
                 "hermiticity_tolerance", "transport_leakage_tolerance",
                 "intertwining_tolerance", "span_tolerance",
                 "transport_rank_tolerance",
                 "singular_value_grouping_tolerance", "occupation_tolerance"):
        setattr(config, name, 1e-9)
    config.grouping_tolerance = 1e-5
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def split_doublet(scale=1.0, **extra):
    """The declaration of the split flavour doubling in two frames and one
    further resolution, the whole of it multiplied by ``scale``."""
    base, actions, _ = iso.monopole_base()
    sigma = np.kron(np.kron(np.eye(6), np.diag([1.0, -1.0])), np.eye(3))
    op, sheet_of, base_of = iso.flavoured(base)
    op1, _, _ = iso.flavoured(base + 0.05 * iso.symmetric_deformation(actions))
    symmetry = iso.lifted(actions)
    frames = [scale * (op + EPS * sigma), scale * (op1 + EPS * sigma)]
    resolution = iso.padded_resolution(frames[0], sheet_of, base_of, symmetry,
                                       shift=50.0 * scale)
    declared = iso.declaration(frames, sheet_of, base_of, symmetry, True,
                               [resolution])
    for name, value in extra.items():
        setattr(declared, name, value)
    return declared


def nonnormal(scale=1.0):
    """A non-normal 5 x 5 operator with the eigenvalues 1, 1.9, 2.5, 6 and 7
    times ``scale``, and its eigenvalues, eigenvectors and their inverse
    from numpy."""
    rng = np.random.default_rng(3)
    s = np.eye(5) + 0.3 * rng.normal(size=(5, 5)) \
        + 0.2j * rng.normal(size=(5, 5))
    h = scale * (s @ np.diag([1.0, 1.9, 2.5, 6.0, 7.0]) @ np.linalg.inv(s))
    values, vectors = np.linalg.eig(h)
    return h, values, vectors, np.linalg.inv(vectors)


# ----------------------------------------------- the isospin read's fields


def test_the_new_tolerances_are_fields_that_default_to_1e_15():
    config = obs.IsospinDoubletConfig()
    for name in ("min_relative_gap", "span_tolerance",
                 "transport_rank_tolerance",
                 "singular_value_grouping_tolerance",
                 "member_splitting_tolerance", "occupation_tolerance"):
        assert getattr(config, name) == 1e-15, name
        setattr(config, name, 1e-7)
        assert getattr(config, name) == 1e-7


def test_the_cap_and_the_limits_are_not_declared_as_built():
    config = obs.IsospinDoubletConfig()
    assert config.condition_number_cap is None
    assert config.decomposed_rank_limit is None
    assert config.decomposed_commutant_limit is None


def test_the_band_projector_is_the_closed_form_of_the_eigendecomposition():
    """The Riesz projector of a band is V_b V_b^-1, the band's eigenvectors
    times the matching rows of the inverse eigenvector matrix. On the band
    {1, 1.9, 2.5} of a non-normal operator, whose isolating circle has
    radius 2.5 about 1.8, it agrees with numpy's to rounding and is
    idempotent to rounding whatever the number of points the resolvent norm
    is sampled at. (A trapezoid rule of the resolvent on that circle is
    4e-15 from it at 64 nodes, 4e-8 at 32, 2e-4 at 16 and 1e-2 at 8.)"""
    h, values, vectors, inverse = nonnormal()
    pick = [i for i in range(5) if values[i].real < 4.0]
    exact = vectors[:, pick] @ inverse[pick, :]
    reads = []
    for nodes in (64, 8):
        frame = obs.IsospinDoublet.bands(
            h, config=loose(grouping_tolerance=0.2, contour_nodes=nodes))
        assert [band.rank for band in frame.bands] == [3, 2]
        band = frame.bands[0]
        projector = np.asarray(band.fiber.projector())
        assert np.max(np.abs(projector - exact)) < 1e-12
        assert band.projector_residual < 1e-13
        assert band.projector_norm == pytest.approx(
            np.linalg.norm(exact, 2), rel=1e-10)
        assert band.contour_center == pytest.approx(1.8, abs=1e-12)
        assert band.contour_radius == pytest.approx(2.5, abs=1e-12)
        assert band.isolated
        reads.append((projector, band.resolvent_max))
    np.testing.assert_array_equal(reads[0][0], reads[1][0])
    # the sampled resolvent norm is the one number the node count moves
    assert reads[0][1] == pytest.approx(2.0893, abs=5e-4)
    assert reads[1][1] == pytest.approx(1.8674, abs=5e-4)


def test_the_frame_reports_the_condition_of_its_eigenbasis():
    """Every band projector of a frame is formed with the inverse of the
    operator's eigenvector matrix, whose reciprocal condition number the
    frame reports: one for a Hermitian operator, whose eigenvectors are
    orthonormal, and small where two eigenvectors are nearly parallel."""
    base, _, _ = iso.monopole_base()
    hermitian = obs.IsospinDoublet.bands(base, config=loose())
    assert hermitian.eigenbasis_reciprocal_condition == 1.0
    h, _, vectors, _ = nonnormal()
    frame = obs.IsospinDoublet.bands(h, config=loose())
    assert 0.0 < frame.eigenbasis_reciprocal_condition < 1.0
    near = np.array([[1.0, 1.0, 0.0], [0.0, 1.0 + 1e-6, 0.0],
                     [0.0, 0.0, 3.0]], dtype=complex)
    frame = obs.IsospinDoublet.bands(near, config=loose(
        grouping_tolerance=1e-9))
    assert 1e-8 < frame.eigenbasis_reciprocal_condition < 1e-5
    assert [band.rank for band in frame.bands] == [1, 1, 1]
    # the two nearly parallel modes' projectors are as large as the
    # reciprocal of the distance between their eigenvalues
    assert frame.bands[0].projector_norm > 1e5


@pytest.mark.parametrize("scale", SCALES)
def test_the_bands_of_an_operator_do_not_depend_on_its_unit(scale):
    """The bands of ``c * h`` are those of ``h``: the same ranks, verdicts
    and contents, and the same projectors."""
    reference = obs.IsospinDoublet.bands(nonnormal()[0], config=loose(
        grouping_tolerance=0.2))
    h = nonnormal(scale)[0]
    frame = obs.IsospinDoublet.bands(h, config=loose(grouping_tolerance=0.2))
    assert [b.rank for b in frame.bands] == [b.rank for b in reference.bands]
    assert [b.isolated for b in frame.bands] == \
        [b.isolated for b in reference.bands]
    assert [b.content for b in frame.bands] == \
        [b.content for b in reference.bands]
    for band, other in zip(frame.bands, reference.bands):
        np.testing.assert_allclose(np.asarray(band.fiber.projector()),
                                   np.asarray(other.fiber.projector()),
                                   atol=1e-10)
        assert band.gap == pytest.approx(scale * other.gap, rel=1e-9)
    assert frame.eigenbasis_reciprocal_condition == pytest.approx(
        reference.eigenbasis_reciprocal_condition, rel=1e-6)


@pytest.mark.parametrize("scale", (1e-13, 1.0, 1e9))
def test_the_hermitian_regime_is_read_relative_to_the_operator(scale):
    """An operator whose departure from its adjoint is 1e-3 of its size is
    not Hermitian in any unit, and one that equals its adjoint is."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hermitian = np.diag([1.0, 2.0, 4.0, 8.0]) + 0.1 * (x + x.conj().T)
    skew = hermitian + 1e-3 * (x - x.conj().T)
    for operator, expected in ((hermitian, True), (skew, False)):
        frame = obs.IsospinDoublet.bands(scale * operator, config=loose())
        assert [band.fiber.certificate().selfAdjoint
                for band in frame.bands] == [expected] * len(frame.bands)


@pytest.mark.parametrize("order", ("sheet-major", "base-major"))
@pytest.mark.parametrize("regime", ("non-normal", "hermitian"))
def test_the_copies_of_an_eigenvalue_on_equal_sheets_are_one_band(order,
                                                                  regime):
    """Three sheets carrying one operator are three equal blocks that no
    entry couples. Each block is decomposed on its own, so the three copies
    of an eigenvalue are equal bit for bit and form one band of rank three
    at the declared tolerances (1e-15), on which the sheets act exactly,
    whichever way the cells are ordered. (One decomposition of the whole
    matrix separates the copies by its rounding.)"""
    if regime == "non-normal":
        sheet = nonnormal()[0]
    else:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        sheet = np.diag([1.0, 2.0, 4.0, 7.0, 11.0]) + 0.1 * (x + x.conj().T)
    if order == "sheet-major":
        operator = np.kron(np.eye(3), sheet)
        sheet_of = [i // 5 for i in range(15)]
        base_of = [i % 5 for i in range(15)]
    else:
        operator = np.kron(sheet, np.eye(3))
        sheet_of = [i % 3 for i in range(15)]
        base_of = [i // 3 for i in range(15)]
    frame = obs.IsospinDoublet.bands(operator, sheet_of_cell=sheet_of,
                                     base_cell_of_cell=base_of)
    assert [band.rank for band in frame.bands] == [3] * 5
    expected = np.sort_complex(np.linalg.eigvals(sheet))
    for band, value in zip(frame.bands, expected):
        first, second, third = band.eigenvalues
        assert first == second == third
        assert first == pytest.approx(value, abs=1e-10)
        assert band.sheet_invariance_residual == 0.0
        assert band.colour_acts
    assert (frame.eigenbasis_reciprocal_condition == 1.0) == \
        (regime == "hermitian")


# ------------------------------------------------ the observed doublet


def _candidates(read):
    return [candidate for candidate in read.candidates if candidate.observed]


@pytest.mark.parametrize("scale", SCALES)
def test_the_member_splitting_is_read_relative_to_the_operator(scale):
    """The two flavours of every band are split by eps = 1e-7 of the
    operator. That splitting is the in-band operator's, in every unit of the
    operator: the three doublets are observed and each takes its members
    from the in-band operator."""
    read = obs.IsospinDoublet.observe(split_doublet(scale), loose())
    assert read.doublet_observed
    assert len(read.candidates) == 3
    for candidate in read.candidates:
        assert [c.status for c in candidate.conditions[:2]] == [Passed,
                                                                Passed]
        assert candidate.charges.member_source == "in-band operator"
        assert list(candidate.charges.isospin) == [0.5, -0.5]


def test_the_member_splitting_tolerance_decides_what_splits_the_members():
    """With the tolerance above the splitting's size relative to the operator
    (the traceless commutant part of the compression has norm 3.5e-7 against
    an operator of norm 25), the in-band operator does not split the members:
    a declared splitting operator does when one is given, and the declared
    trivialization otherwise."""
    config = loose(member_splitting_tolerance=1e-6)
    read = obs.IsospinDoublet.observe(split_doublet(), config)
    assert read.doublet_observed
    for candidate in read.candidates:
        assert candidate.charges.member_source == \
            "declared trivialization (cell-order seed)"
    sigma = np.kron(np.kron(np.eye(6), np.diag([1.0, -1.0])), np.eye(3))
    read = obs.IsospinDoublet.observe(
        split_doublet(member_splitting=sigma), config)
    for candidate in read.candidates:
        assert candidate.charges.member_source == \
            "declared splitting operator"
    # a declared splitting operator below the tolerance, relative to its own
    # size, does not split them either
    weak = np.eye(36) + 1e-9 * sigma
    read = obs.IsospinDoublet.observe(
        split_doublet(member_splitting=weak), config)
    for candidate in read.candidates:
        assert candidate.charges.member_source == \
            "declared trivialization (cell-order seed)"


def _three_quark_density(excess):
    """A one-body density with occupations 2 + excess of the upper member
    and 1 - excess of the lower one of the doublet at 4."""
    base, _, _ = iso.monopole_base()
    values, vectors = np.linalg.eigh(base)
    order = np.argsort(values)
    doublet = vectors[:, order[2:4]]
    band = doublet @ doublet.conj().T
    upper = np.kron(np.kron(band, np.diag([1.0, 0.0])), np.eye(3))
    lower = np.kron(np.kron(band, np.diag([0.0, 1.0])), np.eye(3))
    return ((2.0 + excess) / 6.0) * upper + ((1.0 - excess) / 6.0) * lower


def test_the_occupation_tolerance_decides_the_occupation_pattern():
    """The occupations of the two members of the doublet at 4 are
    2 + 1e-6 and 1 - 1e-6: within an occupation tolerance of 1e-4 the
    pattern is uud, and within 1e-9 it is neither."""
    density = _three_quark_density(1e-6)
    patterns = {}
    for tolerance in (1e-4, 1e-9):
        read = obs.IsospinDoublet.observe(
            split_doublet(three_quark_density=density),
            loose(occupation_tolerance=tolerance))
        candidate = read.candidates[1]
        occupations = np.asarray(candidate.charges.member_occupations)
        np.testing.assert_allclose(occupations, [2.0 + 1e-6, 1.0 - 1e-6],
                                   atol=1e-10)
        patterns[tolerance] = candidate.charges.occupation_pattern
    assert patterns[1e-4] == "uud"
    assert patterns[1e-9].startswith("neither uud nor udd")


def test_the_span_tolerance_is_the_rank_cut_of_the_carried_band():
    """A band is followed to a further resolution by the overlap of its
    prolonged span with the resolution's bands, the span's rank read at the
    span tolerance. The prolonged frame is orthonormal, so every tolerance
    below one keeps its twelve columns and the overlap is one; a cut above
    one keeps none, the overlap is zero and refinement persistence fails."""
    read = obs.IsospinDoublet.observe(split_doublet(), loose())
    for candidate in read.candidates:
        assert list(candidate.resolution_found) == [True]
        assert candidate.resolution_overlap[0] == pytest.approx(1.0,
                                                                abs=1e-9)
    read = obs.IsospinDoublet.observe(split_doublet(),
                                      loose(span_tolerance=2.0))
    assert not read.doublet_observed
    for candidate in read.candidates:
        assert list(candidate.resolution_overlap) == [0.0]
        assert candidate.conditions[0].status == Failed
        assert "refinement-persistence" in candidate.conditions[0].failing


def test_the_transport_tolerances_decide_the_transport_s_rank_and_groups():
    """The frame-to-frame transport of a doublet has twelve singular values
    equal to one to rounding. Grouped at 1e-9 of the largest they are one
    group, the flavour factor's two equal singular values; and the composed
    transport is full rank at every rank tolerance below one and is not at a
    cut above one."""
    read = obs.IsospinDoublet.observe(split_doublet(), loose())
    for candidate in read.candidates:
        (step,) = candidate.transports
        assert len(step.flavour_singular_values) == 2
        np.testing.assert_allclose(step.flavour_singular_values, 1.0,
                                   atol=1e-9)
        assert candidate.conditions[1].status == Passed
    read = obs.IsospinDoublet.observe(
        split_doublet(), loose(transport_rank_tolerance=2.0))
    assert not read.doublet_observed
    for candidate in read.candidates:
        assert candidate.conditions[1].status == Failed
        assert "transport-full-rank" in candidate.conditions[1].failing


def test_no_conditioning_uncertifies_a_band_unless_a_cap_is_declared():
    """The band {1, 1.9, 2.5} of the non-normal operator has a projector of
    norm 2.606. It is isolated with no cap declared and with a cap above
    that norm, and is not with a cap below it; the norm is reported either
    way."""
    h = nonnormal()[0]
    verdicts = {}
    for cap in (None, 3.0, 2.0):
        frame = obs.IsospinDoublet.bands(h, config=loose(
            grouping_tolerance=0.2, condition_number_cap=cap))
        band = frame.bands[0]
        assert band.projector_norm == pytest.approx(2.606360, abs=1e-5)
        verdicts[cap] = band.isolated
    assert verdicts == {None: True, 3.0: True, 2.0: False}


def test_a_band_is_decomposed_whatever_its_rank_unless_a_limit_is_declared():
    """A band of rank eleven on which nothing acts has the whole matrix
    algebra, of dimension 121, for its commutant: one isotype of
    multiplicity eleven. With a rank limit of ten declared its content is
    not read and the limit is named; with a commutant limit of one hundred
    declared the commutant's dimension is reported with the limit named and
    its isotypes are not read."""
    operator = np.diag([1.0] * 11 + [5.0]).astype(complex)
    frame = obs.IsospinDoublet.bands(operator, config=loose())
    band = frame.bands[0]
    assert band.rank == 11
    assert band.commutant_dimension == 121
    assert band.isotype_count == 1
    assert list(band.multiplicities) == [11]
    assert list(band.irreducible_dimensions) == [1]
    assert band.content == "1 x 1 sheet x 11"
    assert band.unexplained_multiplicity

    frame = obs.IsospinDoublet.bands(
        operator, config=loose(decomposed_rank_limit=10))
    band = frame.bands[0]
    assert band.commutant_dimension == 0 and band.content == ""
    assert "above the declared limit of 10" in band.classification
    assert "decomposedRankLimit" in band.classification
    # the band of rank one is within the limit and is read
    assert frame.bands[1].content == "1 x 1 sheet x 1"

    frame = obs.IsospinDoublet.bands(
        operator, config=loose(decomposed_commutant_limit=100))
    band = frame.bands[0]
    assert band.commutant_dimension == 121
    assert band.isotype_count == 0
    assert band.content == ("commutant of dimension 121, above the declared "
                            "limit of 100 (decomposedCommutantLimit)")


# ------------------------------------------------------- the spinor bands


def _supports():
    """The declared host, the relabeled host and one sheet of the refined
    support of the driver, each with its rotation group."""
    host = bp.build_host()
    permutation = list(bp.FINGERPRINT_RELABELING)
    relabeled, _ = bp.sheet_support(bp.relabeled_host(host, permutation), 0,
                                    1e-15)
    _, sheet_data = bp.refined_host(host)
    return {
        "declared": (bp.monopole_support(), bp.rotation_group()),
        "relabeled": (relabeled,
                      bp.conjugated_group(bp.rotation_group(), permutation)),
        "refined": (bp.refined_support(sheet_data[0], 1e-12),
                    bp.refined_rotation_group()),
    }


@pytest.mark.parametrize("name, dimensions, doublet", [
    ("declared", [2, 2, 2], 1),
    ("relabeled", [2, 2, 2], 1),
    ("refined", [2, 2, 2, 2, 2], 3),
])
def test_the_spinor_bands_are_the_exact_pairs_of_the_class(name, dimensions,
                                                           doublet):
    """Under the nontrivial class the eigenvalues of the rotation-averaged
    edge Laplacian are equal in consecutive pairs exactly, and each pair is
    one band at the declared degeneracy tolerance whatever the eigensolver's
    rounding separates its two values by: three bands of rank two on the
    declared and on the relabeled host, five on the refined support. They
    are the bands the driver's `symmetry_bands` reads. Graded at 1e-12,
    above the rounding of the certificates, the j = 1/2 doublet is the
    coexact one, at the place the driver's `reference_doublet` names."""
    support, group = _supports()[name]
    declared = support.spinRead(group)
    assert declared.cocycle.nontrivial
    assert [band.dimension for band in declared.bands] == dimensions

    graded = support.spinRead(group, 1e-15, 1e-12)
    assert [band.dimension for band in graded.bands] == dimensions
    driver = bp.symmetry_bands(support, group, True, 1e-15, 1e-12)
    assert [band["dimension"] for band in driver] == dimensions
    np.testing.assert_allclose([band.eigenvalue for band in graded.bands],
                               [band["eigenvalue"] for band in driver],
                               atol=1e-12)
    reference = bp.reference_doublet(driver)
    assert reference is not None
    assert graded.half_integer_doublet
    assert graded.doublet_index == doublet
    assert driver[doublet] is reference
    assert graded.bands[doublet].eigenvalue == pytest.approx(4.0, abs=1e-12)


@pytest.mark.parametrize("scale", SCALES)
def test_the_spinor_bands_do_not_depend_on_the_operator_s_unit(scale):
    """The degeneracy tolerance is taken relative to the operator's largest
    eigenvalue modulus, so the bands of ``c * h`` are those of ``h``. On the
    even-monopole tetrahedron (the trivial class, where a unit is one
    eigenvalue) the bands at a degeneracy tolerance of 1e-9 are the groups
    of numpy's eigenvalues within 1e-9 of the largest."""
    support = MonopoleSupport.tetrahedron(2)
    group = MonopoleSupport.tetrahedralRotations()
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    values = np.linalg.eigvalsh(averaged)
    width = 1e-9 * max(abs(values[0]), abs(values[-1]))
    expected, start = [], 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or values[stop] - values[start] > width:
            expected.append(stop - start)
            start = stop
    bands = support.spinorBands(scale * averaged, group, False, 1e-9, 1e-9)
    assert [band.dimension for band in bands] == expected
    np.testing.assert_allclose(
        [band.eigenvalue / scale for band in bands],
        [np.mean(values[sum(expected[:k]):sum(expected[:k + 1])])
         for k in range(len(expected))], atol=1e-9)
    for band in bands:
        assert band.hermitian and band.hermiticity_defect <= 1e-12


def test_an_operator_hermitian_to_rounding_is_read_and_says_so():
    """An operator whose departure from its adjoint is at rounding is read,
    and each band carries the departure relative to the operator's size and
    whether it is within the tolerance."""
    support = MonopoleSupport.tetrahedron(1)
    group = MonopoleSupport.tetrahedralRotations()
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    tilted = averaged.copy()
    tilted[0, 1] += 3e-14
    defect = np.max(np.abs(tilted - tilted.conj().T)) / np.max(np.abs(tilted))
    assert defect > 1e-15
    bands = support.spinorBands(tilted, group, True)
    assert [band.dimension for band in bands] == [2, 2, 2]
    for band in bands:
        assert band.hermiticity_defect == pytest.approx(defect, rel=1e-12)
        assert not band.hermitian
    for band in support.spinorBands(tilted, group, True, 1e-15, 1e-12):
        assert band.hermitian


@pytest.mark.parametrize("monopole", (0, 1, 2, 3))
def test_the_coexact_projector_reads_its_rank_relative_to_the_coboundary(
        monopole):
    """The coexact projector is the complement of the image of the twisted
    coboundary, whose rank is the number of its singular values above the
    tolerance times the largest."""
    support = MonopoleSupport.tetrahedron(monopole)
    coboundary = np.asarray(support.twistedCoboundary())
    left, singular, _ = np.linalg.svd(coboundary, full_matrices=False)
    for tolerance in (1e-15, 1e-9):
        keep = singular > tolerance * singular[0]
        expected = np.eye(6) - left[:, keep] @ left[:, keep].conj().T
        projector = np.asarray(support.coexactProjector(tolerance))
        np.testing.assert_allclose(projector, expected, atol=1e-12)
        assert np.max(np.abs(projector @ projector - projector)) < 1e-12
        assert round(np.trace(projector).real) == 6 - int(np.sum(keep))


# ------------------------------------------- the sharp-spin read's sizes


def test_a_determinant_is_the_basis_state_with_the_sign_of_the_order():
    """The determinant of modes listed in the order m_1, ..., m_n is
    a_{m_1}^dagger ... a_{m_n}^dagger on the vacuum: the Fock basis state
    with those modes occupied, with the sign of the permutation that sorts
    the list."""
    modes = (1, 3, 4, 6)
    index = sum(1 << mode for mode in modes)
    for order in itertools.permutations(modes):
        inversions = sum(1 for p, q in itertools.combinations(range(4), 2)
                         if order[p] > order[q])
        state = np.asarray(SharpSpin.determinant(list(order), 7))
        assert state.shape == (1 << 7,)
        assert np.count_nonzero(state) == 1
        assert state[index] == (-1.0) ** inversions


def test_no_mode_count_is_imposed_and_a_declared_limit_is_named():
    """A Fock vector is built for every mode count whose 2^M entries the
    index type counts (`kIndexableModes`); a mode limit the caller declares
    is named when it is reached. The one-particle spin matrices of thirteen
    carriers are 26 x 26."""
    assert SharpSpin.kIndexableModes == 62
    for name in ("kMaxStateModes", "kMaxDenseModes", "kMaxSectorPatterns"):
        assert not hasattr(SharpSpin, name)
    with pytest.raises(ValueError, match="the mode count must lie between "
                                         "one and 62"):
        SharpSpin.determinant([0], 63)
    with pytest.raises(ValueError, match="the mode count must lie between "
                                         "one and 62"):
        SharpSpin.determinant([], 0)
    with pytest.raises(ValueError, match="SharpSpin::determinant: the "
                                         "declared mode limit of 5 was "
                                         "reached: 6 were asked for"):
        SharpSpin.determinant([0, 2, 4], 6, modeLimit=5)
    assert np.count_nonzero(np.asarray(
        SharpSpin.determinant([0, 2, 4], 6, modeLimit=6))) == 1
    with pytest.raises(ValueError, match="the declared mode limit of 5 was "
                                         "reached"):
        SharpSpin.determinantSuperposition([[0], [1]], [1.0, 1.0], 6,
                                           modeLimit=5)
    matrices = SharpSpin.doubletSpinMatrices(13)
    assert all(np.asarray(m).shape == (26, 26) for m in matrices)
    three = SharpSpin.doubletSpinMatrices(3)
    assert np.asarray(SharpSpin.totalSpinSquaredMatrix(three)).shape == \
        (64, 64)
    with pytest.raises(ValueError, match="SharpSpin::totalSpinSquaredMatrix: "
                                         "the declared mode limit of 4 was "
                                         "reached"):
        SharpSpin.totalSpinSquaredMatrix(three, modeLimit=4)
    with pytest.raises(ValueError, match="the mode count must lie between "
                                         "one and 31"):
        SharpSpin.totalSpinSquaredMatrix(SharpSpin.doubletSpinMatrices(16))


def test_no_pattern_count_is_imposed_and_a_declared_limit_is_named():
    """The patterns of sixteen modes with eight particles are 12870; a
    pattern limit the caller declares is named when it is reached, by every
    function that forms the sector."""
    patterns = SharpSpin.sectorPatterns(16, 8)
    assert len(patterns) == 12870
    assert [list(p) for p in patterns[:2]] == [list(range(8)),
                                               list(range(7)) + [8]]
    assert len(SharpSpin.sectorPatterns(18, 3, patternLimit=816)) == 816
    message = "the declared pattern limit of 815 was reached"
    with pytest.raises(ValueError, match=message):
        SharpSpin.sectorPatterns(18, 3, patternLimit=815)
    identity = np.eye(18, dtype=complex)
    with pytest.raises(ValueError, match=message):
        SharpSpin.exteriorPowerMatrix(identity, 3, patternLimit=815)
    state = np.asarray(SharpSpin.determinant([0, 7, 17], 18))
    with pytest.raises(ValueError, match=message):
        SharpSpin.sectorComponent(state, 3, patternLimit=815)
    sector = np.asarray(SharpSpin.sectorComponent(state, 3))
    with pytest.raises(ValueError, match=message):
        SharpSpin.fockVector(sector, 18, 3, patternLimit=815)
    with pytest.raises(ValueError, match="the declared mode limit of 17 was "
                                         "reached"):
        SharpSpin.fockVector(sector, 18, 3, modeLimit=17)
    np.testing.assert_array_equal(
        np.asarray(SharpSpin.fockVector(sector, 18, 3)), state)
    with pytest.raises(ValueError, match=message):
        SharpSpin.isotypicProjector([identity], [1.0 + 0j], 1, 3,
                                    patternLimit=815)
