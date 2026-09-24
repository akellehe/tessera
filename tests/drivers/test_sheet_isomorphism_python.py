# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sheet-isomorphism failure of quark condition 2 in the per-cell reads of
the recursion driver, pinned from the certificate down to the Newton solve.

Whitepaper v17 §8, "Sheet convention (adopted)" (line 355): a k-sheeted
support is k isomorphic copies of a base complex "with equal squared lengths
and equal connection values on corresponding edges"; "the shared geometry is a
stationary sector: the fixed locus of the sheet-permuting group S_k contains
stationary configurations by symmetric criticality", and in the literal case of
disjoint copies the sheet number is "conserved under relaxation".

The per-cell read (`recursion.cell_reads` -> `baryon_poles.scan_point` ->
`evaluate_content` -> `relax_content`) builds three disjoint copies of the cell
(`baryon_poles.build_host`), each edge of each copy carrying its own z_e and
U_e as the paper states, and relaxes all 36 coordinates with
`SelfConsistentMeanField` around `HolomorphicRelaxation`. The certificate
(`SheetedSupport.certifyIsomorphism`) then compares the sheets' squared lengths
and connection values. The tests below establish, on the run's own host cells
(`_recursion_run_2026_09_23`):

* the certificate compares exactly the quantities the paper names and reports
  exactly the injected disagreement;
* the host is built exactly isomorphic, and the seeded covariance and the
  geometric Jacobian respect the sheet permutation;
* the relaxation does not keep the sheets on the S_3-fixed locus: the
  band-filling covariance enters the Newton Jacobian with a rounding-level
  sheet asymmetry that the ill-conditioned solve amplifies, and on some
  contents the complete-orthogonal-decomposition rank decision keeps the
  near-null direction of two sheets and not of the third, so a single Newton
  step separates the sheets' links by O(1);
* the band rule reads its bands on the T-average under the declared projective
  action, which is not gauge covariant, so once the sheets' links differ by a
  gauge transformation the rank-6 colour bands split and the content's
  occupations are assigned to different bands: this is the run's refusal
  "band 2 has rank 2" and the start of the macroscopic separation that fails
  the fibre lift.
"""
import cmath
import math

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

FIRST_CELL = (0, 1, 2, 3)
SECOND_CELL = (0, 1, 3, 4)


# ------------------------------------------------------------------ helpers


def _cell_config(cell, content, newton, mean_field):
    """The configuration `recursion.cell_reads` hands `baryon_poles` for one
    host cell of the tick-0 level, with the iteration budgets reduced to the
    stated values (every other entry as in the run)."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(content)])
    config["host_cell"] = RUN.HOST_CELLS[cell]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    config["newton_iterations"] = newton
    config["mean_field_iterations"] = mean_field
    return config


def _declared_actions():
    return bp.rotation_action([bp.monopole_support()] * bp.SHEETS)


def _sheets(spacetime):
    lengths = [np.array(bp.sheet_squared_lengths(spacetime, t))
               for t in range(bp.SHEETS)]
    links = [np.array(bp.sheet_links(spacetime, t))
             for t in range(bp.SHEETS)]
    return lengths, links


def _largest_gap(vectors):
    return max(float(np.max(np.abs(vectors[s] - vectors[t])))
               for s in range(len(vectors)) for t in range(s + 1,
                                                           len(vectors)))


def _face_holonomies(links):
    fixture = obs.MonopoleSupport.tetrahedron(1)
    stored = {tuple(e): links[i] for i, e in enumerate(fixture.edges)}

    def oriented(a, b):
        return stored[(a, b)] if a < b else 1.0 / stored[(b, a)]

    return np.array([oriented(a, b) * oriented(b, c) * oriented(c, a)
                     for a, b, c in fixture.faces])


def _sheet_permutation(spacetime):
    """The permutation of the 36 relaxation coordinates (18 squared lengths,
    then 18 links, in `getEdgeList()` order) induced by the sheet relabeling
    t -> t + 1 mod 3."""
    records = bp.edge_records(spacetime)
    index = {r: i for i, r in enumerate(records)}
    edges = len(records)
    permutation = np.zeros((2 * edges, 2 * edges))
    for i, (a, b) in enumerate(records):
        image = ((a + 4) % 12, (b + 4) % 12)
        j = index[image] if image in index else index[image[::-1]]
        permutation[j, i] = 1.0
        permutation[edges + j, edges + i] = 1.0
    return permutation


def _gauge_one_sheet(spacetime, vertex, angle):
    """Multiply every link at ``vertex`` by exp(+-i angle): a unit-modulus
    vertex gauge transformation, which leaves every squared length and every
    face holonomy unchanged."""
    for edge in spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        if a == vertex:
            edge.setPhase(edge.getPhase() + angle)
        elif b == vertex:
            edge.setPhase(edge.getPhase() - angle)


def _seeded_action(cell, content):
    """The joint action of `relax_content` with the band-filling covariance
    seeded on the unrelaxed host, as `SelfConsistentMeanField.solve` seeds it
    before its first iteration."""
    config = _cell_config(cell, content, newton=40, mean_field=0)
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    solve = cob.SelfConsistentMeanField(
        action, bp.mean_field_declaration(content, config,
                                          _declared_actions()))
    solve.solve()
    return spacetime, solve.action, config


# ---------------------------------------------- the certificate (C++ core)


def _three_copies(cell=FIRST_CELL):
    host = RUN.HOST_CELLS[cell]
    return ([np.array(host["squared_lengths"]) for _ in range(3)],
            [np.array(host["links"]) for _ in range(3)])


def test_the_certificate_passes_three_exact_copies_with_zero_residuals():
    """Three copies of the run's first host cell (squared lengths
    8.000000000000002, the recorded unit-monopole links) agree exactly, so
    both residuals are exactly zero and the certificate holds at the driver's
    tolerance 1e-8 (WP v17 line 355: equal squared lengths and equal
    connection values on corresponding edges)."""
    lengths, links = _three_copies()
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    assert read.isomorphic
    assert read.squared_length_residual == 0.0
    assert read.connection_residual == 0.0
    assert read.sheet_count == 3 and read.base_cell_count == 6
    assert read.certificate.holds()


@pytest.mark.parametrize("sheet, edge, delta", [(2, 4, 3e-6), (1, 0, -7e-9),
                                                (0, 5, 2.5e-3)])
def test_the_certificate_reports_exactly_an_injected_length_gap(sheet, edge,
                                                                delta):
    """A squared length of one sheet moved by delta: the length residual is
    the stored difference |(z + delta) - z| exactly (the certificate is a
    maximum of absolute differences), the connection residual stays zero, and
    the certificate fails exactly when that difference exceeds the declared
    tolerance."""
    lengths, links = _three_copies()
    lengths[sheet][edge] += delta
    expected = abs(lengths[sheet][edge] - lengths[(sheet + 1) % 3][edge])
    support = obs.SheetedSupport(3, 6)
    read = support.certifyIsomorphism(lengths, links, 1e-8)
    assert read.squared_length_residual == expected
    assert read.squared_length_residual == pytest.approx(abs(delta), rel=1e-6)
    assert read.connection_residual == 0.0
    assert read.isomorphic == (expected <= 1e-8)
    assert support.certifyIsomorphism(lengths, links,
                                      2.0 * abs(delta)).isomorphic


@pytest.mark.parametrize("sheet, edge, angle", [(1, 2, 1e-5), (2, 1, 0.3)])
def test_the_certificate_reports_exactly_an_injected_connection_gap(sheet,
                                                                    edge,
                                                                    angle):
    """One link multiplied by exp(i angle): the connection residual is
    |U (e^{i angle} - 1)| = 2 |sin(angle / 2)| for a unit-modulus U, and the
    length residual stays zero."""
    lengths, links = _three_copies()
    links[sheet][edge] *= cmath.exp(1j * angle)
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    assert read.squared_length_residual == 0.0
    assert read.connection_residual == pytest.approx(
        2.0 * abs(math.sin(angle / 2.0)), rel=1e-9)
    assert not read.isomorphic


def test_the_certificate_takes_the_worst_pair_of_sheets():
    """Opposite gaps on two sheets at one edge (+2e-6 on sheet 1, -1e-6 on
    sheet 2): the residual is the disagreement of the worst pair, 3e-6, not
    the largest single displacement."""
    lengths, links = _three_copies()
    lengths[1][3] += 2e-6
    lengths[2][3] -= 1e-6
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    assert read.squared_length_residual == pytest.approx(3e-6, rel=1e-6)


def test_the_certificate_without_connections_grades_the_lengths_alone():
    """Empty connection vectors on every sheet declare a support with no
    connection values: the connection residual is exactly zero."""
    lengths, _ = _three_copies()
    lengths[0][0] += 1e-3
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(
        lengths, [np.array([], dtype=complex)] * 3, 1e-8)
    assert read.connection_residual == 0.0
    assert read.squared_length_residual == pytest.approx(1e-3, rel=1e-9)
    assert not read.isomorphic


def test_the_certificate_compares_connection_values_not_gauge_classes():
    """A unit-modulus gauge transformation of one sheet at local vertex 2 by
    0.3 changes the links of the three edges at that vertex by exp(+-0.3 i)
    and leaves every squared length and every face holonomy unchanged. The
    certificate follows the paper's literal definition (WP v17 line 355,
    "equal connection values on corresponding edges"), so it reports the
    connection residual 2 sin(0.15) = 0.29887626494719854 and fails. The
    tick-level check in `recursion.tick` compares face holonomies instead,
    so the two certify different notions of isomorphism."""
    lengths, links = _three_copies()
    fixture = obs.MonopoleSupport.tetrahedron(1)
    for i, (a, b) in enumerate(fixture.edges):
        if a == 2:
            links[1][i] *= cmath.exp(1j * 0.3)
        elif b == 2:
            links[1][i] *= cmath.exp(-1j * 0.3)
    assert np.max(np.abs(_face_holonomies(links[1])
                         - _face_holonomies(links[0]))) < 1e-15
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    assert read.squared_length_residual == 0.0
    assert read.connection_residual == pytest.approx(2.0 * math.sin(0.15),
                                                     rel=1e-12)
    assert not read.isomorphic


@pytest.mark.parametrize("tolerance", [0.0, -1e-8])
def test_the_certificate_refuses_a_non_positive_tolerance(tolerance):
    lengths, links = _three_copies()
    with pytest.raises(ValueError,
                       match="SheetedSupport::certifyIsomorphism tolerance"):
        obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, tolerance)


def test_the_certificate_refuses_the_wrong_number_of_sheets():
    lengths, links = _three_copies()
    with pytest.raises(ValueError, match="expected 3 squared-length vectors "
                                         "and 3 connection vectors"):
        obs.SheetedSupport(3, 6).certifyIsomorphism(lengths[:2], links[:2],
                                                    1e-8)


def test_the_certificate_refuses_a_sheet_of_the_wrong_size():
    lengths, links = _three_copies()
    lengths[1] = lengths[1][:5]
    with pytest.raises(ValueError, match="the squared-length vector of sheet "
                                         "1 has 5 entries; the base complex "
                                         "has 6 cells"):
        obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    lengths, links = _three_copies()
    links[2] = links[2][:4]
    with pytest.raises(ValueError, match="the connection vector of sheet 2 "
                                         "has 4 entries"):
        obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)


def test_the_certificate_refuses_connections_on_some_sheets_only():
    lengths, links = _three_copies()
    links[0] = np.array([], dtype=complex)
    with pytest.raises(ValueError, match="supplied for some sheets but not "
                                         "for all of them"):
        obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)


def test_a_support_with_no_sheets_is_refused():
    with pytest.raises(ValueError, match="the sheet count must be at least "
                                         "one"):
        obs.SheetedSupport(0, 6)


# ---------------------------------------------------- the host as built


@pytest.mark.parametrize("cell", [FIRST_CELL, SECOND_CELL])
def test_the_host_is_built_as_three_exact_copies_of_the_cell(cell):
    """`baryon_poles.build_host` with the run's host cell writes the cell's
    data on every sheet: the squared lengths round-trip through the stored
    length to within one unit in the last place of 8, the links through the
    stored phase to 1e-15, the three sheets are identical to the bit, and
    each sheet carries monopole number 1 (WP §10 condition 2)."""
    host = RUN.HOST_CELLS[cell]
    spacetime = bp.build_host(8.0, host)
    lengths, links = _sheets(spacetime)
    for t in range(3):
        assert np.max(np.abs(lengths[t] - np.array(host["squared_lengths"]))) \
            <= 4.0 * np.spacing(8.0)
        assert np.max(np.abs(links[t] - np.array(host["links"]))) < 1e-15
    assert _largest_gap(lengths) == 0.0
    assert _largest_gap(links) == 0.0
    for t in range(3):
        support, departure = bp.sheet_support(spacetime, t)
        assert support.monopoleNumber().monopole_number == 1
        assert departure < 1e-15


def test_the_level_zero_fields_are_the_recorded_ones():
    """`recursion.level_zero` on the declared two-tetrahedron fan reproduces
    the run's recorded tick-0 links to 1e-15 and the squared length 8 on every
    edge, with monopole number 1 in each tetrahedron and 2 through the
    bounding cut, which is the six recorded outward faces."""
    cells, z, links, connection = R.level_zero(R.default_config())
    assert cells == RUN.LEVEL_ZERO_CELLS
    assert connection["residual"] < 1e-14
    for edge, value in RUN.LEVEL_ZERO_LINKS.items():
        assert abs(links[edge] - value) < 1e-15
        assert z[edge] == 8.0
    assert R.monopole_numbers(cells, links) == [1, 1]
    cut = [list(f) for f in R.bounding_cut(cells)]
    assert cut == RUN.LEVEL_ZERO_BOUNDING_CUT
    assert R.cut_monopole_number(cut, links) == \
        RUN.LEVEL_ZERO_CUT_MONOPOLE_NUMBER


# --------------------------------------------- the relaxation's symmetry


def test_the_geometric_jacobian_is_exactly_sheet_symmetric():
    """With the matter term off, the Newton Jacobian of the joint action on
    the three-sheeted first host cell commutes with the sheet relabeling
    t -> t + 1 exactly (every entry of P J P^T - J is zero): the geometric
    part of the relaxation cannot separate the sheets."""
    config = _cell_config(FIRST_CELL, (2, 1, 0), newton=40, mean_field=0)
    spacetime = bp.build_host(8.0, config["host_cell"])
    action = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 1.0, 1.0, matter_weight=0.0))
    relaxation = cob.HolomorphicRelaxation(action,
                                           bp.relaxation_declaration(config))
    n = relaxation.variable_count()
    assert n == 36
    jacobian = np.asarray(relaxation.jacobian()).reshape(n, n)
    p = _sheet_permutation(spacetime)
    assert np.max(np.abs(p @ jacobian @ p.T - jacobian)) == 0.0


@pytest.mark.parametrize("content", [(2, 1, 0), (0, 0, 3), (1, 1, 1)])
def test_the_seeded_covariance_fills_rank_six_colour_bands(content):
    """On the exactly isomorphic host the T-averaged operator's bands are the
    three doublets times the three sheets, rank 6 each (WP v17 §8: every band
    E-bar of the base becomes E-bar (x) C^3), so every declared content fits,
    and the band-filling covariance commutes with the sheet relabeling to
    rounding (1e-15)."""
    config = _cell_config(FIRST_CELL, content, newton=0, mean_field=1)
    spacetime, _, report = bp.relax_content(content, 1.0, 1.0, config,
                                            _declared_actions())
    assert list(report.band_ranks) == [6, 6, 6]
    gamma = np.asarray(report.covariance).reshape(18, 18)
    # the covariance is over the 18 edge cells in `getEdgeList()` order
    p = _sheet_permutation(spacetime)[:18, :18]
    assert np.max(np.abs(p @ gamma @ p.T - gamma)) < 1e-14
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-12)


@pytest.mark.xfail(strict=True, reason=(
    "HolomorphicRelaxation inside SelfConsistentMeanField does not keep a "
    "three-sheeted support on the S_3-fixed locus: the band-filling "
    "covariance enters the finite-difference Jacobian with a sheet asymmetry "
    "of about 2e-12 (the geometric Jacobian's is exactly zero), and the "
    "ill-conditioned minimum-norm solve amplifies it; one Newton step from "
    "exactly identical sheets separates their squared lengths by 1.6e-9 on "
    "content (2, 1, 0) and their links by 0.60 on content (0, 0, 3)"))
@pytest.mark.parametrize("content", [(2, 1, 0), (0, 0, 3)])
def test_one_newton_step_keeps_the_sheets_identical(content):
    """WP v17 line 355: the shared geometry is a stationary sector of the
    sheet-permuting group and the sheet number is conserved under relaxation.
    From the run's first host cell, exactly isomorphic as built, one
    mean-field iteration with one Newton step must leave the three sheets
    identical to machine precision: squared lengths to 1e-13 of 8 and links
    to 1e-13."""
    config = _cell_config(FIRST_CELL, content, newton=1, mean_field=1)
    spacetime, _, _ = bp.relax_content(content, 1.0, 1.0, config,
                                       _declared_actions())
    lengths, links = _sheets(spacetime)
    assert _largest_gap(lengths) <= 8e-13
    assert _largest_gap(links) <= 1e-13


@pytest.mark.parametrize("content, xfail", [
    ((2, 1, 0), False),
    pytest.param((0, 0, 3), True, marks=pytest.mark.xfail(strict=True, reason=(
        "HolomorphicRelaxation's minimum-norm step uses "
        "Eigen::CompleteOrthogonalDecomposition with setThreshold(1e-12); its "
        "rank decision, taken on the pivoted-QR diagonal, keeps two of the "
        "three per-sheet near-null directions of the Jacobian (singular "
        "values 5.9e-12 to 9.5e-12 against a largest singular value near "
        "30) and reports rank 35 where the numerical rank at the same "
        "relative threshold is 33, so the step moves two sheets along "
        "directions the third does not follow")))])
def test_the_newton_solve_uses_the_numerical_rank(content, xfail):
    """The step of `HolomorphicRelaxation` is documented as the minimum-norm
    least-squares solution with singular values below rank_tolerance (1e-12,
    relative) counted as zero. On the seeded action of the run's first host
    cell the Jacobian has exactly three near-null directions, one supported on
    each sheet, below that threshold, so the rank the solve uses must be the
    numerical rank 33, the same number for every content."""
    spacetime, action, config = _seeded_action(FIRST_CELL, content)
    declaration = bp.relaxation_declaration(config)
    declaration.maximum_iterations = 1
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    n = relaxation.variable_count()
    jacobian = np.asarray(relaxation.jacobian()).reshape(n, n)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    numerical_rank = int(np.sum(singular > 1e-12 * singular[0]))
    assert numerical_rank == 33
    report = relaxation.solve()
    assert report.steps[0].jacobian_rank == numerical_rank


def test_the_near_null_directions_are_one_per_sheet():
    """The three near-null right singular vectors of the seeded Jacobian
    (content (0, 0, 3), first host cell) are each supported on one sheet
    (weight 1 on it to 1e-6) and lie in the link coordinates (length part
    below 1e-6): they are per-sheet link directions, which is why a rank
    decision that treats the sheets differently separates the links and not
    the lengths."""
    spacetime, action, config = _seeded_action(FIRST_CELL, (0, 0, 3))
    relaxation = cob.HolomorphicRelaxation(action,
                                           bp.relaxation_declaration(config))
    n = relaxation.variable_count()
    jacobian = np.asarray(relaxation.jacobian()).reshape(n, n)
    _, singular, vh = np.linalg.svd(jacobian)
    assert singular[-4] > 1e-6 * singular[0]
    assert np.all(singular[-3:] < 1e-12 * singular[0])
    records = bp.edge_records(spacetime)
    edges = len(records)
    supports = []
    for k in range(3):
        v = vh[-1 - k].conj()
        assert np.linalg.norm(v[:edges]) < 1e-6
        weights = [np.linalg.norm([v[edges + i] for i, r in enumerate(records)
                                   if r[0] // 4 == t]) for t in range(3)]
        sheet = int(np.argmax(weights))
        assert weights[sheet] == pytest.approx(1.0, abs=1e-6)
        supports.append(sheet)
    assert sorted(supports) == [0, 1, 2]


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "reproduces the run's refusal of contents (0, 0, 3), (0, 3, 0), (3, 0, 0) "
    "on host cell (0, 1, 2, 3): after the first Newton step separates the "
    "sheets' links by a gauge transformation (0.60, face holonomies equal to "
    "3.5e-9), the band rule's T-averaged operator is no longer degenerate "
    "across the sheets, the rank-6 colour band splits into rank-2 bands and "
    "SelfConsistentMeanField refuses 'band 2 has rank 2 and cannot hold the "
    "declared occupation 3'"))
def test_three_quarks_in_one_colour_band_survive_the_relaxation():
    """A colour band of the sheeted support has rank 6 (a doublet times three
    sheets, WP v17 §8), so three quarks in the highest band fit and stay
    fitting while the sheets stay identical; two Newton steps of one
    mean-field iteration on the run's first host cell must complete with the
    band ranks [6, 6, 6]."""
    config = _cell_config(FIRST_CELL, (0, 0, 3), newton=2, mean_field=1)
    _, _, report = bp.relax_content((0, 0, 3), 1.0, 1.0, config,
                                    _declared_actions())
    assert list(report.band_ranks) == [6, 6, 6]


@pytest.mark.xfail(strict=True, reason=(
    "the band rule of SelfConsistentMeanField reads its bands on "
    "(1/|T|) sum_g D(g)^-1 h D(g) with D the declared projective action of "
    "the symmetric monopole connection; that average is not covariant under "
    "a gauge transformation of the links, so gauge-transforming one sheet "
    "(lengths and face holonomies unchanged) splits the rank-6 colour bands "
    "into ranks [4, 2, 4, 2, 2, 4]; whether the band rule should be gauge "
    "covariant is a ruling for the user"))
def test_the_colour_bands_survive_a_gauge_transformation_of_one_sheet():
    """A unit-modulus vertex gauge transformation (vertex 6, local vertex 2
    of sheet 1, angle 0.3) leaves every squared length and every face
    holonomy of the host unchanged; the spectrum of a gauge-covariant band
    operator, and so the band ranks [6, 6, 6], must be unchanged."""
    config = _cell_config(FIRST_CELL, (2, 1, 0), newton=0, mean_field=1)
    spacetime = bp.build_host(8.0, config["host_cell"])
    _gauge_one_sheet(spacetime, 6, 0.3)
    lengths, links = _sheets(spacetime)
    assert _largest_gap(lengths) == 0.0
    holonomies = [_face_holonomies(u) for u in links]
    assert _largest_gap(holonomies) < 1e-15
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    solve = cob.SelfConsistentMeanField(
        action, bp.mean_field_declaration((2, 1, 0), config,
                                          _declared_actions()))
    report = solve.solve()
    assert list(report.band_ranks) == [6, 6, 6]


# ------------------------------------------------------------ fibre lift


def _fibre_lift_residual(spacetime):
    """The quantity `quark_conditions` grades as the fibre lift: the relative
    departure of the T-averaged h_1 from block-scalar form on each doublet
    times the sheets, in the aligned frame."""
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    frame = bp._micro_frame([alignment] * 3)
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    averaged = bp.rotation_averaged(bp.matrix(action.carrier_operator()),
                                    _declared_actions())
    return bp._block_scalar(np.linalg.inv(frame) @ averaged @ frame)[1]


def test_the_fibre_lift_holds_on_the_isomorphic_host():
    """On three identical copies of the run's first host cell the T-averaged
    operator is h-bar (x) I_3 (WP v17 §8 line 357), block-scalar on each
    doublet times the sheets to rounding (below 1e-14), which passes the
    driver's fibre-lift threshold 1e-6."""
    spacetime = bp.build_host(8.0, RUN.HOST_CELLS[FIRST_CELL])
    assert _fibre_lift_residual(spacetime) < 1e-14


def test_the_fibre_lift_fails_linearly_in_a_sheet_separation():
    """Scaling the squared lengths of sheet 1 by (1 + epsilon) makes the
    sheeted operator depart from h-bar (x) I_3 at first order in epsilon: the
    residual at epsilon = 2e-6 is twice the residual at 1e-6 to 1e-3, so the
    fibre lift fails exactly when the sheets' geometry separates, as it did
    in the eight reads of the run whose length residual reached 5e-4 or
    more."""
    residuals = []
    for epsilon in (1e-6, 2e-6):
        host = RUN.HOST_CELLS[FIRST_CELL]
        spacetime = bp.build_host(8.0, host)
        for edge in spacetime.getEdgeList().toVector():
            if int(edge.getSource().getId()) // 4 == 1:
                edge.setLength(edge.getLength() * cmath.sqrt(1.0 + epsilon))
        residuals.append(_fibre_lift_residual(spacetime))
    assert residuals[0] > 1e-8
    assert residuals[1] / residuals[0] == pytest.approx(2.0, rel=1e-3)


def test_the_recorded_failures_are_four_gauge_only_and_eight_geometric():
    """The run's condition-2 record (`_recursion_run_2026_09_23`): of the 16
    reads, 12 failed sheet isomorphism. In four, the sheets' face holonomies
    agree to 3e-9 and their squared lengths to 3e-8 while their links differ
    by 0.85 to 1.61: the sheets differ by a gauge transformation, and the
    fibre lift held. In the other eight the geometry separated (length
    residual 5.4e-4 to 1.7e4) and each of them failed the fibre lift."""
    rows = [row for cell in RUN.CONDITION_TWO_RECORD.values() for row in cell]
    failed = [row for row in rows if not row[1]]
    assert len(rows) == 16 and len(failed) == 12
    gauge_only = [row for row in failed if row[5] < 1e-8]
    geometric = [row for row in failed if row[5] >= 1e-8]
    assert len(gauge_only) == 4 and len(geometric) == 8
    assert all(row[2] < 3e-8 and row[3] > 0.8 and row[4]
               for row in gauge_only)
    assert all(row[2] >= 5e-4 and not row[4] for row in geometric)
