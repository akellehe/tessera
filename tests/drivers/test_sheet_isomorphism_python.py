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
(`baryon_poles.build_host`) and relaxes them with `SelfConsistentMeanField`
around `HolomorphicRelaxation`. The certificate
(`SheetedSupport.certifyIsomorphism`) then compares the sheets' squared lengths
and connection values. The tests below establish, on the run's own host cells
(`_recursion_run_2026_09_23`):

* the certificate compares exactly the quantities the paper names and reports
  exactly the injected disagreement;
* the host is built exactly isomorphic, and the seeded covariance and the
  geometric Jacobian respect the sheet permutation;
* the relaxation carries the sheets as one shared base field
  (`baryon_poles.share_sheet_geometry`: six squared lengths and six links,
  written to every sheet, each driven by the sum of its three edges'
  equations), so identical sheets stay identical to the bit for every
  content. Relaxed as 36 separate coordinates, rounding asymmetries amplified
  by the ill-conditioned Newton step separated the sheets' links by O(1) in
  one step;
* the Newton solve decides its rank on the singular values relative to the
  largest, at the driver's declared tolerance, and reports the rank and the
  gap; a complete orthogonal decomposition deciding on its pivoted-QR
  diagonal kept rank 35 where the numerical rank is 33;
* under the user's ruling (a) the band-filling covariance is built from the
  Riesz bands of h_1 itself (WP v17 §7 line 262: Gamma* a projector onto modes
  of h(z*)), so it commutes with h_1, the Ward current of the seeded action is
  divergence-free and the band rule is covariant under a gauge transformation
  of one sheet. The bands of h_1 on the host are simple on each sheet, rank 3
  over the three sheets; a content names occupations of them. Built from the
  T-averaged operator instead, Gamma did not commute with h_1 (Ward
  divergence 2.1 to 3.4) and a gauge transformation of one sheet split the
  bands.
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
        action, bp.mean_field_declaration(content, config))
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
def test_the_seeded_covariance_fills_rank_three_bands_of_h(content):
    """On the exactly isomorphic host h_1 is h-bar_1-free: its six eigenvalues
    per sheet are simple, so its bands over the three sheets have rank 3 each
    (WP v17 §8: every band E-bar of the base becomes E-bar (x) C^3). Every
    declared content fits, the band-filling covariance is a function of h_1
    and commutes with it to rounding, it commutes with the sheet relabeling
    to rounding (1e-14), and its trace is the three quarks."""
    config = _cell_config(FIRST_CELL, content, newton=0, mean_field=1)
    spacetime, action, report = bp.relax_content(content, 1.0, 1.0, config)
    assert list(report.band_ranks) == [3] * 6
    gamma = np.asarray(report.covariance).reshape(18, 18)
    h = bp.matrix(action.carrier_operator())
    assert np.linalg.norm(gamma @ h - h @ gamma) < 1e-12 * np.linalg.norm(h)
    # the covariance is over the 18 edge cells in `getEdgeList()` order
    p = _sheet_permutation(spacetime)[:18, :18]
    assert np.max(np.abs(p @ gamma @ p.T - gamma)) < 1e-14
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("content", [(2, 1, 0), (0, 0, 3)])
def test_one_newton_step_keeps_the_sheets_identical(content):
    """WP v17 line 355: the shared geometry is a stationary sector of the
    sheet-permuting group and the sheet number is conserved under relaxation.
    From the run's first host cell, exactly isomorphic as built, one
    mean-field iteration with one Newton step leaves the three sheets
    identical to the bit."""
    config = _cell_config(FIRST_CELL, content, newton=1, mean_field=1)
    spacetime, _, _ = bp.relax_content(content, 1.0, 1.0, config)
    lengths, links = _sheets(spacetime)
    assert _largest_gap(lengths) == 0.0
    assert _largest_gap(links) == 0.0


@pytest.mark.parametrize("cell", [FIRST_CELL, SECOND_CELL])
@pytest.mark.parametrize("content", bp.contents())
def test_the_relaxed_sheets_are_bit_identical_for_every_content(cell,
                                                                content):
    """Three Newton steps in each of two mean-field iterations, on both host
    cells of the run and every content: the shared base field is written to
    every sheet, so the squared lengths and links of the three sheets agree
    to the bit and the certificate reports both residuals exactly zero."""
    config = _cell_config(cell, content, newton=3, mean_field=2)
    spacetime, _, _ = bp.relax_content(content, 1.0, 1.0, config)
    lengths, links = _sheets(spacetime)
    assert _largest_gap(lengths) == 0.0
    assert _largest_gap(links) == 0.0
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(lengths, links, 1e-8)
    assert read.squared_length_residual == 0.0
    assert read.connection_residual == 0.0


def test_the_shared_field_has_six_lengths_and_six_links():
    """`baryon_poles.share_sheet_geometry` declares one class per base edge:
    the relaxation has 12 variables, and every class holds one edge of each
    sheet on the same orientation."""
    config = _cell_config(FIRST_CELL, (2, 1, 0), newton=1, mean_field=0)
    spacetime = bp.build_host(8.0, config["host_cell"])
    classes, orientations = bp.sheet_edge_classes(spacetime)
    records = bp.edge_records(spacetime)
    for k in range(6):
        members = [r for r, c in zip(records, classes) if c == k]
        assert sorted(a // 4 for a, _ in members) == [0, 1, 2]
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    relaxation = cob.HolomorphicRelaxation(
        action, bp.share_sheet_geometry(bp.relaxation_declaration(config),
                                        spacetime))
    assert relaxation.variable_count() == 12
    assert set(orientations) <= {1, -1}


@pytest.mark.parametrize("content", [(2, 1, 0), (0, 0, 3)])
def test_the_newton_solve_uses_the_numerical_rank(content):
    """The step of `HolomorphicRelaxation` is the minimum-norm least-squares
    solution with singular values at or below rank_tolerance times the
    largest counted as zero. The driver declares 1e-10
    (`baryon_poles.DECLARED_RANK_TOLERANCE`), a factor fifty above the
    rounding floor of its real-axis-difference Jacobian at radius 1e-4. On the
    seeded action of the run's first host cell the 36-coordinate Jacobian has
    exactly three near-null directions below that threshold, so the rank the
    solve uses is the numerical rank 33 for both contents, and the step
    record reports it with a gap of more than six decades between the
    smallest retained and the largest discarded singular value."""
    spacetime, action, config = _seeded_action(FIRST_CELL, content)
    declaration = bp.relaxation_declaration(config)
    declaration.maximum_iterations = 1
    assert declaration.rank_tolerance == bp.DECLARED_RANK_TOLERANCE == 1e-10
    relaxation = cob.HolomorphicRelaxation(action, declaration)
    n = relaxation.variable_count()
    jacobian = np.asarray(relaxation.jacobian()).reshape(n, n)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    numerical_rank = int(np.sum(singular > 1e-10 * singular[0]))
    assert numerical_rank == 33
    step = relaxation.solve().steps[0]
    assert step.jacobian_rank == numerical_rank
    assert step.rank_tolerance == 1e-10
    assert step.largest_singular_value == pytest.approx(singular[0],
                                                        rel=1e-12)
    assert step.smallest_retained_singular_value == pytest.approx(
        singular[32], rel=1e-9)
    assert step.largest_discarded_singular_value == pytest.approx(
        singular[33], rel=1e-3)
    assert step.rank_gap > 1e6


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
    assert np.all(singular[-3:] < bp.DECLARED_RANK_TOLERANCE * singular[0])
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


def test_three_quarks_in_one_band_survive_the_relaxation():
    """A band of h_1 on the sheeted host has rank 3 (one simple mode per sheet
    times three sheets), so three quarks in the third band fit and stay
    fitting while the sheets stay identical; two Newton steps of one
    mean-field iteration on the run's first host cell complete with the band
    ranks [3, 3, 3, 3, 3, 3]. (With the bands read on the T-averaged operator
    and the sheets relaxed separately, this was the run's refusal "band 2 has
    rank 2 and cannot hold the declared occupation 3".)"""
    config = _cell_config(FIRST_CELL, (0, 0, 3), newton=2, mean_field=1)
    _, _, report = bp.relax_content((0, 0, 3), 1.0, 1.0, config)
    assert list(report.band_ranks) == [3] * 6


def _seeded_band_read(gauge_angle):
    config = _cell_config(FIRST_CELL, (2, 1, 0), newton=0, mean_field=1)
    spacetime = bp.build_host(8.0, config["host_cell"])
    if gauge_angle:
        _gauge_one_sheet(spacetime, 6, gauge_angle)
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    solve = cob.SelfConsistentMeanField(
        action, bp.mean_field_declaration((2, 1, 0), config))
    return spacetime, solve.solve()


def test_the_bands_survive_a_gauge_transformation_of_one_sheet():
    """A unit-modulus vertex gauge transformation (vertex 6, local vertex 2
    of sheet 1, angle 0.3) leaves every squared length and every face
    holonomy of the host unchanged. The band rule reads the bands of h_1,
    which the transformation conjugates, so the band ranks [3] * 6, the
    occupied eigenvalues and the occupied energy tr(Gamma h_1) are unchanged
    (to 1e-12)."""
    spacetime, report = _seeded_band_read(0.3)
    lengths, links = _sheets(spacetime)
    assert _largest_gap(lengths) == 0.0
    holonomies = [_face_holonomies(u) for u in links]
    assert _largest_gap(holonomies) < 1e-15
    _, reference = _seeded_band_read(0.0)
    assert list(report.band_ranks) == list(reference.band_ranks) == [3] * 6
    np.testing.assert_allclose(np.asarray(report.occupied_eigenvalues),
                               np.asarray(reference.occupied_eigenvalues),
                               atol=1e-12)
    assert complex(report.occupied_energy) == pytest.approx(
        complex(reference.occupied_energy), abs=1e-12)


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


# ------------------------------------------- gauge invariance at fixed Gamma


def test_uniform_filling_commutes_with_h():
    """Content (1, 1, 1) puts one quark in each of the three lowest rank-3
    bands of h_1, so the band-filling covariance is (1/3)(P_0 + P_1 + P_2):
    a function of h_1, trace 3, commuting with h_1 to rounding, and the Ward
    current of the seeded action is divergence-free to 1e-14."""
    _, action, _ = _seeded_action(FIRST_CELL, (1, 1, 1))
    gamma = np.asarray(action.declaration.covariance).reshape(18, 18)
    h = bp.matrix(action.carrier_operator())
    assert np.linalg.norm(gamma @ h - h @ gamma) < 1e-12 * np.linalg.norm(h)
    assert np.trace(gamma).real == pytest.approx(3.0, abs=1e-12)
    assert np.max(np.abs(np.asarray(action.ward_current_divergence()))) \
        < 1e-14


@pytest.mark.parametrize("content", [(2, 1, 0), (0, 0, 3)])
def test_the_seeded_action_satisfies_the_ward_identity(content):
    """`JointAction.ward_current_divergence`: the divergence of
    j = U dS/dU vanishes for every gauge-invariant term, and for the matter
    term when Gamma transforms with the operator, as a function of h does.
    The per-cell relaxation of the run starts from the band-filling Gamma of
    h_1 (ruling (a)); its Ward current is divergence-free to 1e-12."""
    _, action, _ = _seeded_action(FIRST_CELL, content)
    assert np.max(np.abs(np.asarray(action.ward_current_divergence()))) \
        < 1e-12
