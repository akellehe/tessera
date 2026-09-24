# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The solvers and readers of the recursion driver's tick, held to known
stationary points, known roots and the run's own tick-0 record.

`recursion.tick` relaxes the level (`HolomorphicRelaxation`, with the bounding
cut's monopole sectors held), runs the Section 15 box (`LevelRecursion` over
the base operator, the partition persisting across the declared resolutions),
and grows the next level; every host cell is then relaxed with band filling
(`SelfConsistentMeanField`) and its poles read (`BoundStatePole`). Expected
values come from:

* stationary points known in closed form: l = l0 for the stiffness alone, a
  flat connection (every face holonomy 1) for the holonomy term alone;
* WP v17 §3 and §11.1: the held bounding cut keeps its monopole number and
  its unit-modulus face holonomies, the bulk face relaxes freely;
* WP v17 line 250: the band-filling covariance Gamma = sum_b (n_b / r_b) P_b,
  on the regular tetrahedron of squared length 8 whose Whitney edge operator
  has the bands 5 and 10 of rank 3 each (WP v17 line 263);
* pencils with known eigenvalues for the pole reader;
* the determinant factorization det R_l = det (R_l)_II det R_{l+1} of the
  level recursion (WP v17 line 193);
* the run of 2026-09-23 (`tests/drivers/_recursion_run_2026_09_23.py` and
  ``recursion.json``): its tick-0 level, partition, persistence, fibres,
  transport and rule shift, reproduced from the recorded fields.
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN


def _declaration(spacetime, stiffness=0.0, beta=0.0,
                 form=cob.HolonomyForm.Villain, reference=None):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 0.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.stiffness_weight = stiffness
    declaration.reference_lengths = (
        list(reference) if reference is not None else
        [complex(e.getLength()) for e in spacetime.getEdgeList().toVector()])
    declaration.holonomy_weight = beta
    declaration.holonomy_form = form
    declaration.matter_weight = 0.0
    return declaration


def _solve(lengths=True, links=True, iterations=30):
    solve = cob.HolomorphicRelaxationDeclaration()
    solve.relax_lengths = lengths
    solve.relax_links = links
    solve.relax_multipliers = False
    solve.maximum_iterations = iterations
    solve.tolerance = 1e-12
    return solve


def _tetrahedron(squared=8.0):
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


# ------------------------------------------------- HolomorphicRelaxation


def test_the_stiffness_alone_relaxes_to_the_reference_lengths():
    """With only (1/2)||l - l0||^2, l0 = sqrt 8 on every edge, the unique
    stationary point is l = l0: from squared lengths 8 + 0.3 k the Newton
    solve converges and returns every length to sqrt 8 within 1e-12."""
    spacetime = _tetrahedron()
    reference = [cmath.sqrt(8.0)] * 6
    for k, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(8.0 + 0.3 * (k + 1)))
    action = cob.JointAction(spacetime, _declaration(spacetime, 1.0,
                                                     reference=reference))
    report = cob.HolomorphicRelaxation(action, _solve(links=False)).solve()
    assert report.converged
    assert report.residual_norm < 1e-12
    for edge in spacetime.getEdgeList().toVector():
        assert abs(complex(edge.getLength()) - cmath.sqrt(8.0)) < 1e-12


@pytest.mark.parametrize("form", [cob.HolonomyForm.Villain,
                                  cob.HolonomyForm.Wilson])
def test_the_holonomy_alone_relaxes_to_a_flat_connection(form):
    """The holonomy term alone is stationary at a flat connection. From
    phases 0.1 sqrt(k + 1) (-1)^k on a single tetrahedron (face holonomies up
    to 0.62 rad from 1) the solve converges and every face holonomy is 1
    within 1e-14; the Villain solve needed no zero-guard damping."""
    spacetime = _tetrahedron()
    for k, edge in enumerate(spacetime.getEdgeList().toVector()):
        edge.setPhase(complex(0.1 * math.sqrt(k + 1) * (-1) ** k))
    action = cob.JointAction(spacetime, _declaration(spacetime, beta=1.0,
                                                     form=form))
    relaxation = cob.HolomorphicRelaxation(action, _solve(lengths=False))
    report = relaxation.solve()
    assert report.converged and report.zero_guard_damped_steps == 0
    faces = np.asarray(relaxation.action.face_holonomies())
    assert np.max(np.abs(faces - 1.0)) < 1e-14


def test_the_run_level_is_stationary_as_built():
    """The run's tick-0 level (the two-tetrahedron fan, squared length 8,
    the declared monopole connection) relaxes in zero Newton iterations with
    the recorded residual 1.18e-14, the Regge term structurally zero and the
    held cut's monopole number 2 on every sheet (run.log)."""
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    cut = R.bounding_cut(cells)
    number = R.cut_monopole_number(cut, links)
    record = R.relax_level(spacetime, config,
                           R.cut_sectors(cut, number, count))
    assert number == RUN.LEVEL_ZERO_CUT_MONOPOLE_NUMBER
    assert record["converged"] and record["iterations"] == 0
    assert record["residual"] == pytest.approx(1.1801457386692923e-14,
                                               rel=1e-6)
    assert record["sector_monopole_numbers"] == [2, 2, 2]
    assert record["regge_structurally_zero"]
    assert record["held_modulus_drift"] == 0.0


def test_the_held_cut_is_kept_and_the_bulk_face_relaxes():
    """WP v17 §3 and §11.1: on the run's level with the link (1, 3) turned
    by 0.05, the relaxation keeps the held cut's monopole number 2 on every
    sheet and the modulus of each of its six faces at 1 (drift below 1e-15),
    while the bulk face (0, 1, 3) that the two tetrahedra share is not held
    and relaxes from e^{0.05 i} to 1; strict emergence (no carried density)
    keeps the three sheets identical to 1e-14."""
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    links = dict(links)
    links[(1, 3)] *= cmath.exp(0.05j)
    spacetime, count = R.build_level(cells, z, links)
    cut = R.bounding_cut(cells)
    record = R.relax_level(spacetime, config,
                           R.cut_sectors(cut, 2, count))
    assert record["converged"] and record["iterations"] >= 1
    assert record["sector_monopole_numbers"] == [2, 2, 2]
    assert record["held_modulus_drift"] < 1e-15
    fields = R.sheet_fields(spacetime, count)
    for face in cut:
        assert abs(abs(R.face_holonomy(fields[0][1], face)) - 1.0) < 1e-14
    assert abs(R.face_holonomy(links, (0, 1, 3)) - cmath.exp(0.05j)) < 1e-15
    assert abs(R.face_holonomy(fields[0][1], (0, 1, 3)) - 1.0) < 1e-12
    for t in (1, 2):
        for edge in fields[0][1]:
            assert abs(fields[t][1][edge] - fields[0][1][edge]) < 1e-14
            assert abs(fields[t][0][edge] - fields[0][0][edge]) < 1e-14


def test_the_relaxation_refusals_are_named():
    spacetime = _tetrahedron()
    action = cob.JointAction(spacetime, _declaration(spacetime, beta=1.0))
    with pytest.raises(ValueError, match="no field is declared relaxable"):
        cob.HolomorphicRelaxation(action, _solve(False, False))
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    level, count = R.build_level(cells, z, links)
    held = R.cut_sectors(R.bounding_cut(cells), 1, count)
    declaration = bp.relaxation_declaration(dict(config, held_sectors=held))
    joint = cob.JointAction(level, bp.action_declaration(level, 1.0, 1.0))
    with pytest.raises(ValueError, match="held sector 0 is declared with "
                                         "monopole number 1 but the starting "
                                         "configuration carries 2"):
        cob.HolomorphicRelaxation(joint, declaration)


# ---------------------------------------------- SelfConsistentMeanField


def _band_filling(content, spacetime=None):
    spacetime = spacetime or _tetrahedron()
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    config = bp.default_config([1.0], [1.0])
    config["newton_iterations"] = 0
    config["mean_field_iterations"] = 1
    return action, cob.SelfConsistentMeanField(
        action, bp.mean_field_declaration(content, config))


def test_band_filling_is_the_weighted_sum_of_band_projectors():
    """WP v17 line 250: Gamma = sum_b (n_b / r_b) P_b. On the regular
    tetrahedron (squared length 8) the Whitney edge operator has the bands 5
    and 10, rank 3 each (WP v17 line 263); occupations (2, 1) give
    Gamma = (2/3) P_5 + (1/3) P_10, trace 3, to 1e-15."""
    action, solve = _band_filling((2, 1))
    report = solve.solve()
    assert list(report.band_ranks) == [3, 3]
    h = bp.matrix(action.carrier_operator())
    values, vectors = np.linalg.eigh(h)
    np.testing.assert_allclose(values, [5, 5, 5, 10, 10, 10], atol=1e-12)
    p5 = vectors[:, :3] @ vectors[:, :3].conj().T
    p10 = vectors[:, 3:] @ vectors[:, 3:].conj().T
    gamma = np.asarray(report.covariance).reshape(6, 6)
    assert np.max(np.abs(gamma - (2 / 3) * p5 - (1 / 3) * p10)) < 1e-15
    assert np.trace(gamma) == pytest.approx(3.0, abs=1e-14)


def test_band_filling_refusals_are_named():
    """A band of rank 3 cannot hold 4; occupations must be non-negative and
    not all zero; more occupations than bands are refused."""
    _, solve = _band_filling((4,))
    with pytest.raises(ValueError, match="band 0 has rank 3 and cannot hold "
                                         "the declared occupation 4"):
        solve.solve()
    with pytest.raises(ValueError, match="cannot be negative"):
        _band_filling((1, -1))
    with pytest.raises(ValueError, match="sum to zero"):
        _band_filling((0, 0))
    _, solve = _band_filling((1, 1, 1))
    with pytest.raises(ValueError, match="3 band occupations were declared "
                                         "but the spectrum groups into only 2 "
                                         "bands"):
        solve.solve()


# ------------------------------------------------------- BoundStatePole


def _poles(a, m, interface, centre, radius):
    return cob.BoundStatePole.poles(np.asarray(a, dtype=complex),
                                    np.asarray(m, dtype=complex), interface,
                                    centre, radius, cob.BoundStatePoleConfig())


def test_a_simple_pole():
    """diag(1, 3, 5) inside the circle of radius 1/2 about 1: one simple pole
    at 1, residue rank 1, no failed certificate."""
    read = _poles(np.diag([1, 3, 5]), np.eye(3), [0, 1, 2], 1.0, 0.5)
    assert len(read.poles) == 1
    assert complex(read.poles[0]) == pytest.approx(1.0, abs=1e-14)
    assert list(read.multiplicity) == [1] and list(read.simple) == [True]
    assert list(read.failed_certificates) == []


@pytest.mark.parametrize("a", [np.diag([2, 2, 5]),
                               [[2, 1, 0], [0, 2, 0], [0, 0, 5]]])
def test_a_double_pole_with_its_multiplicity(a):
    """A double eigenvalue 2, semisimple (diag(2, 2, 5)) or a Jordan block:
    one pole at 2 of multiplicity 2, not simple, residue rank 2 (the rank of
    the Riesz projector)."""
    read = _poles(a, np.eye(3), [0, 1, 2], 2.0, 1.0)
    assert len(read.poles) == 1
    assert complex(read.poles[0]) == pytest.approx(2.0, abs=1e-12)
    assert list(read.multiplicity) == [2]
    assert list(read.simple) == [False]
    assert list(read.residue_rank) == [2]


def test_a_generalized_pencil_and_a_feshbach_response():
    """The pencil (diag(2, 6), diag(1, 2)) has det(A - s M) = 0 at s = 2 and
    3; the response of [[1, 1], [1, 3]] onto coordinate 0,
    F(s) = 1 - s - 1 / (3 - s), vanishes at the eigenvalues 2 -+ sqrt 2, of
    which the circle of radius 1/2 about 1/2 encloses 2 - sqrt 2 and not the
    interior pole at 3."""
    read = _poles(np.diag([2, 6]), np.diag([1, 2]), [0, 1], 2.5, 1.0)
    np.testing.assert_allclose(sorted(complex(p).real for p in read.poles),
                               [2.0, 3.0], atol=1e-13)
    read = _poles([[1, 1], [1, 3]], np.eye(2), [0], 0.5, 0.5)
    assert len(read.poles) == 1
    assert complex(read.poles[0]) == pytest.approx(2 - math.sqrt(2),
                                                   abs=1e-13)
    assert list(read.failed_certificates) == []


def test_an_enclosed_interior_pole_is_refused_by_name():
    """The response of [[1, 1], [1, 1]] onto coordinate 0 has a pole at the
    interior eigenvalue 1; a contour enclosing it cannot count zeros, and the
    read names 'interior-pole-enclosed' and returns no pole."""
    read = _poles([[1, 1], [1, 1]], np.eye(2), [0], 1.0, 1.5)
    assert "interior-pole-enclosed" in list(read.failed_certificates)
    assert list(read.poles) == []


@pytest.mark.parametrize("separation", [
    1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10])
def test_a_near_degenerate_pair_is_read_inside_its_contour(separation):
    """diag(2, 2 + s, 5) inside the circle of radius 1 about 2 encloses
    exactly the two zeros 2 and 2 + s. Whatever the reader resolves, every
    pole it reports must lie inside its own contour, and the reported poles,
    counted with multiplicity, must have the zeros' mean 2 + s / 2 (the
    first moment of the argument principle) to 1e-6."""
    read = _poles(np.diag([2, 2 + separation, 5]), np.eye(3), [0, 1, 2],
                  2.0, 1.0)
    poles = [complex(p) for p in read.poles]
    assert all(abs(p - 2.0) < 1.0 for p in poles)
    total = sum(m * p for m, p in zip(read.multiplicity, poles))
    assert sum(read.multiplicity) == 2
    assert total / 2 == pytest.approx(2 + separation / 2, abs=1e-6)
    assert list(read.failed_certificates) == []


@pytest.mark.parametrize("separation", [1e-5, 1e-6, 1e-7])
def test_an_unresolved_pair_is_read_as_its_local_mean(separation):
    """A pair split below the Hankel pencil's resolution is one moment-method
    root; its small contour counts two zeros, and the reader reports one pole
    of multiplicity 2 at the mean of the pair, 2 + s / 2, to 1e-12, read as
    the first moment over the count on that contour, and runs no Newton step
    on it (NaN)."""
    read = _poles(np.diag([2, 2 + separation, 5]), np.eye(3), [0, 1, 2],
                  2.0, 1.0)
    assert list(read.multiplicity) == [2]
    assert complex(read.poles[0]) == pytest.approx(2 + separation / 2,
                                                   abs=1e-12)
    assert math.isnan(read.newton_step[0])


# ---------------------------------------------- the run's tick-0 recursion


@pytest.fixture(scope="module")
def tick_zero():
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    spacetime, count = R.build_level(cells, z, links)
    R.relax_level(spacetime, config,
                  R.cut_sectors(R.bounding_cut(cells), 2, count))
    base_z, base_links = R.sheet_fields(spacetime, count)[0]
    base = R.base_operator(cells, base_z, base_links)
    return config, cells, base_z, base_links, base


def test_the_tick_zero_partition_is_the_recorded_one(tick_zero):
    """The Section 15 box on the run's level reproduces the record: the
    partition of the nine edges into [2, 5, 7], [0, 1, 4], [3], [6], [8] at
    resolution 1, persistence [5, 5, 1, 1, 1] over the five resolutions,
    isolation gaps 6.562976567802366 and 3.4753767999083234 and 1, 1, 1,
    five accepted rank-1 bands and a zero determinant residual."""
    config, _, _, _, base = tick_zero
    level = R.recursion_turn(base["operator"], config)
    record = R.level_record(level)
    assert record["partition"] == [[2, 5, 7], [0, 1, 4], [3], [6], [8]]
    assert record["selected_resolution"] == 1.0
    assert record["component_persistence"] == [5.0, 5.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(record["isolation_gaps"],
                               [6.562976567802366, 3.4753767999083234,
                                1.0, 1.0, 1.0], rtol=1e-12)
    assert record["band_ranks"] == [1] * 5
    assert record["bands_accepted"] == [True] * 5
    assert record["determinant_residual"] == 0.0


@pytest.mark.parametrize("required, accepted", [(5, 2), (2, 2), (1, 5)])
def test_persistence_accepts_across_the_declared_window(tick_zero, required,
                                                        accepted):
    """WP v17 §5: a component becomes a response vertex only if it persists
    across the stated range of scales. With every resolution required (5)
    the run accepted the two three-edge components and rejected the edges
    0-4, 1-4 and 3-4 by name; with one resolution required all five pass
    (the issue #1249 small-case rerun)."""
    config, _, _, _, base = tick_zero
    level = R.recursion_turn(base["operator"], config)
    kept, rejected = R.persistent_components(level, base["edges"], required)
    assert len(kept) == accepted
    if required == 5:
        assert kept == [[2, 5, 7], [0, 1, 4]]
        assert [r["edges"] for r in rejected] == [["0-4"], ["1-4"], ["3-4"]]


def test_the_tick_zero_interaction_is_the_recorded_one(tick_zero):
    """The two response vertices interact (their images share a top
    simplex): one interaction (0, 1) with transport norm
    0.14606370183565728, no pair without a shared top simplex, and fibre
    restriction determinants 0.50623954329301 and 0.36852966165789997, all as
    recorded."""
    config, _, _, _, base = tick_zero
    stage = R.interaction_stage(base, [[2, 5, 7], [0, 1, 4]], config)
    assert sorted(stage["pairs"]) == [(0, 1)]
    assert float(np.linalg.norm(stage["transports"][(0, 1)])) == \
        pytest.approx(0.14606370183565728, rel=1e-12)
    assert stage["locality"]["pairs_without_a_shared_top_simplex"] == 0
    np.testing.assert_allclose(
        [complex(f["restriction_determinant"]).real for f in stage["fibers"]],
        [0.50623954329301, 0.36852966165789997], rtol=1e-12)
    assert stage["reads"] == []


def test_the_rule_shift_on_the_run_level(tick_zero):
    """The recorded rule shift of the level: relative shifts 0.568125 and
    0.675, row-sum defects 0.1573147725798841 and 0.20900111682582426."""
    _, cells, base_z, base_links, _ = tick_zero
    shift = R.level_rule_shift(cells, base_z, base_links)
    assert [s["cell"] for s in shift] == RUN.LEVEL_ZERO_CELLS
    np.testing.assert_allclose([s["relative_shift"] for s in shift],
                               [0.5681250000000001, 0.675], rtol=1e-12)
    np.testing.assert_allclose([s["row_sum_defect"] for s in shift],
                               [0.1573147725798841, 0.20900111682582426],
                               rtol=1e-10)


@pytest.mark.parametrize("lam", [0.3, 2.0 + 1.0j, -1.5, 7.2 - 0.4j])
def test_the_determinant_factorizes_at_several_lambda(tick_zero, lam):
    """WP v17 line 193: det R_0(lambda) = det (R_0)_II det R_1(lambda) at
    spectral parameters away from the ones the level was built at, with
    det R_0(lambda) = det(h_1 - lambda I) of the base operator itself."""
    config, _, _, _, base = tick_zero
    operator = base["operator"]
    declaration = cob.LevelRecursionDeclaration()
    declaration.resolutions = list(config["resolutions"])
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = 1
    declaration.bands = bands
    n = operator.shape[0]
    recursion = cob.LevelRecursion.overPencil(list(operator.reshape(-1)), [],
                                              n, declaration)
    recursion.advance()
    direct = np.linalg.det(operator - lam * np.eye(n))
    assert complex(recursion.response_determinant(0, lam)) == \
        pytest.approx(direct, rel=1e-12)
    product = complex(recursion.interior_determinant(0, lam)) * \
        complex(recursion.response_determinant(1, lam))
    assert product == pytest.approx(direct, rel=1e-11)
    assert recursion.determinant_factorization_residual(0, lam) < 1e-12


# ------------------------------------------------------- MappingCylinder


def _cylinder(incoming, reduction, outgoing):
    declaration = cob.MappingCylinderDeclaration()
    declaration.incoming_top_cells = [list(c) for c in incoming]
    declaration.reduction_map = dict(reduction)
    declaration.outgoing_top_cells = [list(c) for c in outgoing]
    return cob.MappingCylinder(declaration).read()


def test_the_mapping_cylinder_of_the_run_reduction():
    """WP v17 §3: one tick is the mapping cylinder W of the reduction map
    from K_l onto the response vertices. For the run's fan (five vertices)
    reduced onto response vertices 5 and 6 joined by an edge
    (0, 1, 2 -> 5 and 3, 4 -> 6): a four-dimensional cylinder with one fibre
    edge per incoming vertex, whose incoming end is the fan and whose
    outgoing end carries the image. The fan has a boundary, so the cylinder
    carries a side wall over it and its boundary is not the bare disjoint
    union K_l + K_{l+1}. Response vertices that reuse incoming identifiers
    are refused by name."""
    with pytest.raises(ValueError, match="the response vertex 0 is also an "
                                         "incoming vertex"):
        _cylinder(RUN.LEVEL_ZERO_CELLS, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                  [[0, 1]])
    read = _cylinder(RUN.LEVEL_ZERO_CELLS, {0: 5, 1: 5, 2: 5, 3: 6, 4: 6},
                     [[5, 6]])
    assert read.incoming_dimension == 3
    assert read.cylinder_dimension == 4
    assert list(read.incoming_vertices) == [0, 1, 2, 3, 4]
    assert list(read.response_vertices) == [5, 6]
    assert sorted(list(e) for e in read.fiber_edges) == [
        [0, 5], [1, 5], [2, 5], [3, 6], [4, 6]]
    assert read.incoming_boundary_is_the_incoming_complex
    assert read.outgoing_complex_contains_the_image
    assert not read.has_no_side_wall
    assert not read.boundary_is_the_disjoint_union
    assert read.boundary_residual > 0.0


def test_the_boundary_of_a_closed_level_is_the_disjoint_union():
    """WP v17 §3: dW = K_l disjoint-union K_{l+1}. For a closed level, the
    boundary of the 4-simplex (five tetrahedra, a 3-sphere), mapped
    identically onto a copy of itself, the cylinder has no side wall and its
    boundary is exactly the disjoint union, residual zero."""
    sphere = [[v for v in range(5) if v != k] for k in range(5)]
    image = [[v + 100 for v in cell] for cell in sphere]
    read = _cylinder(sphere, {v: v + 100 for v in range(5)}, image)
    assert read.cylinder_dimension == 4
    assert read.has_no_side_wall
    assert read.boundary_is_the_disjoint_union
    assert read.boundary_residual == 0.0
    assert read.certificate.holds()
