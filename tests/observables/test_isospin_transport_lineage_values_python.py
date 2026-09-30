# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The isospin-doublet detector on the recursion driver's host, complex
transport on constructed rotations of a line, and the lineage number on
hand-built cuts of the run's own level.

`recursion.cell_reads` sets ``isospin_doublet`` on every host cell, so each
content of each read runs `isospin_doublet.observe_host`
(`IsospinDoublet.observe` on h_1(z, U) and on its T-average). Transport and
lineage enter the quark verdict as conditions 5 and 6 (whitepaper v17 §10) and
the recursion's transports; the driver leaves the lineage unmeasured on a
single level, so it is tested here on the run's two-tetrahedron fan reduced
onto two response vertices, the partition the run accepted.

Expected values: the averaged edge spectrum {4 - 2/sqrt 3, 4, 4 + 2/sqrt 3}
of WP v17 line 506; for transport, the rotation R(theta) of the plane carries
the line at angle 0 to the line at angle theta, and the three Kato schemes of
`KatoScheme` evaluate on that pair to R(theta) (direct rotation),
cos(theta) R(theta) (intertwiner) and R(sin(2 theta) / 2) (the exponential
of [P1, P0]), and a loop of lines through half a turn closes with holonomy
-1; for the lineage, the intersection pairing of WP v17 §3 counts signed
crossings.
"""
import math

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp
from tessera.drivers import isospin_doublet as iso

SQRT3 = math.sqrt(3.0)
CT = obs.ComplexTransport
KS = obs.KatoScheme
CL = obs.ClusterLineage


# ------------------------------------------------------ isospin doublet


@pytest.fixture(scope="module")
def host_read():
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0,
                                                              1.0))
    return iso.observe_host(bp.matrix(action.carrier_operator()))


@pytest.mark.parametrize("operator", ["covariant", "t_averaged"])
def test_no_isospin_doublet_on_the_monopole_host(host_read, operator):
    """On the declared three-sheeted unit-monopole host (the run's per-cell
    host at the symmetric connection) no flavour doublet emerges on h_1 or
    on its T-average: no candidate, falsifier 8 ("No isospin doublet") holds
    on the read, and condition 1 (emergence) fails on
    'two-dimensional-flavour-band', as recorded for every content of the
    2026-09-23 run."""
    read = host_read[operator]
    assert read["candidates"] == []
    assert read["doublet_observed"] is False
    assert read["falsifier_8_no_isospin_doublet"] is True
    emergence = read["conditions"][0]
    assert emergence["name"] == "emergence"
    assert emergence["status"] == "Failed"
    assert emergence["failing"] == ["two-dimensional-flavour-band"]


def test_the_averaged_host_bands_are_colour_times_spin_doublets(host_read):
    """The T-averaged operator's bands on the host are the three doublets
    times the three sheets: rank 6 each, colour acting, spin doublets, not
    flavour candidates (WP v17 §8 and line 506)."""
    bands = host_read["t_averaged"]["frames"][0]["bands"]
    assert [b["rank"] for b in bands] == [6, 6, 6]
    for band in bands:
        assert band["colour_acts"] and band["spin_doublet"]
        assert not band["doublet_candidate"]
        assert band["content"] == "2 x 3 sheets x 1"


def test_a_constructed_flavour_doubling_is_found_at_the_averaged_values():
    """h-bar (x) I_2 (x) I_3, with h-bar the T-averaged edge Laplacian of the
    unit monopole, carries a genuine two-dimensional flavour space on each
    band: the detector returns three rank-12 candidate bands centred on
    4 - 2/sqrt 3, 4 and 4 + 2/sqrt 3 (WP v17 line 506)."""
    support = obs.MonopoleSupport.tetrahedron(1)
    group = obs.MonopoleSupport.tetrahedralRotations()
    base = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    op = np.kron(np.kron(base, np.eye(2)), np.eye(3))
    n = op.shape[0]
    actions = [np.kron(np.kron(np.asarray(support.edgeRepresentation(g)),
                               np.eye(2)), np.eye(3)) for g in group]
    frame = obs.IsospinDoublet.bands(
        op, sheet_of_cell=[i % 3 for i in range(n)],
        base_cell_of_cell=[i // 3 for i in range(n)], symmetry=actions,
        spinorial=True)
    assert [b.rank for b in frame.bands] == [12, 12, 12]
    assert all(b.doublet_candidate for b in frame.bands)
    np.testing.assert_allclose(sorted(b.center.real for b in frame.bands),
                               [4 - 2 / SQRT3, 4.0, 4 + 2 / SQRT3],
                               atol=1e-12)


# ------------------------------------------------------ complex transport


def _line(theta):
    v = np.array([math.cos(theta), math.sin(theta)])
    return np.outer(v, v).astype(complex)


def _rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


@pytest.mark.parametrize("theta", [0.1, 0.7, 1.2])
def test_the_kato_schemes_on_a_rotated_line(theta):
    """For P0 the line at angle 0 and P1 the line at angle theta: the direct
    rotation (I - (P1 - P0)^2)^-1/2 (P1 P0 + (I - P1)(I - P0)) is R(theta);
    the intertwiner P1 P0 + (I - P1)(I - P0) is cos(theta) R(theta); the
    exponential exp([P1, P0]) is R(sin(2 theta) / 2), since
    [P1, P0] = sin(theta) cos(theta) [[0, -1], [1, 0]]."""
    p0, p1 = _line(0.0), _line(theta)
    direct = np.asarray(CT.katoStep(p0, p1, KS.DirectRotation))
    assert np.max(np.abs(direct - _rotation(theta))) < 1e-14
    intertwiner = np.asarray(CT.katoStep(p0, p1, KS.Intertwiner))
    assert np.max(np.abs(intertwiner - math.cos(theta) * _rotation(theta))) \
        < 1e-15
    exponential = np.asarray(CT.katoStep(p0, p1, KS.ExponentialGenerator))
    assert np.max(np.abs(exponential
                         - _rotation(math.sin(2 * theta) / 2))) < 1e-14
    generator = np.asarray(CT.katoGenerator(p1 - p0, p0))
    assert np.max(np.abs(generator - ((p1 - p0) @ p0 - p0 @ (p1 - p0)))) \
        == 0.0


def test_a_half_turn_of_lines_closes_with_holonomy_minus_one():
    """Lines at angles k pi / 16, k = 0 .. 16: the path of projectors is
    closed (the line at pi is the line at 0), and the Kato transport by
    direct rotations is R(pi) = -I, the holonomy -1 of the real line bundle
    over the circle of lines, with every step intertwining exactly."""
    path = [_line(k * math.pi / 16) for k in range(17)]
    read = CT.katoTransport(path, KS.DirectRotation)
    assert read.complete
    assert read.steps == 16 and read.rank == 1 and read.dimension == 2
    assert np.max(np.abs(np.asarray(read.transport) + np.eye(2))) < 1e-13
    assert read.intertwiningResidual < 1e-14
    assert read.rankDefect < 1e-15


def test_the_exponential_scheme_intertwines_to_third_order():
    """The exponential step's intertwining residual is third order in the
    step: halving the step angle divides it by 8 to within one percent."""
    residuals = []
    for theta in (0.02, 0.01):
        read = CT.katoTransport([_line(0.0), _line(theta)],
                                KS.ExponentialGenerator)
        residuals.append(read.intertwiningResidual)
    assert residuals[0] / residuals[1] == pytest.approx(8.0, rel=1e-2)


@pytest.mark.parametrize("alpha", [0.0, 0.3, math.pi / 2])
def test_the_leakage_of_a_rotated_transfer(alpha):
    """Source and destination bands both the line e_1 and the transfer
    R(alpha): the leakage ||(I - P_A) T P_B||_2 is |sin alpha|."""
    p = _line(0.0)
    assert CT.leakage(p, _rotation(alpha), p) == \
        pytest.approx(abs(math.sin(alpha)), abs=1e-15)


def test_the_dual_transport_is_the_inverse_transpose():
    """M = [[2, 1], [0, 3]]: M^-T = [[1/2, 0], [-1/6, 1/3]] and its
    determinant is 1/6 = (det M)^-1 exactly; the dual of the dual is M."""
    m = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=complex)
    dual = np.asarray(CT.dualTransport(m))
    expected = np.array([[0.5, 0.0], [-1.0 / 6.0, 1.0 / 3.0]])
    assert np.max(np.abs(dual - expected)) < 1e-15
    assert np.linalg.det(dual) == pytest.approx(1.0 / 6.0, abs=1e-15)
    assert np.max(np.abs(np.asarray(CT.dualTransport(dual)) - m)) < 1e-14
    with pytest.raises(ValueError):
        CT.dualTransport(np.array([[1.0, 2.0], [2.0, 4.0]], dtype=complex))


def test_composition_applies_the_first_link_first():
    a = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    b = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    assert np.max(np.abs(np.asarray(CT.compose([a, b])) - b @ a)) == 0.0
    assert np.max(np.abs(np.asarray(CT.reversedTransfer(a)) - a.T)) == 0.0


# ----------------------------------------------------------- lineage


def _run_level_history():
    """The run's tick-0 base (two tetrahedra sharing the face 0 1 3, five
    vertices) reduced onto two response vertices joined by an edge:
    vertices 0, 1, 2 onto the first and 3, 4 onto the second. In the
    cobordism, level-0 vertex v is v and response vertex r is 5 + r."""
    return CL.history([obs.LevelComplex([[0, 1, 2, 3], [0, 1, 3, 4]], 5),
                       obs.LevelComplex([[0, 1]], 2)],
                      [[0, 0, 0, 1, 1]])


def test_a_fibre_path_crosses_the_level_cut_once():
    """The lineage of vertex 2 along its fibre edge 2 -> 5 crosses the level
    cut once, outward: N_Q = +1, a relative cycle, no interior source."""
    W = _run_level_history()
    read = CL.read(W, CL.levelCut(W, 0), CL.fromFiberPath(W, 2, 1, "Q"))
    assert read.number == 1
    assert read.relativeCycle and read.cutSeparates
    assert list(read.failedCertificates) == []


def test_a_zigzag_lineage_counts_signed_crossings():
    """The vertex path 0 -> 5 -> 1 -> 3 -> 6 crosses the level cut three
    times, outward, inward and outward: the signed count is
    +1 - 1 + 1 = +1, and the reversed path gives -1."""
    W = _run_level_history()
    cut = CL.levelCut(W, 0)
    zigzag = CL.fromVertexPath(W, [0, 5, 1, 3, 6], 1, "zigzag")
    assert CL.intersectionNumber(W, cut, zigzag) == 1
    assert CL.intersectionNumber(W, cut, CL.reversed(zigzag)) == -1
    crossings = [e for e, c in enumerate(np.asarray(zigzag.coefficients))
                 if c != 0 and e in list(cut.crossingEdges)]
    assert len(crossings) == 3


def test_the_side_cut_equals_the_level_cut_and_refuses_a_misplaced_vertex():
    """On a history with no interior vertex the only separating side
    assignment is 0 on the five level-0 vertices and 1 on the two response
    vertices: it reads +1 for the fibre path of vertex 4, as the level cut
    does. Putting response vertex 5 on the incoming side is refused by name,
    'outgoing-boundary-not-on-the-outgoing-side'."""
    W = _run_level_history()
    cut = CL.cutFromSides(W, [0] * 5 + [1] * 2)
    assert cut.separates
    assert CL.intersectionNumber(W, cut, CL.fromFiberPath(W, 4, 1, "Q")) == 1
    wrong = CL.cutFromSides(W, [0] * 6 + [1])
    assert not wrong.separates
    assert "outgoing-boundary-not-on-the-outgoing-side" in \
        list(wrong.failedCertificates)
