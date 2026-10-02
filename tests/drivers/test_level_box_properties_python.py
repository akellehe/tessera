# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Properties of what a tick does on the base of a level after its
relaxation: the covariant operator, the partition, the fibers, the inherited
pairing, the grown-cell rule and the growth step.

Each test states a property and checks it on small levels: the declared host
(a fan of two tetrahedra with a unit monopole on each) and the levels two
runs of the driver recorded (`_recursion_levels_2026_10_01`).

Terms used below:

* a *gauge transformation* is a map g from the vertices to the nonzero
  complex numbers, acting on the links by U_xy -> g_x U_xy / g_y; it is
  *unit-modulus* when every |g_x| is one, and the gauge group of the paper
  (WP v18 section 3) contains every other one as well;
* a *relabeling* is a permutation of the vertices of the base, the fields
  carried to the ascending orientation of the new labels (a link whose
  orientation turns is inverted);
* a *component* is a part of the partition of the level's edge coordinates,
  named here by its edges so that it can be compared across relabelings and
  across the orders in which the partition lists its parts;
* the *inherited pairing* g is the gauge-invariant pairing of the grown-cell
  rule on the determinant line (`recursion.inherited_pairing`), and its
  *row-sum defect* on a grown cell is ||g 1|| / ||g||;
* a *strict expected failure* marks a property the code does not have; its
  reason names the defect, and the test fails when the property starts to
  hold.
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R
from tests.drivers import _recursion_levels_2026_10_01 as RUN

PARTITION_WEIGHT = (
    "the partition's graph weight |R_ij| + |R_ji| changes under a gauge "
    "transformation whose moduli are not one, and the order in which the "
    "components are listed depends on rounding; theory(recursion): the "
    "partition's graph weight |R_ij| + |R_ji| changes under the complex "
    "gauge group, and the paper says the effective components do not, "
    "https://github.com/akellehe/tessera/issues/1360")
GROWTH_BRANCH = (
    "fix(drivers): the growth step scores the Regge term on a branch cut, "
    "where the sign of a rounding-size imaginary part decides it, "
    "https://github.com/akellehe/tessera/issues/1361")
BASE_VERTEX = (
    "the twisted incidences and the dressed Whitney metrics transport "
    "between the base vertices b(sigma) = min sigma of two cells along the "
    "direct link, so on a connection with a face holonomy other than one a "
    "change of the vertex order is not a similarity of h_k(z, U) and its "
    "spectrum changes")
RULE_VERTEX = (
    "GrownCellRule.invertVertexPairing inverts the block of the pairing off "
    "the cell's first vertex; on a pairing whose row sums are not zero the "
    "squared lengths depend on which vertex is listed first")
COMPONENT_ORDER = (
    "the grown squared lengths and the cells the manifold gate keeps depend "
    "on the order in which the partition lists its components: the "
    "grown-cell rule singles out each cell's lowest-numbered response "
    "vertex, and the cells are glued in lexicographic order of their "
    "vertices")
GROWN_UNITS = (
    "the off-diagonal entries of the inherited pairing are divided by "
    "U_vw = det M_vw, which carries the unit of the operator, and "
    "det M_wv is not the inverse of det M_vw; theory(recursion): the grown "
    "connection U = det M carries the operator's units; the dimensionless "
    "phase rule is open, https://github.com/akellehe/tessera/issues/1312")
PARTITION_ROUNDING = (
    "a change of the operator's entries by one part in 1e15 changes the "
    "components the modularity search returns; "
    "theory(recursion): the level partition comes from a heuristic "
    "optimizer of an NP-hard objective, run alone, "
    "https://github.com/akellehe/tessera/issues/1314")
REPRESENTATIVE = (
    "the box partitions the magnitude graph of A M^-1 (h_1 on chains) and "
    "reads each fiber on the component's block of M^-1 A (h_1 on geometric "
    "images, the operator of the cell reads); the two matrices are similar "
    "by the Whitney metric and their magnitude graphs have different "
    "partitions")
GROWTH_GAUGE = (
    "the Hodge term of the growth step's objective "
    "(JointStationarityObjective) changes under a gauge transformation "
    "whose moduli are not one")


def _config():
    config = R.default_config()
    config["persistence_required"] = 1
    return config


def _declared_host():
    cells, z, links, _ = R.level_zero(R.default_config())
    return cells, z, links


def _tick0():
    return (RUN.TICK0_CELLS, dict(RUN.TICK0_SQUARED_LENGTHS),
            dict(RUN.TICK0_LINKS))


def _tick1():
    return (RUN.TICK1_CELLS, dict(RUN.TICK1_SQUARED_LENGTHS),
            dict(RUN.TICK1_LINKS))


def _tick1_copy():
    return (RUN.TICK1_CELLS, dict(RUN.TICK1_COPY_SQUARED_LENGTHS),
            dict(RUN.TICK1_COPY_LINKS))


LEVELS = {"declared host": _declared_host, "tick 0": _tick0,
          "tick 1": _tick1, "tick 1, gauge copy": _tick1_copy}


def _vertices(cells):
    return 1 + max(max(c) for c in cells)


def _gauge(links, g):
    """U_ab -> g_a U_ab / g_b."""
    return {(a, b): g[a] * u / g[b] for (a, b), u in links.items()}


def _unit_gauge(count, seed):
    rng = np.random.default_rng(seed)
    return [np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) for _ in range(count)]


def _complex_gauge(count, seed):
    rng = np.random.default_rng(seed)
    return [np.exp(0.7 * rng.normal() + 1j * rng.uniform(0.0, 2.0 * np.pi))
            for _ in range(count)]


def _relabel(cells, z, links, permutation):
    """The level with vertex v named ``permutation[v]``."""
    moved_cells = sorted(sorted(permutation[v] for v in c) for c in cells)
    moved_z, moved_links = {}, {}
    for (a, b), value in z.items():
        x, y = permutation[a], permutation[b]
        key = (min(x, y), max(x, y))
        moved_z[key] = value
        moved_links[key] = links[(a, b)] if x < y else 1.0 / links[(a, b)]
    return moved_cells, moved_z, moved_links


def _named(partition, edges, original=None):
    """The components of a partition, each the set of its edges, an edge
    named by its two vertices; ``original`` maps a relabeled vertex back."""
    out = []
    for part in partition:
        names = []
        for coordinate in part:
            a, b = edges[coordinate]
            if original is not None:
                a, b = original[a], original[b]
            names.append((min(a, b), max(a, b)))
        out.append(frozenset(names))
    return out


def _spectrum(cells, z, links, degree):
    """The spectrum of h_degree(z, U) of a base, sorted."""
    complex_ = cob.ChainComplex.fromTopCells([sorted(c) for c in cells])
    edges = [tuple(e) for e in complex_.kSimplexVertices(1)]
    hodge = ch.ChainHodge(complex_, [complex(z[e]) for e in edges])
    covariant = ch.CovariantChainHodge(
        hodge, ch.Connection(complex_, [complex(links[e]) for e in edges]))
    return np.sort_complex(np.linalg.eigvals(
        np.asarray(covariant.covariantOperator(degree))))


def _spectra_differ(first, second):
    return float(np.max(np.abs(first - second)) / np.max(np.abs(first)))


def _partition(cells, z, links, config):
    base = R.base_operator(cells, z, links)
    level = R.recursion_turn(base["operator"], config)
    return base, [list(p) for p in level.partition]


# ------------------------------------------------- the covariant operator


@pytest.mark.parametrize("level", sorted(LEVELS))
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_the_spectrum_of_the_covariant_operator_is_gauge_invariant(level,
                                                                   degree):
    """h_k(z, U^g) is similar to h_k(z, U) for every gauge transformation of
    the paper's group (WP v18 section 3), so its spectrum is the same under a
    unit-modulus one and under one whose moduli are not one."""
    cells, z, links = LEVELS[level]()
    reference = _spectrum(cells, z, links, degree)
    count = _vertices(cells)
    for g in (_unit_gauge(count, 1), _complex_gauge(count, 2)):
        moved = _spectrum(cells, z, _gauge(links, g), degree)
        assert _spectra_differ(reference, moved) < 1e-12


@pytest.mark.parametrize("level", ["declared host", "tick 1"])
def test_the_dual_connections_pencil_is_the_transpose(level):
    """The pencil of the inverse connection is the transpose of the
    connection's, A(U^-1) = A(U)^T and M(U^-1) = M(U)^T, so the two operators
    have one spectrum: the dual band of a fiber is a band of the same
    eigenvalues (spec Prop. 4)."""
    cells, z, links = LEVELS[level]()
    base = R.base_operator(cells, z, links)
    for key in ("pencil", "metric"):
        scale = np.linalg.norm(base[key])
        transpose = base[key].T
        assert np.linalg.norm(base["dual_" + key] - transpose) < 1e-12 * scale
    assert _spectra_differ(
        np.sort_complex(np.linalg.eigvals(base["operator"])),
        np.sort_complex(np.linalg.eigvals(base["dual_operator"]))) < 1e-12


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3, 1e9])
@pytest.mark.parametrize("level", ["declared host", "tick 0"])
def test_the_covariant_operator_is_homogeneous_in_the_squared_lengths(
        level, factor):
    """h_1(c z, U) = h_1(z, U) / c: the operator carries the unit of an
    inverse squared length and nothing else of the scale, from cells of
    order 1e-6 to cells of order 1e9."""
    cells, z, links = LEVELS[level]()
    reference = _spectrum(cells, z, links, 1)
    scaled = _spectrum(cells, {e: factor * v for e, v in z.items()}, links, 1)
    assert _spectra_differ(reference, factor * scaled) < 1e-10


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_a_pure_gauge_operator_does_not_depend_on_the_vertex_labels(degree):
    """On a pure-gauge connection, U_xy = g_x / g_y, every face holonomy is
    one and a relabeling of the vertices is a similarity of h_k: the
    spectrum is the same for every one of the 24 labelings of a tetrahedron
    with six different squared lengths."""
    rng = np.random.default_rng(7)
    cells = [[0, 1, 2, 3]]
    edges = list(itertools.combinations(range(4), 2))
    z = {e: complex(3.0 + 2.0 * rng.uniform()) for e in edges}
    g = _complex_gauge(4, 8)
    links = {(a, b): g[a] / g[b] for a, b in edges}
    reference = _spectrum(cells, z, links, degree)
    for permutation in itertools.permutations(range(4)):
        moved = _spectrum(*_relabel(cells, z, links, permutation), degree)
        assert _spectra_differ(reference, moved) < 1e-12


@pytest.mark.xfail(strict=True, reason=BASE_VERTEX)
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_the_spectrum_of_the_covariant_operator_does_not_depend_on_the_labels(
        degree):
    """The spectrum of h_k(z, U) is a property of the level and not of the
    names of its vertices (WP v18 section 10, condition 7: the spectral
    fingerprint is stable under vertex relabeling; section 12: every
    reported observable is invariant under relabeling). Measured on the
    recorded tick-1 level, whose face holonomies are not one: the
    transposition of the two highest vertices moves the spectrum by 0.013
    (degree 0), 0.056 (degree 1) and 0.19 (degree 2) of its largest
    modulus."""
    cells, z, links = _tick1()
    reference = _spectrum(cells, z, links, degree)
    count = _vertices(cells)
    for permutation in ([1, 0, 2, 3, 4], [0, 1, 2, 4, 3], [4, 2, 1, 3, 0]):
        assert len(permutation) == count
        moved = _spectrum(*_relabel(cells, z, links, permutation), degree)
        assert _spectra_differ(reference, moved) < 1e-12


def _host_cell(cells, z, links, cell):
    fixture = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    c = sorted(cell)
    return {"squared_lengths": [z[(c[i], c[j])] for i, j in fixture],
            "links": [links[(c[i], c[j])] for i, j in fixture]}


def test_the_declared_relabeling_of_condition_7_keeps_the_spectrum():
    """The relabeling quark condition 7 declares, the transposition of a
    cell's two lowest vertices (`baryon_poles.FINGERPRINT_RELABELING`),
    leaves the spectrum of h_1 of a recorded host cell as it is."""
    cells, z, links = _tick0()
    config = bp.default_config([1.0], [1.0])
    host = bp.build_host(8.0, _host_cell(cells, z, links, cells[0]))
    reference = np.sort_complex(np.linalg.eigvals(
        bp._carrier(host, 1.0, 1.0, config)[:6, :6]))
    moved = np.sort_complex(np.linalg.eigvals(bp._carrier(
        bp.relabeled_host(host, bp.FINGERPRINT_RELABELING), 1.0, 1.0,
        config)[:6, :6]))
    assert _spectra_differ(reference, moved) < 1e-12


@pytest.mark.xfail(strict=True, reason=BASE_VERTEX)
def test_every_relabeling_of_a_host_cell_keeps_the_spectrum():
    """Quark condition 7 asks for a fingerprint stable under vertex
    relabeling. Measured on the first host cell of the recorded tick 0: 6 of
    the 24 relabelings keep the spectrum of h_1 (the declared one among
    them), and the transposition of the two highest vertices moves it by
    0.53 of its largest modulus."""
    cells, z, links = _tick0()
    config = bp.default_config([1.0], [1.0])
    host = bp.build_host(8.0, _host_cell(cells, z, links, cells[0]))
    reference = np.sort_complex(np.linalg.eigvals(
        bp._carrier(host, 1.0, 1.0, config)[:6, :6]))
    for permutation in itertools.permutations(range(4)):
        moved = np.sort_complex(np.linalg.eigvals(bp._carrier(
            bp.relabeled_host(host, permutation), 1.0, 1.0, config)[:6, :6]))
        assert _spectra_differ(reference, moved) < 1e-12


def _two_representatives(cells, z, links):
    """h_1 of a base as the joint action carries it on one sheet and as the
    box forms it, with the box's record."""
    sheet, _ = R.build_level(cells, z, links, sheets=1)
    action = cob.JointAction(sheet, bp.action_declaration(sheet, 1.0, 1.0))
    carrier = np.asarray(action.carrier_operator()).reshape(len(z), len(z))
    return sheet, carrier, R.base_operator(cells, z, links)


@pytest.mark.parametrize("level", ["declared host", "tick 1"])
def test_the_cell_reads_and_the_box_carry_one_operator_in_two_bases(level):
    """The carrier operator of the joint action on one sheet (the operator of
    the cell reads) and the operator of the box are on the same edges in the
    same order and have one spectrum. They are two matrices: with the pencil
    (A, M) of the base, the box's is A M^-1, h_1 acting on chains, and the
    carrier is M^-1 A, h_1 acting on geometric images, the similarity by the
    Whitney metric M between them."""
    from tessera.drivers import cell_solve
    cells, z, links = LEVELS[level]()
    sheet, carrier, box = _two_representatives(cells, z, links)
    assert cell_solve.carrier_cells(sheet) == box["edges"]
    pencil, metric = box["pencil"], box["metric"]
    scale = np.linalg.norm(box["operator"])
    assert np.linalg.norm(box["operator"]
                          - pencil @ np.linalg.inv(metric)) < 1e-12 * scale
    assert np.linalg.norm(carrier
                          - np.linalg.solve(metric, pencil)) < 1e-12 * scale
    assert _spectra_differ(
        np.sort_complex(np.linalg.eigvals(carrier)),
        np.sort_complex(np.linalg.eigvals(box["operator"]))) < 1e-12
    assert np.linalg.norm(carrier - box["operator"]) > 0.1 * scale


def _components_of(operator):
    read = cob.RecursiveQuotient.persistentPartitionOverResolutions(
        list(np.asarray(operator, dtype=complex).reshape(-1)),
        operator.shape[0], list(R.DECLARED_RESOLUTIONS))
    return [sorted(int(v) for v in part) for part in read.components]


@pytest.mark.xfail(strict=True, reason=REPRESENTATIVE)
def test_the_partition_is_the_same_in_the_two_bases_of_the_operator():
    """The components of a level are components of h_1, whichever of the two
    bases the run carries it in. Measured on the recorded tick-1 level: the
    box's matrix gives the edge sets {0, 1, 4}, {2, 5, 7}, {3, 6, 8, 9} and
    the cell reads' matrix gives {0, 3, 6, 8, 9}, {1, 4}, {2, 5, 7}."""
    _, carrier, box = _two_representatives(*_tick1())
    assert sorted(_components_of(carrier)) == sorted(
        _components_of(box["operator"]))


# ------------------------------------------------------------ the partition


@pytest.mark.parametrize("level", ["tick 1", "tick 1, gauge copy"])
def test_the_partition_is_the_same_under_a_unit_modulus_gauge(level):
    """A unit-modulus gauge transformation leaves every modulus |R_ij| as it
    is up to rounding, and on the recorded tick-1 levels the components are
    the same sets of edges. (On the declared host a change of one rounding
    changes them: `test_the_partition_does_not_turn_on_one_rounding`.)"""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition = _partition(cells, z, links, config)
    reference = set(_named(partition, base["edges"]))
    for seed in (1, 2, 3):
        g = _unit_gauge(_vertices(cells), seed)
        moved_base, moved = _partition(cells, z, _gauge(links, g), config)
        assert set(_named(moved, moved_base["edges"])) == reference


@pytest.mark.xfail(strict=True, reason=PARTITION_WEIGHT)
@pytest.mark.parametrize("level", sorted(LEVELS))
def test_the_partition_is_the_same_under_a_complex_gauge(level):
    """The effective components are gauge-invariant under the paper's gauge
    group (WP v18 section 3). Measured: the declared host's five components
    become seven under one gauge transformation whose moduli are not one,
    and the two recorded tick-1 levels, gauge copies of each other, have
    three and six."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition = _partition(cells, z, links, config)
    reference = set(_named(partition, base["edges"]))
    g = _complex_gauge(_vertices(cells), 2)
    moved_base, moved = _partition(cells, z, _gauge(links, g), config)
    assert set(_named(moved, moved_base["edges"])) == reference


@pytest.mark.xfail(strict=True, reason=PARTITION_WEIGHT)
def test_two_gauge_copies_of_a_level_have_one_partition():
    """The tick-1 levels of the two recorded runs have the same squared
    lengths and the same face holonomies. Their partitions are three
    components (no grown cell: the run of 2026-10-01 ended there) and six
    (five grown cells)."""
    config = _config()
    first_base, first = _partition(*_tick1(), config)
    second_base, second = _partition(*_tick1_copy(), config)
    assert (set(_named(first, first_base["edges"]))
            == set(_named(second, second_base["edges"])))


def test_the_recorded_copies_are_gauge_copies():
    """The two recorded tick-1 levels have squared lengths equal to 2.5e-7
    of their size, face holonomies equal to 1e-12, spectra of h_1 equal to
    1e-6, and links whose moduli differ."""
    cells, z, links = _tick1()
    _, copy_z, copy_links = _tick1_copy()
    assert max(abs(z[e] - copy_z[e]) / abs(z[e]) for e in z) < 1e-6
    for a, b, c in itertools.combinations(range(5), 3):
        first = links[(a, b)] * links[(b, c)] / links[(a, c)]
        second = copy_links[(a, b)] * copy_links[(b, c)] / copy_links[(a, c)]
        assert abs(first - second) < 1e-12
    assert _spectra_differ(_spectrum(cells, z, links, 1),
                           _spectrum(cells, copy_z, copy_links, 1)) < 1e-6
    assert max(abs(abs(links[e]) - abs(copy_links[e])) for e in links) > 0.1


@pytest.mark.xfail(strict=True, reason=PARTITION_WEIGHT)
def test_the_components_are_listed_in_one_order_under_a_unit_modulus_gauge():
    """The order of the components numbers the response vertices of the next
    level. Measured on the declared host: a unit-modulus gauge transformation,
    which changes the weights of the partition's graph by rounding alone,
    lists the same five components in another order."""
    config = _config()
    cells, z, links = _declared_host()
    base, partition = _partition(cells, z, links, config)
    reference = _named(partition, base["edges"])
    for seed in range(1, 9):
        g = _unit_gauge(_vertices(cells), seed)
        moved_base, moved = _partition(cells, z, _gauge(links, g), config)
        assert _named(moved, moved_base["edges"]) == reference


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3, 1e9])
@pytest.mark.parametrize("level", ["tick 1", "tick 1, gauge copy"])
def test_the_partition_does_not_depend_on_the_unit_of_length(level, factor):
    """A change of the unit of length multiplies every weight of the
    partition's graph by one number, which the modularity does not see: the
    components are the same."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition = _partition(cells, z, links, config)
    scaled_base, scaled = _partition(
        cells, {e: factor * v for e, v in z.items()}, links, config)
    assert (set(_named(scaled, scaled_base["edges"]))
            == set(_named(partition, base["edges"])))


@pytest.mark.xfail(strict=True, reason=PARTITION_ROUNDING)
def test_the_partition_does_not_turn_on_one_rounding():
    """The components of a level do not change when every entry of its
    operator changes by one part in 1e15, the size of the rounding of any
    step that forms it. Measured on the declared host: of forty such
    changes thirty-four return the five components the run reads, the edge
    sets {0, 1, 4}, {2, 5, 7}, {3}, {6}, {8}, and six return the six
    components {0}, {1}, {2, 5, 7, 8}, {3}, {4}, {6}."""
    cells, z, links = _declared_host()
    operator = R.base_operator(cells, z, links)["operator"]
    reference = sorted(_components_of(operator))
    rng = np.random.default_rng(21)
    for _ in range(40):
        noise = 1.0 + 1e-15 * rng.uniform(-1.0, 1.0, operator.shape)
        assert sorted(_components_of(operator * noise)) == reference


def test_the_partition_is_the_same_at_every_call():
    """The search has no seed to vary: three calls return one partition, in
    one order, and the operator and the inherited pairing bit for bit."""
    config = _config()
    cells, z, links = _tick1_copy()
    seen = set()
    for _ in range(3):
        base, partition = _partition(cells, z, links, config)
        stage = R.interaction_stage(base, partition, config)
        seen.add((json.dumps(partition), base["operator"].tobytes(),
                  stage["pairing"].tobytes()))
    assert len(seen) == 1


def test_the_partition_does_not_depend_on_the_number_of_threads():
    """The partition of the recorded tick-1 level is the same list with one
    OpenMP thread and with two."""
    script = (
        "import json, sys\n"
        "sys.path[:0] = %r\n"
        "from tessera.drivers import recursion as R\n"
        "from tests.drivers import _recursion_levels_2026_10_01 as RUN\n"
        "config = R.default_config()\n"
        "base = R.base_operator(RUN.TICK1_CELLS,"
        " RUN.TICK1_COPY_SQUARED_LENGTHS, RUN.TICK1_COPY_LINKS)\n"
        "level = R.recursion_turn(base['operator'], config)\n"
        "print(json.dumps([list(p) for p in level.partition]))\n"
        % (list(sys.path),))
    outputs = []
    for threads in ("1", "2"):
        environment = dict(os.environ, OMP_NUM_THREADS=threads)
        outputs.append(subprocess.run(
            [sys.executable, "-c", script], env=environment, check=True,
            capture_output=True, text=True).stdout.strip())
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])) == 6


def test_the_partition_routine_is_covariant_under_a_relabeling():
    """`RecursiveQuotient.persistentPartitionOverResolutions` on an operator
    of three blocks of four coordinates with weak couplings between the
    blocks: a permutation of the coordinates permutes the components."""
    rng = np.random.default_rng(8)
    operator = rng.uniform(0.1, 1.0, (12, 12))
    operator = operator + operator.T
    for i, j in itertools.product(range(12), repeat=2):
        if i // 4 != j // 4:
            operator[i, j] *= 0.05
    resolutions = list(R.DECLARED_RESOLUTIONS)

    def components(matrix):
        read = cob.RecursiveQuotient.persistentPartitionOverResolutions(
            list(matrix.astype(complex).reshape(-1)), 12, resolutions)
        return [sorted(int(v) for v in part) for part in read.components]

    reference = components(operator)
    assert sorted(reference) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    for _ in range(4):
        permutation = rng.permutation(12)
        moved = components(operator[np.ix_(permutation, permutation)])
        assert sorted(sorted(int(permutation[i]) for i in part)
                      for part in moved) == sorted(reference)


# ------------------------------------------- the fibers and the pairing


def _stage(cells, z, links, config, partition=None):
    base = R.base_operator(cells, z, links)
    if partition is None:
        partition = [list(p) for p in
                     R.recursion_turn(base["operator"], config).partition]
    return base, partition, R.interaction_stage(base, partition, config)


def _grown_holonomies(stage, count):
    out = {}
    for a, b, c in itertools.combinations(range(count), 3):
        if all(key in stage["links"] for key in ((a, b), (b, c), (a, c))):
            out[(a, b, c)] = (stage["links"][(a, b)] * stage["links"][(b, c)]
                              / stage["links"][(a, c)])
    return out


@pytest.mark.parametrize("level", ["declared host", "tick 0",
                                   "tick 1, gauge copy"])
@pytest.mark.parametrize("unit", [True, False])
def test_the_grown_fields_are_gauge_invariant_at_a_given_partition(level,
                                                                   unit):
    """With the partition held, the inherited pairing, the grown squared
    lengths, the row-sum defects and the face holonomies of the grown links
    are the same under a gauge transformation of the level, unit-modulus or
    not: the grown links change by a gauge transformation of the next level
    and by nothing else."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition, reference = _stage(cells, z, links, config)
    count = _vertices(cells)
    g = _unit_gauge(count, 5) if unit else _complex_gauge(count, 6)
    _, _, moved = _stage(cells, z, _gauge(links, g), config, partition)
    scale = np.abs(reference["pairing"]).max()
    assert np.abs(moved["pairing"] - reference["pairing"]).max() < 1e-8 * scale
    assert sorted(moved["z"]) == sorted(reference["z"]) and reference["z"]
    for edge, value in reference["z"].items():
        assert abs(moved["z"][edge] - value) < 1e-8 * abs(value)
    holonomies = _grown_holonomies(reference, len(partition))
    assert holonomies
    for face, value in _grown_holonomies(moved, len(partition)).items():
        assert abs(value - holonomies[face]) < 1e-8 * abs(holonomies[face])
    for first, second in zip(reference["reads"], moved["reads"]):
        if "row_sum_defect" in first:
            assert abs(first["row_sum_defect"]
                       - second["row_sum_defect"]) < 1e-8


@pytest.mark.xfail(strict=True, reason=GROWN_UNITS)
@pytest.mark.parametrize("level", ["declared host", "tick 1, gauge copy"])
def test_the_grown_links_are_a_connection(level):
    """A link read backward is the inverse of the link read forward,
    U_wv U_vw = 1, which is what makes the holonomy of a grown face turn
    into its inverse when the face is traversed the other way. Measured:
    |U_wv U_vw - 1| reaches 22.5 on the declared host (the two
    determinants have one modulus, 4.85 on that edge) and 1 on the recorded
    tick-1 level."""
    config = _config()
    _, _, stage = _stage(*LEVELS[level](), config)
    assert stage["groupoid"]
    assert max(stage["groupoid"].values()) < 1e-8


@pytest.mark.parametrize("level", ["declared host", "tick 1, gauge copy"])
def test_the_inherited_pairing_follows_the_order_of_the_components(level):
    """The inherited pairing between two components does not depend on the
    order in which the partition lists them: listing them in another order
    permutes its rows and columns and changes no entry."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition, reference = _stage(cells, z, links, config)
    order = list(reversed(range(len(partition))))
    _, _, moved = _stage(cells, z, links, config,
                         [partition[i] for i in order])
    scale = np.abs(reference["pairing"]).max()
    assert (np.abs(moved["pairing"]
                   - reference["pairing"][np.ix_(order, order)]).max()
            < 1e-12 * scale)


def test_the_grown_cell_rule_on_a_pairing_with_zero_row_sums_is_label_free():
    """On a pairing whose rows sum to zero (the vertex Gram of a
    tetrahedron's Whitney forms, which the level-0 identification returns)
    the squared lengths the rule gives an edge are the same whichever
    vertex of the cell is listed first."""
    pairing = np.array([[3.0, -1.0, -1.0, -1.0], [-1.0, 3.0, -1.0, -1.0],
                        [-1.0, -1.0, 3.0, -1.0], [-1.0, -1.0, -1.0, 3.0]],
                       dtype=complex)
    pairing += 0.3 * np.array([[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, -1.0, 1.0]])
    assert np.abs(pairing @ np.ones(4)).max() < 1e-15
    edges = list(itertools.combinations(range(4), 2))
    reference = dict(zip(edges, ch.GrownCellRule.invertVertexPairing(
        pairing, 1e-15).squaredLengths))
    for permutation in itertools.permutations(range(4)):
        moved = ch.GrownCellRule.invertVertexPairing(
            pairing[np.ix_(permutation, permutation)], 1e-15).squaredLengths
        for (i, j), value in zip(edges, moved):
            edge = tuple(sorted((permutation[i], permutation[j])))
            assert abs(value - reference[edge]) < 1e-12 * abs(reference[edge])


@pytest.mark.xfail(strict=True, reason=RULE_VERTEX)
def test_the_grown_cell_rule_does_not_depend_on_the_vertex_listed_first():
    """The squared length the rule gives a grown edge is a property of the
    cell's pairing and not of the order of its vertices. Measured on the
    first grown cell of the recorded tick-1 level (row-sum defect 1.24):
    listing its second vertex first changes a squared length by 0.76 of
    its value, and reversing the vertices by a factor of 54."""
    config = _config()
    _, _, stage = _stage(*_tick1_copy(), config)
    pairing = stage["reads"][0]["pairing"]
    edges = list(itertools.combinations(range(4), 2))
    reference = dict(zip(edges, ch.GrownCellRule.invertVertexPairing(
        pairing, 1e-15).squaredLengths))
    for permutation in ((1, 0, 2, 3), (3, 2, 1, 0)):
        moved = ch.GrownCellRule.invertVertexPairing(
            pairing[np.ix_(permutation, permutation)], 1e-15).squaredLengths
        for (i, j), value in zip(edges, moved):
            edge = tuple(sorted((permutation[i], permutation[j])))
            assert abs(value - reference[edge]) < 1e-8 * abs(reference[edge])


def _grown_level(stage, partition, edges):
    """The grown level named by components: the kept cells and the squared
    length of every grown edge."""
    names = _named(partition, edges)
    cells = {frozenset(names[v] for v in read["vertices"])
             for read in stage["reads"] if "squared_lengths" in read}
    lengths = {frozenset((names[v], names[w])): value
               for (v, w), value in stage["z"].items()}
    return cells, lengths


@pytest.mark.xfail(strict=True, reason=COMPONENT_ORDER)
@pytest.mark.parametrize("level", ["declared host", "tick 1, gauge copy"])
def test_the_grown_level_does_not_depend_on_the_order_of_the_components(
        level):
    """The next level is the same complex with the same squared lengths, up
    to the names of its vertices, whatever order the partition lists its
    components in. Measured: on the declared host, where every grown cell is
    kept, the components listed in reverse change a grown squared length by
    0.97 of its value; on the recorded tick-1 level, where the manifold gate
    keeps five of fifteen cells, it keeps five others."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition, reference = _stage(cells, z, links, config)
    kept, lengths = _grown_level(reference, partition, base["edges"])
    order = list(reversed(range(len(partition))))
    moved_partition = [partition[i] for i in order]
    _, _, moved = _stage(cells, z, links, config, moved_partition)
    moved_kept, moved_lengths = _grown_level(moved, moved_partition,
                                             base["edges"])
    assert moved_kept == kept
    for edge, value in lengths.items():
        assert abs(moved_lengths[edge] - value) < 1e-8 * abs(value)


@pytest.mark.xfail(strict=True, reason=GROWN_UNITS)
@pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3, 1e9])
@pytest.mark.parametrize("level", ["declared host", "tick 0"])
def test_the_grown_squared_lengths_scale_with_the_unit_of_length(level,
                                                                 factor):
    """With the partition held, a change of the unit of length of the level
    by the factor c multiplies every grown squared length by one number.
    Measured on both levels: the ratios of the ten grown squared lengths
    to their values at c = 1 run from 1.0007e-6 to 1.0405e-6 at c = 1e-6
    and from 6.8e23 to 2.0e24 at c = 1e9. The diagonal of the pairing
    scales as c^(1/2) and its off-diagonal entries, divided by det M, as
    c^(3/2)."""
    config = _config()
    cells, z, links = LEVELS[level]()
    base, partition, reference = _stage(cells, z, links, config)
    _, _, scaled = _stage(cells, {e: factor * v for e, v in z.items()},
                          links, config, partition)
    assert sorted(scaled["z"]) == sorted(reference["z"])
    ratios = [scaled["z"][e] / reference["z"][e] for e in reference["z"]]
    for ratio in ratios:
        assert abs(ratio - ratios[0]) < 1e-6 * abs(ratios[0])


# ---------------------------------------------------- the driver's helpers


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_a_level_returns_the_fields_it_is_built_from(level):
    """`build_level` writes the base fields on every sheet and
    `sheet_fields` reads them back; `sheet_base` and `base_fields` return
    the base."""
    cells, z, links = LEVELS[level]()
    spacetime, count = R.build_level(cells, z, links)
    for sheet_z, sheet_links in R.sheet_fields(spacetime, count):
        assert sorted(sheet_z) == sorted(z)
        for edge in z:
            assert abs(sheet_z[edge] - z[edge]) < 1e-12 * abs(z[edge])
            assert abs(sheet_links[edge] - links[edge]) < 1e-12 * abs(
                links[edge])
    back_cells, back_z, back_links, relabeling = R.base_fields(
        R.sheet_base(spacetime, count))
    assert back_cells == sorted(sorted(c) for c in cells)
    assert relabeling == {v: v for v in range(count)}
    for edge in z:
        assert abs(back_z[edge] - z[edge]) < 1e-12 * abs(z[edge])
        assert abs(back_links[edge] - links[edge]) < 1e-12 * abs(links[edge])


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 8])
def test_the_fan_carries_a_unit_monopole_on_every_tetrahedron(count):
    """The declared connection of a fan of any length solves
    d phi = theta - 2 pi n to rounding with phases of order one, and every
    tetrahedron reads the monopole number one: the rank of the face
    coboundary (edges minus vertices plus one) is decided at the declared
    rank tolerance, the zero singular values at 7e-17 to 1.8e-16 of the
    largest against the tolerance 1e-15."""
    cells = R.fan(count)
    connection = R.monopole_connection(cells)
    assert connection["residual"] < 1e-12
    assert np.linalg.norm(connection["phases"]) < 10.0
    links = {edge: np.exp(1j * phase) for edge, phase in
             zip(connection["edges"], connection["phases"])}
    assert R.monopole_numbers(cells, links) == [1] * count


def _parity(sequence):
    """The sign of the permutation that sorts a sequence of distinct
    numbers."""
    sign = 1
    for i, j in itertools.combinations(range(len(sequence)), 2):
        if sequence[i] > sequence[j]:
            sign = -sign
    return sign


@pytest.mark.parametrize("level", ["declared host", "tick 1"])
def test_a_monopole_number_carries_the_orientation_of_the_vertex_order(level):
    """The monopole number of a tetrahedron is read through its faces in the
    orientation of its ascending vertices, so a relabeling that is an odd
    permutation of a tetrahedron's vertex order turns the sign of its
    number, and an even one keeps it: the modulus is a property of the
    tetrahedron and the sign of the labels."""
    cells, z, links = LEVELS[level]()
    reference = {tuple(sorted(c)): number for c, number in
                 zip(cells, R.monopole_numbers(cells, links))}
    count = _vertices(cells)
    rng = np.random.default_rng(12)
    permutations = [[1, 0] + list(range(2, count)),
                    list(range(count - 2)) + [count - 1, count - 2],
                    [int(v) for v in rng.permutation(count)]]
    turned = 0
    for permutation in permutations:
        moved_cells, _, moved_links = _relabel(cells, z, links, permutation)
        numbers = dict(zip((tuple(c) for c in moved_cells),
                           R.monopole_numbers(moved_cells, moved_links)))
        for cell, number in reference.items():
            image = [permutation[v] for v in cell]
            sign = _parity(image)
            turned += sign < 0 and number != 0
            assert numbers[tuple(sorted(image))] == sign * number
    assert turned > 0


@pytest.mark.parametrize("level", ["declared host", "tick 1"])
def test_the_monopole_numbers_are_gauge_invariant(level):
    """The monopole number of a tetrahedron is read from its face
    holonomies, which no gauge transformation changes."""
    cells, z, links = LEVELS[level]()
    reference = R.monopole_numbers(cells, links)
    count = _vertices(cells)
    for g in (_unit_gauge(count, 3), _complex_gauge(count, 4)):
        assert R.monopole_numbers(cells, _gauge(links, g)) == reference


@pytest.mark.parametrize("level", ["declared host", "tick 1"])
def test_the_rule_shift_is_gauge_invariant_and_zero_at_a_pure_gauge(level):
    """The shift of the grown-cell rule on a level's own tetrahedra is the
    same under a gauge transformation, unit-modulus or not, and is zero on
    a pure-gauge connection."""
    cells, z, links = LEVELS[level]()
    reference = [s["relative_shift"]
                 for s in R.level_rule_shift(cells, z, links)]
    count = _vertices(cells)
    for g in (_unit_gauge(count, 3), _complex_gauge(count, 4)):
        moved = [s["relative_shift"]
                 for s in R.level_rule_shift(cells, z, _gauge(links, g))]
        np.testing.assert_allclose(moved, reference, rtol=1e-8, atol=1e-10)
    g = _complex_gauge(count, 9)
    flat = {(a, b): g[a] / g[b] for a, b in links}
    for shift in R.level_rule_shift(cells, z, flat):
        assert shift["relative_shift"] < 1e-10


def test_a_band_that_splits_an_exactly_repeated_eigenvalue_has_no_value():
    """A band of rank one on a block whose lowest eigenvalue is repeated
    exactly has no projector, and the read says so by name; the band of
    rank two that takes both is read and accepted; and a band that splits
    two eigenvalues one rounding apart is read and not accepted."""
    block = np.diag([1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="leaves an eigenvalue exactly equal"):
        R.riesz_band(block, 1, 1e-15)
    whole = R.riesz_band(block, 2, 1e-15)
    assert whole["accepted"] and whole["isolation_gap"] == 1.0
    near = R.riesz_band(np.diag([1.0, 1.0 + 2.0 ** -52, 2.0]), 1, 1e-15)
    assert not near["accepted"] and near["isolation_gap"] == 2.0 ** -52


# ----------------------------------------------------------- the growth step


def _growth_objective(z, links, config):
    return R.pachner_stage(RUN.TICK1_CELLS, z, links,
                           config)[3]["objective_before"]


def test_the_growth_objective_is_the_same_under_a_unit_modulus_gauge():
    """The objective of the growth step on the level grown at tick 0 is the
    same number under a unit-modulus gauge transformation of its links."""
    config = R.default_config()
    z, links = dict(RUN.GROWN_SQUARED_LENGTHS), dict(RUN.GROWN_LINKS)
    reference = _growth_objective(z, links, config)
    moved = _growth_objective(z, _gauge(links, _unit_gauge(5, 1)), config)
    assert abs(moved - reference) < 1e-9 * abs(reference)


@pytest.mark.xfail(strict=True, reason=GROWTH_GAUGE)
def test_the_growth_objective_is_the_same_under_a_complex_gauge():
    """The objective of the growth step is a function of the level and not
    of the gauge of its links. Measured on the level grown at tick 0: a
    gauge transformation whose moduli are not one moves the objective from
    3.1517 to 3.1600, all of it in the Hodge term (the spectral entropy is
    read on the singular values of the operator, which a unit-modulus gauge
    transformation keeps and another does not)."""
    config = R.default_config()
    z, links = dict(RUN.GROWN_SQUARED_LENGTHS), dict(RUN.GROWN_LINKS)
    reference = _growth_objective(z, links, config)
    rng = np.random.default_rng(5)
    g = [np.exp(0.7 * rng.normal()) for _ in range(5)]
    moved = _growth_objective(z, _gauge(links, g), config)
    assert abs(moved - reference) < 1e-9 * abs(reference)


@pytest.mark.xfail(strict=True, reason=GROWTH_BRANCH)
def test_the_growth_objective_does_not_turn_on_a_rounding_of_a_length():
    """The squared lengths of the level grown at tick 0 are real up to
    rounding on eight of its ten edges. Measured: with those imaginary
    parts set to zero the objective is 4.0067; with one of them at +1e-16
    it is 2.91, 2.94, 3.50, 4.24, 4.28 or 8.97, depending on the edge."""
    config = R.default_config()
    links = dict(RUN.GROWN_LINKS)
    real = {edge: complex(value.real,
                          value.imag if abs(value.imag) > 1e-10 else 0.0)
            for edge, value in RUN.GROWN_SQUARED_LENGTHS.items()}
    reference = _growth_objective(real, links, config)
    for edge, value in real.items():
        if value.imag != 0.0:
            continue
        moved = dict(real)
        moved[edge] = complex(value.real, 1e-16)
        assert abs(_growth_objective(moved, links, config)
                   - reference) < 1e-9 * abs(reference)


def test_the_growth_step_with_no_update_returns_the_base():
    """Zero updates run no move: the base comes back as it is, with the
    record saying so."""
    config = R.default_config()
    config["pachner_updates"] = 0
    z, links = dict(RUN.GROWN_SQUARED_LENGTHS), dict(RUN.GROWN_LINKS)
    cells, out_z, out_links, record = R.pachner_stage(RUN.TICK1_CELLS, z,
                                                      links, config)
    assert cells == RUN.TICK1_CELLS and out_z == z and out_links == links
    assert record["changed"] is False and record["after"] == record["before"]
