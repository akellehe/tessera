# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The records of a drive whose base complex a committed Pachner move
changed (`tessera.drivers.cell_solve`, `tessera.drivers.recursion`,
`tessera.drivers.baryon_poles`).

Terms used below:

* *S* is the 1-4 subdivision of the base tetrahedron on vertices 0..3
  (`baryon_poles.build_base` at squared edge length 8, with its
  unit-monopole connection), the inserted vertex 4 joined to the corners by
  edges of squared length 1 and link 1, as `AddMove` builds it. A drive of
  the stationarity system of the joint action on its three-sheeted support
  (`cell_solve.GeometricSystem`) commits the 4-1 move that removes vertex 4
  at its first move update, from residual norm 32.2 to 0;
* *S0* is the subdivision of the tetrahedron 0123 by the vertex 4, with the
  tetrahedron 1235 glued on the face 123: squared length 8 on every edge
  but those of vertex 4, the base's links on the edges of 0123 and 1
  elsewhere. The engine's 4-1 move removes vertex 4, and its 1-4 move on
  1235 then inserts a vertex to which it gives the freed id 4, so that the
  cell 1234 and the edges 14, 24 and 34 are named by their ids on S0 and on
  the complex after both moves, and are other cells there;
* *S3* is the subdivision of the tetrahedron 0123 by the vertex 4 (the
  cells of S), squared length 3 on the edges of vertex 4 and 8 on the
  others, the base's links on the edges of 0123 and 1 elsewhere. The 4-1
  move removes vertex 4, and the 1-4 move on the tetrahedron 0123 it leaves
  then inserts a vertex joined to the same four corners by edges of squared
  length 1 and link 1: without fresh vertex ids
  (`MultiCobordism.fresh_vertex_ids`) the engine gives that vertex the
  freed id 4, so that the complex after both moves has S3's cells, named by
  the same ids;
* *F* is the scalar `_CountedEdges` scores a complex with: the sum over its
  edges of the modulus of the squared length less 1, plus a weight times
  the modulus of the number of cells less a count, plus a per-cell term
  times the number of cells; it reads no vertex id;
* a vertex's *name* is the drive's (`cell_solve.VertexNames`), which no two
  vertices share;
* the *places* of a band are the positions of its modes in the ordered
  spectrum of the complex it is read on.
"""
import cmath
import itertools
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R
from tests.drivers import test_cell_solve_python as tc

S0_CELLS = [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4],
            [1, 2, 3, 5]]
S_CELLS = S0_CELLS[:4]
BASE_PHASES = {(1, 2): -math.pi / 2, (1, 3): math.pi / 2,
               (2, 3): -math.pi / 2}


def _s0(moves=0, interior=3.0):
    """S0 with the squared length ``interior`` on the edges of vertex 4,
    after the first ``moves`` of: the engine's 4-1 move (it removes vertex
    4), its 1-4 move (it subdivides 1235 by a vertex it gives the id 4)."""
    edges = sorted({e for c in S0_CELLS for e in itertools.combinations(c, 2)})
    z = {e: complex(interior if 4 in e else 8.0) for e in edges}
    links = {e: cmath.exp(1j * BASE_PHASES.get(e, 0.0)) for e in edges}
    spacetime, _ = R.build_level(S0_CELLS, z, links, sheets=1)
    if moves >= 1:
        move = T.RemoveMove(spacetime, 10, T.PachnerMode.PreGeometric, False)
        assert move.propose() and move.apply()
    if moves >= 2:
        move = T.AddMove(spacetime, 1, False, T.PachnerMode.PreGeometric,
                         False)
        assert move.propose() and move.apply()
    return spacetime


def _s3():
    """S3, one sheet, its vertices 0..4."""
    edges = sorted({e for c in S_CELLS for e in itertools.combinations(c, 2)})
    z = {e: complex(3.0 if 4 in e else 8.0) for e in edges}
    links = {e: cmath.exp(1j * BASE_PHASES.get(e, 0.0)) for e in edges}
    return z, links, R.build_level(S_CELLS, z, links, sheets=1)[0]


class _CountedEdges(cob.CobordismObjective):
    """F, with the weight, the count and the per-cell term given."""

    def __init__(self, weight=0.0, count=0, per_cell=0.0):
        super().__init__()
        self.weight, self.count, self.per_cell = weight, count, per_cell

    def value(self, spacetime):
        cells = len(spacetime.getTopSimplices())
        return (sum(abs(edge.getLength() ** 2 - 1.0)
                    for edge in spacetime.getEdgeList().toVector())
                + self.weight * abs(cells - self.count)
                + self.per_cell * cells)

    def begin(self, *args):
        """No clock: the declared interface of `StationarityObjective`."""

    def name(self):
        return "counted_edges"

    def term_names(self):
        return [cob.ObjectiveTermName.REGGE_STATIONARITY]

    def terms(self, context):
        out = cob.MultiCobordism.ObjectiveTerms()
        out.regge_stationarity = self.value(context.spacetime)
        return out

    def direction(self, context):
        out = cob.ObjectiveDirection()
        out.ascent = np.zeros(context.edge_count, dtype=complex)
        out.baseline = self.value(context.spacetime)
        out.baseline_computed = True
        return out

    def is_target_conditioned(self):
        return False


def _composed(fresh):
    """One stage-1 update of S3 under F with weight 10 and count 4, every
    composition of exactly two moves scored first (a combinatorial length
    of two), on a node of `cell_solve.cell_node` with ``fresh`` vertex ids:
    its trace, the complex it commits, and the drive's names of that
    complex's vertices."""
    base = _s3()[2]
    names = cs.VertexNames()
    names.begin(base)
    node = cs.cell_node(base, _CountedEdges(weight=10.0, count=4))
    node.fresh_vertex_ids = fresh
    trace = node.run_stage1(max_steps=1, n_candidate_moves=0,
                            grow_boundaries=False, max_lookahead=2,
                            combinatorial_breadth=2)
    assert node.last_stage1_lookahead == 2
    after = node.spacetime()
    return list(trace), after, names.of(cs.vertex_ids(after))


def _s():
    base = bp.build_base(8.0)
    move = T.AddMove(base, 0, False, T.PachnerMode.PreGeometric, False)
    assert move.propose() and move.apply()
    return base


def _geometric_system(base):
    """The stationarity system of the joint action on the three-sheeted
    support of ``base``, nothing held."""
    config = bp.default_config(kappas=[1.0], betas=[1.0])
    config["held_sectors"] = []
    host = cs.sheeted_support(base, bp.SHEETS)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"])

    def geometry_of(support):
        return bp.support_geometry(config, support, host)

    return cs.GeometricSystem(declare, geometry_of, bp.SHEETS)


def _content_system(host_base, band_reference):
    """The content system of `test_cell_solve_python.CONTENT` with host
    ``host_base``, as `baryon_poles.relax_content` builds it."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[tuple(tc.CONTENT)])
    config["held_sectors"] = []
    host = cs.sheeted_support(host_base, bp.SHEETS)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"])

    count = bp.fiber_moment_count(
        bp.mean_field_declaration(tc.CONTENT, config),
        cob.JointAction(host.spacetime, declare(host.spacetime)),
        config["fiber_moments"], config["fiber_pinning"])

    def mean_field_of(support):
        declaration = bp.mean_field_declaration(tc.CONTENT, config)
        declaration.geometry = bp.support_geometry(config, support, host)
        declaration.fiber_moments = count
        return declaration

    return cs.ContentSystem(declare, mean_field_of, host_base, bp.SHEETS,
                            band_reference)


# ------------------------------------------------------ the names of vertices


def test_the_engine_gives_a_freed_id_to_the_vertex_it_inserts():
    """After the 4-1 move frees the id 4, the 1-4 move on another
    tetrahedron gives the vertex it inserts that id: the cell 1234 is named
    by the same ids before and after, and is another cell."""
    before, removed, after = _s0(), _s0(1), _s0(2)
    assert sorted(cs.top_cells(removed)) == [(0, 1, 2, 3), (1, 2, 3, 5)]
    assert sorted(cs.top_cells(after)) == [
        (0, 1, 2, 3), (1, 2, 3, 4), (1, 2, 4, 5), (1, 3, 4, 5),
        (2, 3, 4, 5)]
    assert set(cs.top_cells(before)) & set(cs.top_cells(after)) == {
        (1, 2, 3, 4)}


def test_a_vertex_a_move_inserts_takes_a_name_no_vertex_had():
    """The drive names the vertices of its first complex by their ids. A
    vertex keeps its name while its id is held at the accepted points; a
    vertex whose id was not held at the last accepted point takes a name
    above every name given, so the vertex inserted after the removal is 6,
    not 4, and no cell after the two moves is named as a cell of S0. A name
    is not given again: the vertex inserted after 6 was removed is 7."""
    before, removed, after = _s0(), _s0(1), _s0(2)
    names = cs.VertexNames()
    assert not names.begun
    assert names.of([0, 4, 9]) == {0: 0, 4: 4, 9: 9}
    names.begin(before)
    assert names.of(cs.vertex_ids(before)) == {v: v for v in range(6)}
    names.accept(removed)
    assert names.of(cs.vertex_ids(removed)) == {0: 0, 1: 1, 2: 2, 3: 3,
                                                 5: 5}
    named = names.of(cs.vertex_ids(after))
    assert named == {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 5}
    cells_before = {tuple(sorted(c)) for c in cs.top_cells(before)}
    cells_after = {tuple(sorted(named[v] for v in c))
                   for c in cs.top_cells(after)}
    assert cells_before & cells_after == set()
    names.accept(after)
    names.accept(removed)
    assert names.of(cs.vertex_ids(after))[4] == 7


def test_a_vertex_reinserted_within_one_update_takes_a_new_name():
    """The removal and the insertion committed as one update of two moves:
    on S3, F (weight 10, count 4) is 50, no single move lowers it, and the
    4-1 move followed by the 1-4 move on the tetrahedron 0123 lowers it to
    42, the edges of the vertex inserted at squared length 1. The node of a
    drive declares fresh vertex ids, so the vertex inserted takes the id 5,
    which no complex of the drive held, and the drive names it 5: no cell
    after the update is named as a cell of S3, and its edges are not taken
    for those of the vertex removed."""
    trace, after, named = _composed(fresh=True)
    assert trace == pytest.approx([50.0, 42.0], rel=1e-15)
    assert sorted(cs.top_cells(after)) == [(0, 1, 2, 5), (0, 1, 3, 5),
                                           (0, 2, 3, 5), (1, 2, 3, 5)]
    assert named == {0: 0, 1: 1, 2: 2, 3: 3, 5: 5}
    assert not {tuple(sorted(named[v] for v in c))
                for c in cs.top_cells(after)} & set(cs.top_cells(_s3()[2]))
    assert {(a, b): (length * length).real
            for a, b, length, _ in cs.edge_fields(after) if 5 in (a, b)} == {
        (0, 5): 1.0, (1, 5): 1.0, (2, 5): 1.0, (3, 5): 1.0}


def test_fresh_vertex_ids_change_only_the_id_of_the_vertex_inserted():
    """The same update on a node without fresh vertex ids, the engine's
    default: the same composition is committed with the same trace, and the
    vertex inserted takes the freed id 4, so the complex has S3's cells
    named by S3's ids and the drive's names take the vertex inserted for the
    vertex removed. With fresh vertex ids the complex is the same on the
    relabeling 5 -> 4, every squared length and link the same."""
    fresh_trace, fresh, _ = _composed(fresh=True)
    trace, after, named = _composed(fresh=False)
    assert trace == fresh_trace
    assert sorted(cs.top_cells(after)) == sorted(cs.top_cells(_s3()[2]))
    assert named == {v: v for v in range(5)}
    relabel = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
    assert sorted(tuple(sorted(relabel[v] for v in c))
                  for c in cs.top_cells(fresh)) == sorted(
        cs.top_cells(after))
    moved = {tuple(sorted((relabel[a], relabel[b]))): (length, phase)
             for a, b, length, phase in cs.edge_fields(fresh)}
    assert moved == {(min(a, b), max(a, b)): (length, phase)
                     for a, b, length, phase in cs.edge_fields(after)}


def test_a_growth_step_of_two_updates_names_the_vertex_it_reinserts(
        monkeypatch):
    """The growth step (`recursion.pachner_stage`) of two updates on S3,
    scored by F with a per-cell term of -1 in place of the joint action's
    stationarity: F is 46, the first update commits the 4-1 move (41, below
    the 43 of a 1-4 move) and the second the 1-4 move on the tetrahedron
    0123 that leaves (38). The vertex inserted has the search's id 5, so
    ``vertex_relabeling`` has no entry for the vertex 4 the first update
    removed, the cells changed although they are S3's on the labels 0..4,
    and the lineage of a response vertex on the vertex 4 of the grown base
    is None, not the vertex inserted."""
    z, links, _ = _s3()
    objective = _CountedEdges(per_cell=-1.0)

    class Begun:
        def begin(self, base):
            pass

    monkeypatch.setattr(R, "level_system",
                        lambda base, config, sectors, sheets: (Begun(), None))
    monkeypatch.setattr(cs, "StationarityObjective", lambda system: objective)
    config = R.default_config(pachner_updates=2)
    assert config["pachner_objective"] == "joint-action"
    cells, _, _, record = R.pachner_stage(S_CELLS, z, links, config)
    assert record["trace"] == pytest.approx([46.0, 41.0, 38.0], rel=1e-15)
    assert record["vertex_relabeling"] == {"0": 0, "1": 1, "2": 2, "3": 3,
                                           "5": 4}
    assert cells == [sorted(c) for c in S_CELLS]
    assert record["changed"]
    lineage = R.growth_lineage({0: 0, 1: 4, 2: None}, record)
    assert lineage == {"0": 0, "1": None, "2": None}
    assert R.growth_lineage({0: 4}, {"updates": 0}) == {"0": 4}


def test_a_reference_band_carries_no_weight_onto_a_reinserted_vertex():
    """A content followed from its host S0 (`BAND_REFERENCES` "host"),
    read after the two moves: the edges of the inserted vertex are cells
    the host does not have, so the reference projectors carry nothing onto
    them. Named by their ids, the edges 14, 24 and 34 are taken for the
    host's edges of the removed vertex and carry its entries (up to 0.058
    in the first band)."""
    before, removed, after = _s0(), _s0(1), _s0(2)
    system = _content_system(before, "host")
    system.begin(before)
    system.accept(removed)
    field, action, support = system._field(after)
    reference, cells = system._reference(support, action)
    assert len(cells) == 39
    assert all(4 not in edge for _, edge in cells)
    new = [index for index, (_, edge) in enumerate(cells) if 6 in edge]
    assert len(new) == 12
    by_id = cs.reference_on(
        cs.support_cells(support), cs.support_cells(
            cs.sheeted_support(before, bp.SHEETS)), system.reference)
    for band, taken in zip(reference, by_id):
        projector = np.asarray(band.projector).reshape(39, 39)
        assert not projector[new, :].any() and not projector[:, new].any()
        wrong = np.asarray(taken.projector).reshape(39, 39)
        assert np.abs(wrong[np.ix_(new, new)]).max() > 0.05


def test_an_edge_of_a_reinserted_vertex_starts_where_it_is_scored():
    """The Regge sheets of a drive begun on S0 (the edges of vertex 4 at
    squared length 3): after the removal is accepted, an edge of the
    inserted vertex has no start and starts at its own squared length, 1;
    named by its ids it would start at 3, on the sheet of the removed
    vertex's edge."""
    before, removed, after = _s0(), _s0(1), _s0(2)
    start = cs.ReggeStart()
    start.begin(before)
    start.accept(removed)
    declared = start.declared(cob.JointActionDeclaration(),
                              cs.sheeted_support(after, 1))
    fields = cs.edge_fields(after)
    starts = list(declared.regge_start_squared_lengths)
    assert len(starts) == len(fields) == 13
    for (a, b, length, _), squared in zip(fields, starts):
        if 4 in (a, b):
            assert squared == length * length == pytest.approx(1.0,
                                                               rel=1e-15)
        else:
            assert squared == pytest.approx(8.0, rel=1e-15)


def test_the_covariance_change_is_measured_on_the_same_cells():
    """The covariance change of a content's measurement is measured from
    the covariance of the last accepted point when the point has its cells:
    on S0 displaced, 0.0274, the distance between the two covariances. The
    complex after both moves, committed as one update, has as many cells
    (39) and other ones: no change is measured there, where comparing the
    covariances entry by entry would read 5.47 (S0 with unit edges at vertex
    4, the squared length the inserted vertex's edges have)."""
    before, after = _s0(interior=1.0), _s0(2, interior=1.0)
    system = _content_system(before, "previous")
    system.begin(before)
    system.accept(before)
    step, cells = system.iterate(after)
    assert len(cells) == len(system._covariance_cells) == 39
    assert cells != system._covariance_cells
    assert step.covariance_change == 0.0
    assert np.linalg.norm(np.asarray(cs.covariance_of(step.bands))
                          - np.asarray(system._covariance)) \
        == pytest.approx(5.474, rel=1e-3)
    displaced = _s0(interior=1.0)
    for edge in displaced.getEdgeList().toVector():
        edge.setPhase(edge.getPhase() + 0.01)
    step, cells = system.iterate(displaced)
    assert cells == system._covariance_cells
    distance = np.linalg.norm(np.asarray(cs.covariance_of(step.bands))
                              - np.asarray(system._covariance))
    assert step.covariance_change == pytest.approx(distance, rel=1e-13)
    assert distance == pytest.approx(0.02739, rel=1e-3)


# ------------------------------------------------------ a committed 4-1 move


def test_a_drive_records_the_names_of_the_vertices_it_ended_on():
    """The drive from S commits the 4-1 move and ends on the tetrahedron
    0123, whose vertices keep their names; the drive from S0 ends on 0123
    and 1235."""
    drive = cs.solve(_s(), _geometric_system(_s()))
    assert drive["moves_committed"] == 1 and drive["changed"]
    assert drive["trace"] == pytest.approx([32.2138, 0.0], abs=1e-4)
    assert cs.top_cells(drive["spacetime"]) == [(0, 1, 2, 3)]
    assert drive["vertex_names"] == {0: 0, 1: 1, 2: 2, 3: 3}
    base = _s0()
    drive = cs.solve(base, _geometric_system(base))
    assert drive["moves_committed"] == 1
    assert sorted(cs.top_cells(drive["spacetime"])) == [(0, 1, 2, 3),
                                                        (1, 2, 3, 5)]
    assert drive["vertex_names"] == {0: 0, 1: 1, 2: 2, 3: 3, 5: 5}


def test_the_pole_read_is_made_on_a_tetrahedron_a_committed_move_left():
    """`geometry_without_value` reads the base a drive ended on: the drive
    from S ends on one tetrahedron, which is read; the drive from S0 ends
    on two, which have no pole read."""
    host = bp.build_host()
    drive = cs.solve(_s(), _geometric_system(_s()))
    assert bp.geometry_without_value(host, drive) is None
    base = _s0()
    drive = cs.solve(base, _geometric_system(base))
    assert bp.geometry_without_value(host, drive) == (
        bp.NO_VALUE_MOVED,
        bp.NO_VALUE_MOVED + " (1 committed move updates left a base complex "
        "of 5 vertices, 9 edges and 2 cells), and the pole read is defined "
        "on the three-sheeted tetrahedron")


@pytest.mark.parametrize("band_reference", cs.BAND_REFERENCES)
def test_a_band_is_at_its_places_on_the_tetrahedron_a_committed_4_1_left(
        band_reference):
    """The content chosen on S (its band at places 6, 7, 8 of 30), read on
    the tetrahedron the drive from S ended on: the band is at the places it
    holds at the first measurement there, 3, 4, 5 of 18, and the read
    reports no crossing. With the host's places carried over, the same read
    takes the same modes and reports a crossing."""
    host = _s()
    drive = cs.solve(_s(), _geometric_system(_s()))
    tetrahedron = drive["spacetime"]
    system = _content_system(host, band_reference)
    assert list(system._places.values()) == [[[6, 7, 8]]]
    field, action, support = system._field(tetrahedron)
    carried, cells = system._reference(support, action)
    assert len(cells) == 18
    with_host_places = field.iterate(carried, [])
    assert with_host_places.band_crossing
    assert [list(b.declared_positions) for b in with_host_places.bands] == \
        [[6, 7, 8]]
    step = system.accept(tetrahedron)
    assert not step.band_crossing
    assert [list(b.positions) for b in step.bands] == [[3, 4, 5]]
    assert [list(b.declared_positions) for b in step.bands] == [[3, 4, 5]]
    for band, other in zip(step.bands, with_host_places.bands):
        assert list(band.modes) == list(other.modes)
    assert system._places[tuple(cells)] == [[3, 4, 5]]


def test_the_records_of_a_level_a_committed_4_1_changed(monkeypatch):
    """The tick on S relabeled, its inserted vertex 0 and the tetrahedron's
    corners 1..4, as the declared level 0 (the tetrahedron's monopole
    connection, the edges of vertex 0 at squared length 1 and link 1, its
    bounding cut held): the relaxation commits the 4-1 move, and the moved
    level is the tetrahedron on its vertices relabeled 0..3.

    The monopole numbers before, [-1, 0, 0, 0], are of the four cells
    before, each read on its ascending vertices; the number after, [1], is
    of the cell after, named by the drive's vertex names [1, 2, 3, 4] and by
    the level's [0, 1, 2, 3]. The held cut is oriented coherently on the
    subdivision, from the ascending order of its first cell 0123, which is
    opposite to the tetrahedron 1234's: its faces are the tetrahedron's
    outward faces reversed, and it reads the one unit the subdivision
    encloses as -1, which is the sum of the numbers before weighted by the
    cells' orientations (`cell_orientations`, [1, -1, 1, -1]), before and
    after the move."""
    cells = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]]
    connection = R.monopole_connection([[1, 2, 3, 4]])
    links = {e: cmath.exp(1j * p) for e, p in zip(connection["edges"],
                                                  connection["phases"])}
    z = {e: 8.0 + 0j for e in connection["edges"]}
    for v in range(1, 5):
        links[(0, v)] = 1.0 + 0j
        z[(0, v)] = 1.0 + 0j

    def reads(cells, z, links, config):
        return [{"cell": sorted(c), "host_cell": {}, "failed_contents": [],
                 "flagged_contents": [], "contents": [], "ratios": {},
                 "pole_table": {}} for c in cells]

    def no_turn(operator, config):
        raise ValueError("stop here")

    monkeypatch.setattr(R, "cell_reads", reads)
    monkeypatch.setattr(R, "recursion_turn", no_turn)
    config = R.default_config(tetrahedra=2, pachner_updates=0)
    record, _ = R.tick(0, cells, z, links, config)
    relaxation = record["relaxation"]
    assert relaxation["moves_committed"] == 1 and relaxation["changed"]
    assert relaxation["vertex_relabeling"] == {"1": 0, "2": 1, "3": 2,
                                               "4": 3}
    level = record["level"]
    assert level["cells_before"] == cells
    numbers = level["bulk_monopole_numbers_before"]
    assert numbers == [-1, 0, 0, 0]
    signs = R.cell_orientations(cells)
    assert signs == [1, -1, 1, -1]
    assert level["cells"] == [[0, 1, 2, 3]]
    assert level["cells_after"] == [[1, 2, 3, 4]]
    assert level["bulk_monopole_numbers_after"] == [1]
    held = level["held_cut"]

    def cyclic(face):
        low = face.index(min(face))
        return tuple(face[low:] + face[:low])
    assert sorted(cyclic(list(f)) for f in held["faces"]) == sorted(
        cyclic(list(reversed(f))) for f in R.outward_faces([1, 2, 3, 4]))
    assert held["faces_after"] == [[v - 1 for v in f] for f in held["faces"]]
    assert held["faces_after_on_boundary"] is True
    assert held["monopole_number_before"] == held["monopole_number_after"] \
        == sum(s * n for s, n in zip(signs, numbers)) == -1
    text = R.summary({"host": {"monopole_numbers": [1]}, "ticks": [record]})
    assert ("committed moves changed the level's cells: the numbers before "
            "are of the cells [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], "
            "[0, 2, 3, 4]], those after of the cells [[1, 2, 3, 4]] (by the "
            "drive's vertex names; on the level's vertices [[0, 1, 2, 3]])"
            ) in text
