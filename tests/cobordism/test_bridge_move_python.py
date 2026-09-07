# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The bridge move and the collar seed (#960, T1 of
``docs/design/qubit_cobordism_spec.md``).

Two qubit tori (``SimplicialQubit.flat_torus``) can be seeded into one
3-dimensional host as their own triangles with their lengths and no 3-cell
(``MultiCobordism.seed_from_surfaces``), each torus one input block; that is
the host the bridge primitive is exercised on. ``SurgicalCone.bridge`` creates
a tetrahedron on existing vertices split across the two blocks (1+3, 2+2,
3+1), gated by the manifold check and nothing else, rolled back bit-exactly
on refusal. The experiment's host is the COLLAR (``MultiCobordism.seed_collar``,
spec S3): the minimal manifold connecting the two tori, the prism over their
shared triangulation created as one gated whole, so that the boundary of W is
exactly the two tori by construction; everything after it is emergent, with
``bridge`` one more gated stage-1 move kind. A per-cell drawing from nothing
cannot reach that boundary (a cell the gate accepts meets the complex along a
disk, so the drawing stays a ball), which is why the collar is the seed.

Coverage:

* the bare seed holds each torus exactly (faces, edges, lengths, zero phases,
  disjoint id ranges, no 3-cell) and the blocks' own surfaces are the tori;
* the gate refuses a non-manifold bridge (a cell sharing only a vertex or only
  an edge with the drawing, a third cell on a face) and malformed input,
  leaving the complex unchanged;
* an accepted bridge round-trips bit-exactly, including after the cell's face
  lattice was materialized by a read, and LIFO with a second bridge;
* the collar seed from 3x3 and from 4x4 tori is a manifold with boundary
  exactly T_A and T_B, complete by construction, chordless, every cell
  straddling the blocks, the tori's lengths verbatim and the interior edges
  auto-wired, Betti numbers [1, 2, 1, 0], monodromy the identity for the
  markings carried through the id maps (swap and reversal of the far marking
  give the swap and sign matrices); interior layers are fresh vertices; a
  mismatched pair is refused by name;
* on the collar every bridge the gate refuses is refused by name with the
  complex unchanged, an accepted one rolls back bit-exactly, the stage-1
  bridge kind has nothing to draw, and a cone-out dent reopens the phase;
* stage 1 never sees the bridge kind on an ordinary node, and a committed
  rebuild keeps uncovered torus faces.
"""
import math
import os

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def torus(tau, n):
    return obs.SimplicialQubit.flat_torus(tau, n, n)


def mapped_faces(q, ids):
    return sorted(tuple(sorted(ids[int(v)] for v in f)) for f in q.faces())


def mapped_edges(q, ids):
    return sorted(tuple(sorted((ids[int(i)], ids[int(j)]))) for i, j in q.edges())


def mapped_lengths(q, ids):
    return {tuple(sorted((ids[int(i)], ids[int(j)]))): l for (i, j), l in zip(q.edges(), q.lengths())}


def host_marking(q, ids):
    """The torus's marking as cycles of directed host steps (u -> v)."""
    edges = q.edges()

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = edges[int(e)]
            out.append((ids[int(i)], ids[int(j)]) if sign > 0 else (ids[int(j)], ids[int(i)]))
        return out

    return [cycle(q.cycle_A()), cycle(q.cycle_B())]


def tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices())) for s in st.getTopSimplices())


def registered(st, size):
    return sorted(
        tuple(sorted(v.getId() for v in s.getVertices()))
        for s in st.getSimplices() if len(s.getVertices()) == size
    )


def edge_geometry(st):
    out = {}
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        out[tuple(sorted((u, v)))] = (complex(e.getLength()), complex(e.getPhase()))
    return out


def vertex_ids(st):
    return sorted(v.getId() for v in st.getVertexList().toVector())


def boundary(st):
    return sorted(tuple(f) for f in st.getBoundary())


def surface_cells(st):
    """Registered triangles that are facets of no top cell: the surfaces' own
    cells. Facets of surviving top cells are lazily materialized bookkeeping a
    read re-creates identically, so a rollback leaves them as it found them."""
    covered = {tuple(sorted(set(cell) - {v})) for cell in tops(st) for v in cell}
    return [f for f in registered(st, 3) if f not in covered]


def state(st):
    return (tops(st), surface_cells(st), edge_geometry(st), vertex_ids(st))


def seeded(seed_value=0, einstein_hilbert=False):
    qa, qb = torus(TAU_A, 3), torus(TAU_B, 4)
    seed = MC.seed_from_surfaces([qa.spacetime(), qb.spacetime()])
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value,
              einstein_hilbert=einstein_hilbert)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    return qa, qb, seed, node


def block_split(cell, regions):
    return tuple(sum(1 for v in cell if v in region) for region in regions)


# --------------------------------------------------------------------------- #
# 1. the seed holds the tori exactly
# --------------------------------------------------------------------------- #
def test_seed_from_surfaces_holds_the_tori():
    qa, qb, seed, node = seeded()
    st = seed.host
    assert tops(st) == [], "the seed must hold no 3-cell"
    assert registered(st, 4) == []
    assert boundary(st) == []
    ids_a, ids_b = seed.vertex_ids
    assert sorted(ids_a.values()) == list(range(9))
    assert sorted(ids_b.values()) == list(range(9, 25))
    assert vertex_ids(st) == list(range(25))
    assert registered(st, 3) == sorted(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    geometry = edge_geometry(st)
    assert sorted(geometry) == sorted(mapped_edges(qa, ids_a) + mapped_edges(qb, ids_b))
    for q, ids in ((qa, ids_a), (qb, ids_b)):
        for edge, length in mapped_lengths(q, ids).items():
            assert geometry[edge] == (complex(length), 0j), edge
    # each torus is one input block, marked as a surface; its own surface is the torus
    assert len(node.inputs) == 2 and node.has_surface_inputs()
    for block, q, ids in zip(node.inputs, (qa, qb), (ids_a, ids_b)):
        assert block.surface
        assert sorted(block.vertices) == sorted(ids.values())
        surface = MC.block_surface(block, st)
        assert sorted(tuple(f) for f in surface.faces) == mapped_faces(q, ids)
        assert sorted(tuple(e) for e in surface.edges) == mapped_edges(q, ids)
    assert len(node.uncovered_input_faces()) == 18 + 32
    assert not node.bridge_phase_complete()
    with pytest.raises(ValueError):
        MC.seed_from_surfaces([])
    with pytest.raises(ValueError, match="differ in dimension"):
        MC.seed_from_surfaces([qa.spacetime(), MC.seed_simplex(3)])


# --------------------------------------------------------------------------- #
# 2. the gate refuses a non-manifold bridge and leaves the complex unchanged
# --------------------------------------------------------------------------- #
def test_bridge_gate_refuses_non_manifold_cells():
    qa, qb, seed, node = seeded()
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    faces_a = mapped_faces(qa, ids_a)
    b0 = ids_b[0]
    sc = cob.SurgicalCone(st)
    # malformed input, nothing applied
    for bad in ([0, 1, 2], [0, 0, 1, b0], [0, 1, 2, 999]):
        ok, reason = sc.bridge(bad)
        assert not ok, reason
    assert state(st) == state(seed.host) and sc.depth == 0
    before = state(st)
    face = faces_a[0]
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    assert sc.depth == 1 and sc.validate()[0]
    assert tops(st) == [tuple(sorted(face + (b0,)))]
    ok, reason = sc.bridge(list(face) + [b0])
    assert not ok and "already exists" in reason
    # a cell sharing only the vertex b0 with the drawing: b0's link is two
    # disjoint triangles, a pinch
    disjoint = next(f for f in faces_a if not set(f) & set(face))
    ok, reason = sc.bridge(list(disjoint) + [b0])
    assert not ok, "a cell sharing only a vertex is not a manifold"
    assert "pinch" in reason or "disconnected" in reason, reason
    # a cell sharing only an edge: that edge's link is two disjoint paths
    shared_edge = next(f for f in faces_a if len(set(f) & set(face)) == 2)
    other = ids_b[5]
    ok, reason = sc.bridge(list(shared_edge) + [other])
    assert not ok, "a cell sharing only an edge is not a manifold"
    assert sc.depth == 1 and tops(st) == [tuple(sorted(face + (b0,)))]
    # a third cell on one face: three cofaces
    b1 = next(v for (u, v) in [tuple(sorted(e)) for e in mapped_edges(qb, ids_b)] if u == b0)
    ok, reason = sc.bridge([face[0], face[1], b0, b1])
    assert ok, reason
    ok, reason = sc.bridge([face[0], face[1], b0, ids_b[7]])
    assert not ok and "cofaces" in reason, reason
    assert sc.depth == 2
    assert sc.rollbackAll() == 2
    assert state(st) == before, "the refused and rolled-back bridges must leave the complex unchanged"


# --------------------------------------------------------------------------- #
# 3. bit-exact rollback, also after the lattice was materialized, and LIFO
# --------------------------------------------------------------------------- #
def test_bridge_rollback_is_bit_exact():
    qa, qb, seed, node = seeded()
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    face = mapped_faces(qa, ids_a)[4]
    b0 = ids_b[3]
    before = state(st)
    sc = cob.SurgicalCone(st)
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    after = edge_geometry(st)
    fresh = sorted(set(after) - set(before[2]))
    assert fresh == sorted(tuple(sorted((v, b0))) for v in face), "a bridge wires exactly the missing edges"
    for edge in fresh:
        assert after[edge] == (1 + 0j, 0j), "auto-wired spacelike unit length, zero phase"
    for edge, geometry in before[2].items():
        assert after[edge] == geometry, "a bridge never touches an existing edge"
    # materialize the cell's face lattice through the reads the engine makes
    K = cob.ChainComplex.fromSpacetime(st)
    assert K.bettiNumbers() == [1, 0, 0, 0]  # the drawn 3-ball alone (fromSpacetime seeds from the top cells)
    assert sc.bettiNumbers() == [1, 0, 0, 0]
    assert sc.rollback()
    assert state(st) == before, "round trip after materialization"
    # LIFO with two bridges sharing a face, materialized in between
    ok, reason = sc.bridge(list(face) + [b0])
    assert ok, reason
    mid = state(st)
    b1 = next(v for (u, v) in [tuple(sorted(e)) for e in mapped_edges(qb, ids_b)] if u == b0)
    ok, reason = sc.bridge([face[0], face[1], b0, b1])
    assert ok, reason
    cob.ChainComplex.fromSpacetime(st).bettiNumbers()
    assert sc.rollback()
    assert state(st) == mid
    assert sc.rollback()
    assert state(st) == before
    assert registered(st, 3) == before[1], "no materialized face survives the full unwind"
    assert not sc.rollback()


# --------------------------------------------------------------------------- #
# 4. the collar seed: the minimal manifold connecting the two tori (spec S3)
# --------------------------------------------------------------------------- #
def collar(n, layers=1, seed_value=0):
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), layers)
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value, einstein_hilbert=False)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    return qa, qb, seed, node


@pytest.mark.parametrize("n", [3, 4])
def test_collar_seed_is_the_manifold_between_the_tori(n, whitney_default):
    qa, qb, seed, node = collar(n)
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    assert sorted(ids_a.values()) == list(range(n * n))
    assert sorted(ids_b.values()) == list(range(n * n, 2 * n * n))
    assert vertex_ids(st) == list(range(2 * n * n)), "no interior vertex in a one-layer collar"
    # one gated whole: a manifold-with-boundary whose boundary is exactly T_A and T_B
    ok, reason = cob.SurgicalCone(st).validate()
    assert ok, reason
    torus_faces = sorted(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    assert boundary(st) == torus_faces
    assert len(tops(st)) == 3 * len(q_faces := qa.faces()) == 6 * n * n
    splits = {block_split(cell, regions) for cell in tops(st)}
    assert splits == {(1, 3), (2, 2), (3, 1)}
    # the two surfaces are the node's surface input blocks; completion holds by construction
    assert node.has_surface_inputs() and all(b.surface for b in node.inputs)
    assert node.bridge_phase_complete()
    assert node.uncovered_input_faces() == []
    for block, q, ids in zip(node.inputs, (qa, qb), (ids_a, ids_b)):
        surface = MC.block_surface(block, st)
        assert sorted(tuple(f) for f in surface.faces) == mapped_faces(q, ids), "no chord face"
        assert sorted(tuple(e) for e in surface.edges) == mapped_edges(q, ids), "no chord edge"
        assert not any(set(cell) <= set(block.vertices) for cell in tops(st))
    # the surfaces' lengths verbatim, zero phases; every other edge the auto-wired length
    geometry = edge_geometry(st)
    surface_lengths = {**mapped_lengths(qa, ids_a), **mapped_lengths(qb, ids_b)}
    for edge, (length, phase) in geometry.items():
        assert phase == 0j
        if edge in surface_lengths:
            assert length == complex(surface_lengths[edge])
        else:
            assert not (edge[0] in regions[0] and edge[1] in regions[0])
            assert not (edge[0] in regions[1] and edge[1] in regions[1])
            assert length == 1 + 0j
    assert set(surface_lengths) <= set(geometry)
    # Betti numbers and the monodromy of the seed: the collar, identity for matched markings
    assert cob.ChainComplex.fromSpacetime(st).bettiNumbers() == [1, 2, 1, 0]
    marking_a, marking_b = host_marking(qa, ids_a), host_marking(qb, ids_b)
    read = MC.monodromy(st, marking_a, marking_b)
    assert read.obstruction == "", read.obstruction
    assert read.betti == [1, 2, 1, 0] and read.harmonic_rank == 2
    np.testing.assert_allclose(np.asarray(read.monodromy), np.eye(2), atol=1e-9)
    assert read.rounded == [[1, 0], [0, 1]]
    assert read.rounding_residual < 1e-9 and read.fit_residual < 1e-9
    a, b = marking_b
    swapped = MC.monodromy(st, marking_a, [b, a])
    assert swapped.rounded == [[0, 1], [1, 0]] and swapped.rounding_residual < 1e-9
    flipped = MC.monodromy(st, marking_a, [a, [(v, u) for (u, v) in reversed(b)]])
    assert flipped.rounded == [[1, 0], [0, -1]] and flipped.rounding_residual < 1e-9
    # the periods are taken with parallel transport (T2-bis), so a cycle must be one closed
    # walk: a single step is refused by name before any operator is built ...
    absent = MC.monodromy(st, marking_a, [[(0, 2 * n * n - 1)], b])
    assert "does not form one closed walk" in absent.obstruction, absent.obstruction
    # ... and a closed walk across a pair that is not an edge of the whole is named as such
    edges = {tuple(int(v) for v in e) for e in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(1)}
    u, v = next((u, v) for u in range(2 * n * n) for v in range(u + 1, 2 * n * n) if (u, v) not in edges)
    missing = MC.monodromy(st, marking_a, [[(u, v), (v, u)], b])
    assert "not an edge of the whole" in missing.obstruction, missing.obstruction
    print(f"\n[bridge T1] collar {n}x{n}: cells={len(tops(st))} betti={read.betti} monodromy rounded={read.rounded} "
          f"rounding residual={read.rounding_residual:.1e} fit residual={read.fit_residual:.1e}")


def test_collar_with_interior_layers(whitney_default):
    qa, qb, seed, node = collar(3, layers=2)
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    assert sorted(ids_a.values()) == list(range(9)) and sorted(ids_b.values()) == list(range(18, 27))
    assert vertex_ids(st) == list(range(27)), "one layer of fresh interior vertices"
    ok, reason = cob.SurgicalCone(st).validate()
    assert ok, reason
    assert boundary(st) == sorted(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    assert node.bridge_phase_complete() and len(tops(st)) == 2 * 54
    assert cob.ChainComplex.fromSpacetime(st).bettiNumbers() == [1, 2, 1, 0]
    read = MC.monodromy(st, host_marking(qa, ids_a), host_marking(qb, ids_b))
    assert read.obstruction == "" and read.rounded == [[1, 0], [0, 1]] and read.rounding_residual < 1e-9


def test_collar_refuses_mismatched_surfaces():
    qa, qb = torus(TAU_A, 3), torus(TAU_B, 4)
    with pytest.raises(ValueError, match="differ in combinatorics: surface A has 9 vertices, surface B 16"):
        MC.seed_collar(qa.spacetime(), qb.spacetime())
    # same vertex count, different faces: the intrinsic Delaunay pass flips edges
    flipped = qa.intrinsic_delaunay()
    assert flipped.delaunay_flip_count() > 0
    with pytest.raises(ValueError, match=r"differ in combinatorics: face \(\d+,\d+,\d+\) of surface A"):
        MC.seed_collar(qa.spacetime(), flipped.spacetime())
    with pytest.raises(ValueError, match="layers must be at least one"):
        MC.seed_collar(qa.spacetime(), torus(TAU_B, 3).spacetime(), 0)
    with pytest.raises(ValueError, match="differ in dimension"):
        MC.seed_collar(qa.spacetime(), MC.seed_simplex(3))


# --------------------------------------------------------------------------- #
# 5. a bridge on the collar: the gate's verdict, the complex unchanged
# --------------------------------------------------------------------------- #
def test_bridge_on_the_collar(whitney_default):
    qa, qb, seed, node = collar(3)
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    before = state(st)
    sc = cob.SurgicalCone(st)
    a_faces = mapped_faces(qa, ids_a)
    a_edges = mapped_edges(qa, ids_a)
    b_edges = mapped_edges(qb, ids_b)
    b_faces = mapped_faces(qb, ids_b)
    # every split cell that does not already exist is tried; the collar is
    # complete, so the gate must refuse each one (a face gains a third coface,
    # or an edge or vertex link breaks) and name the reason
    existing = set(tops(st))
    verdicts = {}
    candidates = ([tuple(f) + (b,) for f in a_faces for b in sorted(ids_b.values())]
                  + [tuple(e) + tuple(g) for e in a_edges for g in b_edges]
                  + [(a,) + tuple(g) for a in sorted(ids_a.values()) for g in b_faces])
    accepted = 0
    for cell in candidates:
        if tuple(sorted(cell)) in existing:
            continue
        ok, reason = sc.bridge(list(cell))
        if ok:
            accepted += 1
            assert sc.validate()[0]
            assert sc.rollback()
            assert state(st) == before, "an accepted bridge on the collar must roll back bit-exactly"
        else:
            verdicts[reason.split(":")[0].split(" ")[0]] = verdicts.get(reason.split(":")[0].split(" ")[0], 0) + 1
            assert reason, "a refusal names its reason"
    assert sc.depth == 0 and state(st) == before
    print(f"\n[bridge T1] bridges on the 3x3 collar: {sum(v for v in verdicts.values()) + accepted} "
          f"candidates, accepted={accepted}, refusals by kind={verdicts}")
    # the stage-1 bridge kind has nothing to draw on a complete collar ...
    assert node.uncovered_input_faces() == [] and node.bridge_phase_complete()
    node.run_stage1(max_steps=2, n_candidate_moves=6)
    st2 = node.spacetime()
    for block, q, ids in zip(node.inputs, (qa, qb), (ids_a, ids_b)):
        assert sorted(tuple(e) for e in MC.block_surface(block, st2).edges) == mapped_edges(q, ids)
    # ... and a cone-out dent reopens the phase: the dented torus face leaves W
    # and its edges persist; rolling the dent back restores completion. The
    # gate decides which cells may be dented (removing a cell can pinch a
    # vertex link); the first it accepts is taken, and a refusal is named.
    st = node.spacetime()
    sc = cob.SurgicalCone(st)
    complete_before = node.bridge_phase_complete()
    refusals = []
    for cell in tops(st):
        if block_split(cell, [set(ids_a.values()), set(ids_b.values())]) != (3, 1):
            continue
        ok, reason = sc.coneOut(list(cell))
        if not ok:
            assert reason
            refusals.append(reason)
            continue
        assert not node.bridge_phase_complete()
        assert sorted(tuple(e) for e in MC.block_surface(node.inputs[0], st).edges) == mapped_edges(qa, ids_a)
        assert sc.rollback()
        assert node.bridge_phase_complete() == complete_before
        break
    print(f"[bridge T1] cone-out dents on the 3x3 collar: accepted after {len(refusals)} refusals "
          f"({refusals[:2]})")


# --------------------------------------------------------------------------- #
# 6. ordinary nodes never see the bridge kind; a rebuild keeps uncovered faces
# --------------------------------------------------------------------------- #
def test_ordinary_nodes_are_untouched_and_rebuilds_keep_the_tori(whitney_default):
    ordinary = MC(MC.seed_simplex(3), [[1.0 + 0j]], [], degrees=[1], einstein_hilbert=False)
    ordinary.seed_inputs([0])
    assert not ordinary.has_surface_inputs()
    assert not ordinary.inputs[0].surface
    assert not ordinary.bridge_phase_complete()
    assert ordinary.uncovered_input_faces() == []
    assert not hasattr(MC.BuildAction, "BRIDGE")
    with pytest.raises(ValueError, match="not a vertex of the host"):
        ordinary.seed_inputs([[0, 99]])

    # On the bare surfaces one accepted bridge leaves 49 faces uncovered. A
    # committed move rebuilds the complex from a snapshot; the uncovered torus
    # faces must ride along. refine_geometry commits a gated cone-in (a fresh
    # apex on a boundary facet) through exactly that rebuild.
    qa, qb, seed, node = seeded(seed_value=3, einstein_hilbert=True)
    ids_a, ids_b = seed.vertex_ids
    face = mapped_faces(qa, ids_a)[0]
    ok, reason = cob.SurgicalCone(node.spacetime()).bridge(list(face) + [ids_b[0]])
    assert ok, reason
    faces = [sorted(tuple(f) for f in MC.block_surface(b, node.spacetime()).faces) for b in node.inputs]
    assert faces == [mapped_faces(qa, ids_a), mapped_faces(qb, ids_b)]
    uncovered = sorted(tuple(f) for f in node.uncovered_input_faces())
    assert len(uncovered) == 50 - 1
    # This case is about what a COMMITTED cone preserves, not about whether
    # one is allowed: a node with a declared boundary refuses cone-ins by
    # default (#997), so the old behaviour is asked for by name here.
    node.set_boundary_may_extend(True)
    thresholds = MC.RefinementIndicators()
    thresholds.mesh_quality = 2.0  # a lower bound every mesh crosses: refine now
    node.set_refinement_thresholds(thresholds)
    assert node.refine_geometry(1) == 1
    st = node.spacetime()
    assert len(tops(st)) == 2 and len(vertex_ids(st)) == 26
    assert sorted(tuple(f) for f in node.uncovered_input_faces()) == uncovered
    assert set(uncovered) <= set(registered(st, 3)), "the uncovered torus faces are registered cells of the rebuilt host"
    assert [sorted(tuple(f) for f in MC.block_surface(b, st).faces) for b in node.inputs] == faces
