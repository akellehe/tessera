# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A surface block's own Laplacian (#961, T2 of ``docs/design/qubit_cobordism_spec.md``).

Each input torus of the collar keeps representing its input state through the
zero mode of its OWN Laplacian while its cells and lengths move (spec R3, R7,
D2). For a surface block (``seed_inputs`` by region) the block's own complex is
its surface — the 2-complex of its own triangles inside its vertex set with
the host's live lengths and phases (``MultiCobordism.block_surface_subcomplex``)
— and the block's fiber residual and its analytic gradient are read on that
complex in the ZERO MODE of its own pencil (the harmonic contour of the
block's own assembled pencil), never band 1 and never a contour copied from the
whole. Before this, the per-block read took the host's top cells inside the
vertex set, which a surface block of a 3-complex has none of, so a torus
scored the full leak 1.0 forever. Nothing is pinned: the residual is what
holds the state, weighted by ``set_input_residual_weight`` next to the bulk
terms of the objective. Ordinary blocks and nodes without surface inputs are
bit-identical to before (their read is the sub-complex at the fiber's stored
contour, band 1 by default).

The state fiber of each torus is its holomorphic form on its edges (degree 1,
``SimplicialQubit.holomorphic_form()`` in the torus's edge order carried
through the collar's id map, the harmonic contour set on the fiber). The
cotangent operator of the qubit read and the engine's Whitney pencil agree
exactly on flat tori, which is why the zero mode carries the form to rounding.

Observed (3x3 and 4x4 flat tori, tau_A = 0.3 + 1.1i, tau_B = -0.2 + 0.8i):

* (a) each block's own-Laplacian residual on the collar seed equals the
  residual of the same fiber read on the standalone torus, both at rounding
  (3x3: 2.199e-30 and 1.294e-30, the PencilLayer leak of the zero mode
  2.165e-30 and 1.244e-30; 4x4: 2.929e-30 and 2.247e-30, PencilLayer
  2.898e-30 and 2.472e-30): the zero mode of the flat torus carries its own
  holomorphic form exactly. Before this change both blocks scored 1.0. A
  fiber attached with NO contour reads bit-identically (the block's read
  chooses its own zero mode);
* (b) after an interior Pachner add (a 1->4 stellar subdivision of a collar
  cell) and after a cone-out dent of a torus-adjacent cell (two-layer collar:
  all 54 dents of a one-layer collar are refused by the gate), each block's
  surface is still its torus and its residual is bit-identical; the dented
  face is reported uncovered and the bridge phase reopens; a committed
  stage-1 pass keeps the tori and the floor;
* (c) rescaling every bulk edge (an endpoint outside both tori) leaves both
  residuals bit-identical; rescaling one edge of torus A by 1.3 moves A's
  residual to 3.5e-3 and leaves B's bit-identical;
* (d) on jittered lengths (residuals 9.1e-2 and 3.0e-2) the analytic gradient
  of each block residual (``fiber_mode_ascent``) is supported only on that
  block's own edges, the two blocks add, the Euler identity
  sum_e s_e dF_e = 0 holds to 3.6e-16 and 4.8e-17 (the leak is invariant
  under a common scale of the squared lengths), and a central difference on
  one torus edge gives -7.590525e-01 against the analytic -7.590524e-01 (the
  finite difference is a sanity check in this test only; the engine never
  uses one); at the flat point the gradient vanishes;
* (e) an ordinary node (degree-0 fibers on two tetrahedra of a grown simplex
  seed) reads exactly the sub-complex at the fiber's stored contour, band 1
  by default (0.847302004056 and 0.952165203558, two-body 0.937383793190,
  r_U 2.736851000804), and its ascent is the gradient of r_U; the same dump
  is bit-identical to the build before this change;
* (f) stage 2 on the collar (Regge term on, real locus, input weight 1e6, 20
  steps): the objective descends 123.1 -> 91.5, the bulk's edges move (up to
  3.6e-4, torus edges up to 8.6e-3), every edge stays spacelike, and both
  residuals hold (3.4e-9 and 3.1e-7); at weight 1 the Regge term wins
  (9.0e-3 and 1.4e-2 after 10 steps). Before this change stage 2 under the
  Regge term did not descend the residual at all (see
  ``ObjectiveContext.fiber_residuals``).
"""
import itertools
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
FLOOR = 1e-24  # the zero mode of a flat torus carries its holomorphic form to rounding (observed ~1e-30)


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


def state_fiber(q, ids, contour_on=None):
    """The torus's state as a fiber on its edges in host ids: degree 1, the
    holomorphic form in the torus's edge order carried through the id map,
    and the harmonic contour of ``contour_on``'s own pencil (the zero mode)
    when given."""
    f = cob.BoundaryFiber()
    f.degree = 1
    f.cells = [sorted((ids[int(i)], ids[int(j)])) for i, j in q.edges()]
    f.images = np.asarray(q.holomorphic_form()).reshape(-1, 1)
    if contour_on is not None:
        f.contour = cob.PencilLayer.harmonic_contour(cob.PencilLayer.assemble([contour_on]), 1)
    return f


def collar(n, seed_value=0, einstein_hilbert=False, attach=(0, 1), with_contour=True, layers=1,
           real_squared_lengths_only=False, weight=None):
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), layers)
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value,
              einstein_hilbert=einstein_hilbert, real_squared_lengths_only=real_squared_lengths_only)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    if weight is not None:
        node.set_input_residual_weight(weight)
    for i in attach:
        q, ids = (qa, qb)[i], seed.vertex_ids[i]
        f = state_fiber(q, ids, q.spacetime() if with_contour else None)
        node.attach_input_fiber(i, f, f.cells)
    return qa, qb, seed, node


def standalone_residual(q):
    """The residual of the torus's own state fiber read on the standalone
    torus: the whole-complex fiber residual of a node whose whole complex IS
    the torus (``SimplicialQubit.spacetime()`` with the torus's lengths)."""
    ident = {v: v for v in range(len(q.vertices()))}
    alone = MC(q.spacetime(), [], [], degrees=[1], einstein_hilbert=False)
    alone.set_whole_complex_fiber_target(state_fiber(q, ident, q.spacetime()))
    alone.use_fiber_residuals(True)
    return alone.whole_complex_fiber_residual()


def pencil_leak(st, fiber):
    """The same leak through ``PencilLayer`` alone: the least-squares residual
    of the fiber's images in the zero mode's images on its cells."""
    assembled = cob.PencilLayer.assemble([st])
    read = cob.PencilLayer.read_boundary_fiber(assembled, 1, fiber.contour, fiber.cells)
    Z, psi = np.asarray(read.images), fiber.images
    c = np.linalg.lstsq(Z, psi, rcond=None)[0]
    return float(np.linalg.norm(Z @ c - psi) ** 2 / np.linalg.norm(psi) ** 2), int(Z.shape[1])


def edge_geometry(st):
    out = {}
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        out[tuple(sorted((u, v)))] = (complex(e.getLength()), complex(e.getPhase()))
    return out


def edge_keys(st):
    return [tuple(sorted((e.getSource().getId(), e.getTarget().getId()))) for e in st.getEdgeList().toVector()]


def tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices())) for s in st.getTopSimplices())


def block_split(cell, regions):
    return tuple(sum(1 for v in cell if v in region) for region in regions)


def residuals(node):
    return [node.fiber_residual_for_input_block(i) for i in range(len(node.inputs))]


def surface_is_the_torus(node, st, q, ids, index):
    surface = MC.block_surface(node.inputs[index], st)
    assert sorted(tuple(f) for f in surface.faces) == mapped_faces(q, ids)
    assert sorted(tuple(e) for e in surface.edges) == mapped_edges(q, ids)
    own = MC.block_surface_subcomplex(node.inputs[index], st)
    assert tops(own) == mapped_faces(q, ids)
    host = edge_geometry(st)
    geometry = edge_geometry(own)
    assert sorted(geometry) == mapped_edges(q, ids)
    for edge, value in geometry.items():
        assert value == host[edge], "the surface carries the host's live length and phase verbatim"


def scale_edges(st, factor, keep):
    """Multiply the length of every edge ``keep`` selects by ``factor``."""
    for e in st.getEdgeList().toVector():
        if keep(tuple(sorted((e.getSource().getId(), e.getTarget().getId())))):
            e.setLength(complex(e.getLength()) * factor)


def jitter(st, rng, scale=0.25):
    """Generic real positive lengths so no torus sits on its flat point."""
    for e in st.getEdgeList().toVector():
        e.setLength(complex(e.getLength()) * (1.0 + scale * rng.uniform(-1, 1)))


def holomorphic(packed):
    """dF from the packed (2 Re dF, -2 Im dF)."""
    packed = np.asarray(packed)
    return 0.5 * (packed.real - 1j * packed.imag)


# --------------------------------------------------------------------------- #
# (a) the block's own-Laplacian residual is the standalone torus read
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_block_residual_equals_the_standalone_torus_read(n, whitney_default):
    qa, qb, seed, node = collar(n)
    st = node.spacetime()
    ids_a, ids_b = seed.vertex_ids
    on_collar = residuals(node)
    alone = [standalone_residual(qa), standalone_residual(qb)]
    leaks = [pencil_leak(qa.spacetime(), state_fiber(qa, {v: v for v in range(n * n)}, qa.spacetime())),
             pencil_leak(qb.spacetime(), state_fiber(qb, {v: v for v in range(n * n)}, qb.spacetime()))]
    for index, (q, ids) in enumerate(zip((qa, qb), (ids_a, ids_b))):
        assert node.inputs[index].surface
        surface_is_the_torus(node, st, q, ids, index)
        assert on_collar[index] < FLOOR, f"block {index}: {on_collar[index]:.3e}"
        assert alone[index] < FLOOR
        assert abs(on_collar[index] - alone[index]) < FLOOR
        assert leaks[index][1] == 2, "the torus's zero mode is its two harmonic forms"
        assert leaks[index][0] < FLOOR
    # r_U under fiber residuals is the sum of the two block terms (weight 1), nothing else
    assert node.r_u(st) == pytest.approx(sum(on_collar), abs=FLOOR)
    # the block's read chooses its own zero mode: a fiber attached with no
    # contour at all reads bit-identically (the band-1 default is never reached)
    _, _, _, bare = collar(n, with_contour=False)
    for index in range(2):
        assert not bare.inputs[index].fiber.contour.nodes
        assert bare.fiber_residual_for_input_block(index) == on_collar[index]
    print(f"\n[T2] {n}x{n}: block residuals on the collar {on_collar[0]:.3e} {on_collar[1]:.3e}; "
          f"standalone torus read {alone[0]:.3e} {alone[1]:.3e}; PencilLayer leak {leaks[0][0]:.3e} {leaks[1][0]:.3e}")


# --------------------------------------------------------------------------- #
# (b) stage-1 moves keep the surface and the residual
# --------------------------------------------------------------------------- #
def test_stage1_moves_keep_the_surface_and_the_residual(whitney_default):
    """A cone-out dent and an interior Pachner add leave each block's surface
    its torus and its residual bit-identical. On a ONE-layer collar no cell can
    be dented: every (3,1)/(1,3) cell's apex is a boundary vertex of the other
    torus (removing an interior triangle of its disk link punctures it) and
    every (2,2) cell sits mid-path in a torus edge's link, so the manifold gate
    refuses all 54 by name; the dent is exercised on a TWO-layer collar, whose
    torus-adjacent cells have an interior apex. A dent uncovers a torus face
    and the host's orphan prune drops that face's registration; the block
    carries its own faces, so its surface and `uncovered_input_faces` still
    say the face is the torus's (the bridge phase reopens on it)."""
    qa, qb, seed, node = collar(3)
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    before = residuals(node)
    st = node.spacetime()
    assert [sorted(tuple(f) for f in b.faces) for b in node.inputs] == [mapped_faces(qa, ids_a), mapped_faces(qb, ids_b)]
    # one layer: every dent is refused by name, the complex unchanged
    sc = cob.SurgicalCone(st)
    refusals = {}
    for cell in tops(st):
        ok, reason = sc.coneOut(list(cell))
        assert not ok, f"a one-layer collar cell {cell} was dented"
        assert reason
        refusals[block_split(cell, regions)] = refusals.get(block_split(cell, regions), 0) + 1
    assert refusals == {(3, 1): 18, (2, 2): 18, (1, 3): 18} and sc.depth == 0
    assert residuals(node) == before and node.bridge_phase_complete()
    # an interior Pachner add: the 1->4 stellar subdivision of a collar cell
    add = T.AddMove(st, 5, False, T.PachnerMode.PreGeometric, True)
    assert add.propose() and add.apply()
    assert len(tops(st)) == 54 + 3 and len(st.getVertexList().toVector()) == 19
    after_add = residuals(node)
    for index, (q, ids) in enumerate(zip((qa, qb), (ids_a, ids_b))):
        surface_is_the_torus(node, st, q, ids, index)
    assert after_add == before, "an interior add leaves the block's own Laplacian untouched"
    add.rollback()
    assert residuals(node) == before and len(tops(st)) == 54
    # two layers: a dent of a torus-adjacent cell (apex interior) is accepted
    qa, qb, seed, node = collar(3, layers=2)
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    before = residuals(node)
    st = node.spacetime()
    assert len(tops(st)) == 2 * 54 and node.bridge_phase_complete()
    sc = cob.SurgicalCone(st)
    dented = None
    for cell in tops(st):
        if block_split(cell, regions) not in ((3, 0), (0, 3)):
            continue
        ok, reason = sc.coneOut(list(cell))
        if ok:
            dented = cell
            break
        assert reason
    assert dented is not None, "no torus-adjacent cell of the two-layer collar could be dented"
    torus_face = tuple(sorted(v for v in dented if v in regions[0] | regions[1]))
    assert not node.bridge_phase_complete()
    assert [tuple(f) for f in node.uncovered_input_faces()] == [torus_face], "the dented face is uncovered, not gone"
    after_dent = residuals(node)
    for index, (q, ids) in enumerate(zip((qa, qb), (ids_a, ids_b))):
        surface_is_the_torus(node, st, q, ids, index)
    assert after_dent == before, "a dent leaves the block's own Laplacian untouched"
    assert sc.rollback()
    assert residuals(node) == before and node.bridge_phase_complete() and node.uncovered_input_faces() == []
    print(f"\n[T2] residuals before {before[0]:.3e} {before[1]:.3e}; after the interior add {after_add[0]:.3e} "
          f"{after_add[1]:.3e}; after the dent of {dented} (two-layer collar) {after_dent[0]:.3e} {after_dent[1]:.3e}; "
          f"one-layer dents refused by split: {refusals}")


def test_committed_stage1_pass_keeps_the_tori(whitney_default):
    """A stage-1 pass driven by the engine (Regge term on, so moves are scored
    by the bulk's stationarity as well): whatever it commits, the rebuilt
    complex keeps each torus as its block's surface and the residuals at the
    floor."""
    qa, qb, seed, node = collar(3, seed_value=2, einstein_hilbert=True)
    ids_a, ids_b = seed.vertex_ids
    before = residuals(node)
    cells_before = len(tops(node.spacetime()))
    node.run_stage1(max_steps=3, n_candidate_moves=8)
    st = node.spacetime()
    after = residuals(node)
    for index, (q, ids) in enumerate(zip((qa, qb), (ids_a, ids_b))):
        surface_is_the_torus(node, st, q, ids, index)
        assert after[index] < FLOOR
    print(f"\n[T2] stage 1 on the collar: cells {cells_before} -> {len(tops(st))}, residuals "
          f"{before[0]:.3e} {before[1]:.3e} -> {after[0]:.3e} {after[1]:.3e}")


# --------------------------------------------------------------------------- #
# (c) only the block's own edges enter its residual
# --------------------------------------------------------------------------- #
def test_bulk_edges_do_not_enter_the_block_residual(whitney_default):
    qa, qb, seed, node = collar(3)
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    st = node.spacetime()
    before = residuals(node)
    rng = np.random.default_rng(0)
    bulk = lambda edge: not any(edge[0] in r and edge[1] in r for r in regions)
    for e in st.getEdgeList().toVector():
        edge = tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
        if bulk(edge):
            e.setLength(complex(e.getLength()) * (1.0 + 0.5 * rng.uniform(-1, 1)))
    assert sum(1 for edge in edge_keys(st) if bulk(edge)) == 9 + 27  # the collar: vertical + diagonal cross edges
    assert residuals(node) == before, "bulk edges are not in the block's own Laplacian"
    # one edge of torus A: only A's residual moves
    a_edge = mapped_edges(qa, ids_a)[4]
    scale_edges(st, 1.3, lambda edge: edge == a_edge)
    moved = residuals(node)
    assert moved[0] > 1e-6, f"torus A's residual did not move: {moved[0]:.3e}"
    assert moved[1] == before[1]
    print(f"\n[T2] bulk rescaled: residuals unchanged at {before[0]:.3e} {before[1]:.3e}; one A edge x1.3: "
          f"{moved[0]:.3e} {moved[1]:.3e}")


# --------------------------------------------------------------------------- #
# (d) the analytic gradient: support, Euler identity, finite-difference sign
# --------------------------------------------------------------------------- #
def test_gradient_support_euler_identity_and_sign(whitney_default):
    def jittered(attach):
        qa, qb, seed, node = collar(3, attach=attach)
        jitter(node.spacetime(), np.random.default_rng(7))
        return qa, qb, seed, node

    qa, qb, seed, both = jittered((0, 1))
    _, _, _, only_a = jittered((0,))
    _, _, _, only_b = jittered((1,))
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    keys = edge_keys(both.spacetime())
    assert keys == edge_keys(only_a.spacetime()) == edge_keys(only_b.spacetime())
    inside = [np.array([edge[0] in r and edge[1] in r for edge in keys]) for r in regions]
    bulk = ~(inside[0] | inside[1])
    s = np.array([complex(e.getLength()) ** 2 for e in both.spacetime().getEdgeList().toVector()])
    off_floor = residuals(both)
    assert all(r > 1e-4 for r in off_floor), off_floor
    total_l, total_p = (np.asarray(g) for g in both.fiber_mode_ascent())
    a_l, a_p = (np.asarray(g) for g in only_a.fiber_mode_ascent())
    b_l, _ = (np.asarray(g) for g in only_b.fiber_mode_ascent())
    # support: each block's gradient lives on its own edges only; the total is the sum
    assert np.all(total_l[bulk] == 0) and np.all(a_l[~inside[0]] == 0) and np.all(b_l[~inside[1]] == 0)
    assert np.abs(a_l[inside[0]]).max() > 0 and np.abs(b_l[inside[1]]).max() > 0
    np.testing.assert_allclose(total_l, a_l + b_l, rtol=1e-12, atol=1e-16)
    assert np.all(total_p == 0) and np.all(a_p == 0), "degree 1 carries no phase gradient"
    # Euler identity per block: sum_e s_e dF_e = 0 (scale invariance of the leak)
    euler = []
    for g in (a_l, b_l):
        euler.append(abs(np.sum(s * holomorphic(g))) / (np.abs(g).max() * np.abs(s).max()))
        assert euler[-1] < 1e-10, f"Euler identity violated: {euler[-1]:.3e}"
    # a central difference on the A edge with the largest sensitivity (sanity only)
    st = only_a.spacetime()
    edges = st.getEdgeList().toVector()
    i = int(np.argmax(np.abs(a_l) * inside[0]))
    e, h = edges[i], 1e-6
    s0 = complex(e.getLength()) ** 2
    e.setLength(np.sqrt(s0 + h))
    plus = only_a.fiber_residual_for_input_block(0)
    e.setLength(np.sqrt(s0 - h))
    minus = only_a.fiber_residual_for_input_block(0)
    e.setLength(np.sqrt(s0))
    fd = (plus - minus) / (2 * h)
    analytic = a_l[i].real  # d/dRe s of the packed gradient
    assert np.sign(fd) == np.sign(analytic) and fd != 0
    assert abs(fd - analytic) < 1e-5 * abs(analytic)
    print(f"\n[T2] jittered residuals {off_floor[0]:.3e} {off_floor[1]:.3e}; Euler defect {euler[0]:.2e} {euler[1]:.2e}; "
          f"edge {keys[i]}: analytic {analytic:.6e} central difference {fd:.6e}")
    # at the flat point the residual is at its floor and so is the gradient (a minimum)
    _, _, _, flat = collar(3)
    flat_l, _ = (np.asarray(g) for g in flat.fiber_mode_ascent())
    assert np.abs(flat_l).max() < 1e-12, np.abs(flat_l).max()


# --------------------------------------------------------------------------- #
# (e) an ordinary node reads exactly as before
# --------------------------------------------------------------------------- #
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])


def flip_flop(psi, phi):
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return np.outer(D @ psi, D.T @ phi) + np.outer(D.T @ psi, D @ phi)


def degree0_fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = [[0], [1], [2], [3]]
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def ordinary_read(st, block):
    """The ordinary block read replicated through the public surface: the
    host's top cells inside the block's vertex set with the host's geometry,
    the fiber's stored contour or band 1 above the zero mode, the
    least-squares leak on the fiber's cells."""
    region = set(int(v) for v in block.vertices)
    cells = [[int(v) for v in t] for t in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(3)
             if set(int(v) for v in t) <= region]
    sub = T.Spacetime.fromCells(3, cells, 1.0, 0j)
    host = edge_geometry(st)
    for e in sub.getEdgeList().toVector():
        length, phase = host[tuple(sorted((e.getSource().getId(), e.getTarget().getId())))]
        e.setLength(length)
        e.setPhase(phase)
    assembled = cob.PencilLayer.assemble([sub])
    fiber = block.fiber
    contour = fiber.contour if fiber.contour.nodes else cob.PencilLayer.band_contour(assembled, 0, 1)
    read = cob.PencilLayer.read_boundary_fiber(assembled, 0, contour, fiber.cells)
    Z, psi = np.asarray(read.images), np.asarray(fiber.images)
    c = np.linalg.lstsq(Z, psi, rcond=None)[0]
    return float(np.linalg.norm(Z @ c - psi) ** 2 / np.linalg.norm(psi) ** 2)


def test_ordinary_nodes_read_as_before(whitney_default):
    rng = np.random.default_rng(11)
    psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=0, precone=8, einstein_hilbert=False)
    for e in node.spacetime().getEdgeList().toVector():
        s = 1.0 + 0.15 * rng.uniform(-1, 1) + 1j * 0.15 * rng.uniform(-1, 1)
        e.setLength(np.sqrt(complex(s)))
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(node.spacetime()).kSimplexVertices(3)]
    a, b = next((x, y) for x, y in itertools.combinations(tets, 2) if not set(x) & set(y))
    node.seed_inputs([0, 1])
    node.attach_input_fiber(0, degree0_fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, degree0_fiber(phi), [[v] for v in b])
    node.set_two_body_target(flip_flop(psi, phi))
    node.use_fiber_residuals(True)
    assert not node.has_surface_inputs() and not any(block.surface for block in node.inputs)
    st = node.spacetime()
    block_residuals = residuals(node)
    for index in range(2):
        assert block_residuals[index] == pytest.approx(ordinary_read(st, node.inputs[index]), rel=1e-12, abs=1e-15)
        assert block_residuals[index] > 1e-3, "an ordinary block off its band is not at any floor"
    assert node.r_u(st) == pytest.approx(sum(block_residuals) + node.two_body_residual(), rel=1e-12)
    # the ascent is the exact gradient of r_U (a central difference at its own accuracy)
    total_l, _ = (np.asarray(g) for g in node.fiber_mode_ascent())
    edges = st.getEdgeList().toVector()
    fd = np.zeros(len(edges), dtype=complex)
    for i, e in enumerate(edges):
        l0 = complex(e.getLength())
        s0 = l0 * l0
        parts = []
        for step in (1e-6, 1e-6j):
            e.setLength(np.sqrt(s0 + step))
            plus = node.r_u(st)
            e.setLength(np.sqrt(s0 - step))
            minus = node.r_u(st)
            parts.append((plus - minus) / 2e-6)
        e.setLength(l0)
        fd[i] = complex(parts[0], parts[1])
    assert np.abs(fd - total_l).max() < 1e-5 * max(1.0, np.abs(total_l).max())
    print(f"\n[T2] ordinary node: block residuals {block_residuals[0]:.12e} {block_residuals[1]:.12e}, "
          f"two-body {node.two_body_residual():.12e}, r_U {node.r_u(st):.12e}")


# --------------------------------------------------------------------------- #
# (f) stage 2: the residual is descended next to the bulk term
# --------------------------------------------------------------------------- #
def spacelike_real(st):
    s = np.array([complex(e.getLength()) ** 2 for e in st.getEdgeList().toVector()])
    return bool(np.all(s.real > 0) and np.all(np.abs(s.imag) < 1e-15))


# Marked slow: a stage-2 bulk relaxation with the residuals re-read afterwards;
# about forty-five seconds.
@pytest.mark.slow
def test_stage2_moves_the_bulk_and_holds_the_residuals(whitney_default):
    """Stage 2 on the collar with both fibers attached and the Regge term on
    (the tori are real and spacelike, so the node runs on the real locus):
    the bulk's edges move, the objective descends, and each block residual is
    descended next to the Regge term through the analytic fiber ascent. The
    level a residual holds is the balance of ``set_input_residual_weight``
    against the Regge pull on the tori's own edges (about 1/weight^2): at
    weight 1e6 both residuals hold between 1e-9 and 1e-7 over the first 40
    steps (observed 3e-9 and 3e-7 after 20 steps; asserted below 1e-5) while
    the bulk moves from the first steps on; at weight 1 the Regge term wins and
    the residuals are 1e-2 after 10 steps, which is the control here."""
    qa, qb, seed, node = collar(3, einstein_hilbert=True, real_squared_lengths_only=True, weight=1e6)
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    st = node.spacetime()
    before = residuals(node)
    terms_before = node.objective_terms()
    geometry_before = edge_geometry(st)
    objective_before = node.objective()
    assert spacelike_real(st)
    trace = node.run_stage2(beta=1.0, max_iters=20, tolerance=1e-15)
    st = node.spacetime()
    after = residuals(node)
    terms_after = node.objective_terms()
    geometry_after = edge_geometry(st)
    moved = {edge: abs(geometry_after[edge][0] - geometry_before[edge][0]) for edge in geometry_before}
    bulk_moved = max(v for edge, v in moved.items() if not any(edge[0] in r and edge[1] in r for r in regions))
    torus_moved = max(v for edge, v in moved.items() if any(edge[0] in r and edge[1] in r for r in regions))
    assert len(trace) > 10, f"stage 2 stalled after {len(trace) - 1} steps"
    assert trace[-1] < objective_before and terms_after.regge_stationarity < terms_before.regge_stationarity
    assert bulk_moved > 1e-4, "stage 2 must move the bulk's edges"
    assert spacelike_real(st), "the collar stays real and spacelike at this weight"
    for index, (q, ids) in enumerate(zip((qa, qb), (ids_a, ids_b))):
        surface_is_the_torus(node, st, q, ids, index)
        assert after[index] < 1e-5, f"block {index} left its level: {after[index]:.3e}"
    # the control: at weight 1 the Regge term wins and the residuals leave the floor
    _, _, _, control = collar(3, einstein_hilbert=True, real_squared_lengths_only=True, weight=1.0)
    control.run_stage2(beta=1.0, max_iters=10, tolerance=1e-15)
    weak = residuals(control)
    assert min(weak) > 1e-4 > max(after)
    print(f"\n[T2] stage 2 at weight 1e6 ({len(trace) - 1} accepted steps): objective {objective_before:.6e} -> {trace[-1]:.6e}, "
          f"Regge stationarity {terms_before.regge_stationarity:.4e} -> {terms_after.regge_stationarity:.4e}; "
          f"bulk edges moved by up to {bulk_moved:.3e}, torus edges by up to {torus_moved:.3e}; "
          f"residuals {before[0]:.3e} {before[1]:.3e} -> {after[0]:.3e} {after[1]:.3e}; "
          f"at weight 1 after 10 steps {weak[0]:.3e} {weak[1]:.3e}")
