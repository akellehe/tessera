# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Committed stage-1 moves keep every surviving edge's link phase (#977).

A committed stage-1 move rebuilds the complex from a snapshot
(``MultiCobordism::snapshotOf`` / ``build``). The snapshot recorded lengths
only, so the first committed move reset every phase to zero. The tori of the
qubit cobordism carry pure-gauge link phases (``docs/design/qubit_cobordism_spec.md``
S1) and under the Whitney pencil the operator depends on them at every degree,
so that rebuild changed the physics of the state a torus carries: its input
fiber is a zero mode of the TWISTED Laplacian, not of the untwisted one, and
the block's residual jumps from its floor to order one when the phases go.

Coverage, on a collar node whose two tori carry pure-gauge phases
phi_e = g(target) - g(source) and whose bulk carries phases too:

(a) the snapshot/rebuild round trip through a committed cone-in
    (``refine_geometry``, the one rebuild every committed move goes through) is
    bit-exact for every surviving edge's length and phase; the apex's new
    edges take the auto-wired length and a zero phase;
(b) a committed interior Pachner add and a committed cone-out dent (two-layer
    collar), driven through stage 1 by an injected objective, keep every
    surviving edge's length and phase bit-exactly and give the new edges a
    zero phase;
(c) the qubit read of each torus (``SimplicialQubit(surface, cycle_A,
    cycle_B)``, pure gauge) and each block's residual are unchanged to
    rounding by those committed moves;
(d) an ordinary node (degree-0 tetrahedral seed, zero phases) reproduces the
    saved full-precision dump of origin/main's build (``data/
    stage1_keeps_phases_ordinary_node_dump.json``): cells, lengths and phases
    bit for bit, the objective and the residuals to rounding.
"""
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
DUMP = pathlib.Path(__file__).with_name("data") / "stage1_keeps_phases_ordinary_node_dump.json"


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
def quiet(build):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def torus(tau, n):
    return quiet(lambda: obs.SimplicialQubit.flat_torus(tau, n, n))


def gauge_function(n_vertices, seed):
    rng = np.random.default_rng(seed)
    return [complex(x) for x in rng.uniform(-math.pi, math.pi, size=n_vertices)]


def gauged_torus(q, g):
    """The torus read through its own Spacetime with the pure gauge
    phi_e = g(target) - g(source) written on every edge's stored orientation
    (the connection U_xy = g_x^{-1} g_y, g_x = exp(i g(x)))."""
    st = q.spacetime()
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        edge.setPhase(g[v] - g[u])
    return st, quiet(lambda: obs.SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B())))


def state_fiber(gauged, gauged_st, ids):
    """The gauged torus's holomorphic form on its edges as a degree-1 fiber on
    the host's edge ids, with the harmonic contour of the torus's own pencil."""
    f = cob.BoundaryFiber()
    f.degree = 1
    f.cells = [sorted((ids[int(i)], ids[int(j)])) for i, j in gauged.edges()]
    f.images = np.asarray(gauged.holomorphic_form(), dtype=complex).reshape(-1, 1)
    f.contour = cob.PencilLayer.harmonic_contour(cob.PencilLayer.assemble([gauged_st]), 1)
    return f


def mapped_faces(q, ids):
    return sorted(tuple(sorted(ids[int(v)] for v in f)) for f in q.faces())


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


def canonical(source, target, phase):
    """The phase on the canonical min->max orientation of the pair: an edge
    stored max->min carries the inverse link, so its phase is negated (as
    0 - phi, so a zero stays +0 and hex dumps compare bit for bit)."""
    return phase if source < target else 0j - phase


def edge_geometry(st):
    """{(min, max): (length, canonical phase)} over every edge of ``st``."""
    out = {}
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        out[(min(u, v), max(u, v))] = (complex(e.getLength()), canonical(u, v, complex(e.getPhase())))
    return out


def tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices())) for s in st.getTopSimplices())


def vertex_ids(st):
    return sorted(v.getId() for v in st.getVertexList().toVector())


def block_split(cell, regions):
    return tuple(sum(1 for v in cell if v in region) for region in regions)


def residuals(node):
    return [node.fiber_residual_for_input_block(i) for i in range(len(node.inputs))]


def surface_cycles(marking, surface):
    """A host marking as (edge index, sign) steps in ``surface``'s edge order:
    the ``SimplicialQubit`` Spacetime constructor indexes edges in ascending
    (i, j) order of the vertex ids."""
    edges = sorted(tuple(sorted((e.getSource().getId(), e.getTarget().getId())))
                   for e in surface.getEdgeList().toVector())
    index = {edge: n for n, edge in enumerate(edges)}
    return [[(index[(min(u, v), max(u, v))], 1 if u < v else -1) for u, v in steps] for steps in marking]


def qubit_read(node, index, st, marking):
    """The qubit read of block ``index``'s live surface: its own faces with the
    host's live lengths and phases (spec S6)."""
    own = MC.block_surface_subcomplex(node.inputs[index], st)
    cycles = surface_cycles(marking, own)
    return quiet(lambda: obs.SimplicialQubit(own, cycles[0], cycles[1]))


def leak_without_phases(node, index, st):
    """What the block's residual would be if the rebuild dropped the phases:
    the block's fiber (the TWISTED zero mode) read on its own surface with
    every phase zeroed."""
    own = MC.block_surface_subcomplex(node.inputs[index], st)
    for e in own.getEdgeList().toVector():
        e.setPhase(0j)
    fiber = node.inputs[index].fiber
    assembled = cob.PencilLayer.assemble([own])
    contour = cob.PencilLayer.harmonic_contour(assembled, 1)
    read = cob.PencilLayer.read_boundary_fiber(assembled, 1, contour, fiber.cells)
    Z, psi = np.asarray(read.images), np.asarray(fiber.images)
    c = np.linalg.lstsq(Z, psi, rcond=None)[0]
    return float(np.linalg.norm(Z @ c - psi) ** 2 / np.linalg.norm(psi) ** 2)


BULK_PHASES = [complex(0.37, -0.11), complex(-1.9, 0.05), complex(2.4, 0.0), complex(0.0, 0.3), complex(-0.8, -0.2)]


def collar_with_phases(n, layers=1, seed_value=0, gauge_seeds=(11, 12)):
    """The collar node (spec S3) of two flat tori carrying the pure gauges
    ``gauge_seeds`` on the host, with a nonzero phase on every sixth bulk edge
    as well, each torus's gauged holomorphic form attached as its state fiber."""
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), layers)
    st = seed.host
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    gauges, reads = [], []
    for q, ids, gauge_seed in zip((qa, qb), (ids_a, ids_b), gauge_seeds):
        g = gauge_function(len(q.vertices()), gauge_seed)
        gauges.append({ids[i]: g[i] for i in range(len(g))})
        reads.append(gauged_torus(q, g))
    # phi_e = g(target) - g(source) on the host's stored orientation, per torus
    bulk_index = 0
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        block = next((k for k, region in enumerate(regions) if u in region and v in region), None)
        if block is not None:
            edge.setPhase(gauges[block][v] - gauges[block][u])
        else:
            if bulk_index % 6 == 0:
                edge.setPhase(BULK_PHASES[(bulk_index // 6) % len(BULK_PHASES)])
            bulk_index += 1
    node = MC(st, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value, einstein_hilbert=False)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    for i, ((gauged_st, gauged), ids) in enumerate(zip(reads, (ids_a, ids_b))):
        f = state_fiber(gauged, gauged_st, ids)
        node.attach_input_fiber(i, f, f.cells)
    markings = [host_marking(qa, ids_a), host_marking(qb, ids_b)]
    return (qa, qb), seed, node, [read for _, read in reads], markings


def assert_surviving_edges_kept(before, after, new_edges_expected):
    """Every edge of ``before`` is in ``after`` with the identical (length,
    phase); the edges only ``after`` holds are the move's, with a zero phase."""
    missing = set(before) - set(after)
    assert not missing, f"edges dropped by the rebuild: {sorted(missing)[:5]}"
    changed = {edge: (before[edge], after[edge]) for edge in before if after[edge] != before[edge]}
    assert not changed, f"edges whose geometry changed through the rebuild: {list(changed.items())[:5]}"
    new = {edge: after[edge] for edge in set(after) - set(before)}
    assert len(new) == new_edges_expected, sorted(new)
    for edge, (length, phase) in new.items():
        assert phase == 0j, f"a new edge {edge} carries a phase {phase}"
        assert length.imag == 0.0 and length.real > 0.0, f"a new edge {edge} is not auto-wired spacelike: {length}"
    return new


def check_reads(node, st, standalone, markings, residuals_before, label):
    """(c): each torus's qubit read equals its standalone gauged read and the
    input, and each block's residual is unchanged."""
    taus = []
    for index, (q, marking) in enumerate(zip(standalone, markings)):
        read = qubit_read(node, index, st, marking)
        assert not read.trivial_connection(), "the surface read must see the pure gauge"
        taus.append(read.tau())
        assert abs(read.tau() - q.tau()) < 1e-12, (label, index, read.tau(), q.tau())
    after = residuals(node)
    assert after == residuals_before, (label, after, residuals_before)
    return taus, after


# --------------------------------------------------------------------------- #
# the injected objectives that drive stage 1 to a chosen move kind
# --------------------------------------------------------------------------- #
class CellCountObjective(cob.CobordismObjective):
    """F = sign * (number of top cells): with sign -1 the 1->4 add (+3 cells)
    is the best-scoring move stage 1 can draw; a cone-in or a 2->3 flip (+1)
    only wins if no add is drawn."""

    def __init__(self, sign):
        super().__init__()
        self.sign = sign

    def name(self):
        return "cell_count"

    def term_names(self):
        return [cob.ObjectiveTermName.REGGE_STATIONARITY]

    def terms(self, context):
        out = MC.ObjectiveTerms()
        out.regge_stationarity = self.sign * float(len(context.spacetime.getTopSimplices()))
        return out

    def direction(self, context):
        out = cob.ObjectiveDirection()
        out.ascent = np.zeros(context.edge_count, dtype=complex)
        out.baseline = self.sign * float(len(context.spacetime.getTopSimplices()))
        out.baseline_computed = True
        return out

    def is_target_conditioned(self):
        return False


class UncoveredTorusFacesObjective(cob.CobordismObjective):
    """F = -(number of torus faces no top cell covers): only a cone-out dent
    of a torus-adjacent cell lowers it."""

    def __init__(self, torus_faces):
        super().__init__()
        self.torus_faces = set(torus_faces)

    def uncovered(self, st):
        covered = {tuple(sorted(set(cell) - {v})) for cell in tops(st) for v in cell}
        return sorted(self.torus_faces - covered)

    def name(self):
        return "uncovered_torus_faces"

    def term_names(self):
        return [cob.ObjectiveTermName.REGGE_STATIONARITY]

    def terms(self, context):
        out = MC.ObjectiveTerms()
        out.regge_stationarity = -float(len(self.uncovered(context.spacetime)))
        return out

    def direction(self, context):
        out = cob.ObjectiveDirection()
        out.ascent = np.zeros(context.edge_count, dtype=complex)
        out.baseline = -float(len(self.uncovered(context.spacetime)))
        out.baseline_computed = True
        return out

    def is_target_conditioned(self):
        return False


# --------------------------------------------------------------------------- #
# (a) the round trip through a committed cone-in
# --------------------------------------------------------------------------- #
def test_committed_cone_in_keeps_every_surviving_edge_bit_exactly(whitney_default):
    (qa, qb), seed, node, standalone, markings = collar_with_phases(3, seed_value=3)
    st0 = node.spacetime()
    before = edge_geometry(st0)
    regions = [set(ids.values()) for ids in seed.vertex_ids]
    torus_edges = [e for e in before if any(e[0] in r and e[1] in r for r in regions)]
    bulk_edges = [e for e in before if e not in torus_edges]
    assert all(before[e][1] != 0j for e in torus_edges), "every torus edge carries its gauge phase"
    assert sum(before[e][1] != 0j for e in bulk_edges) >= 5, "some bulk edges carry a phase"
    residuals_before = residuals(node)
    taus_before, _ = check_reads(node, st0, standalone, markings, residuals_before, "seed")
    for index, q in enumerate(standalone):
        assert abs(q.tau() - (TAU_A, TAU_B)[index]) < 1e-9
    dropped = [leak_without_phases(node, index, st0) for index in range(2)]
    assert all(leak > 1e-3 for leak in dropped), dropped
    # the committed cone-in: a fresh apex on a boundary (torus) face, through the rebuild
    # This case is about what a COMMITTED cone preserves, not about whether
    # one is allowed: a node with a declared boundary refuses cone-ins by
    # default (#997), so the old behaviour is asked for by name here.
    node.set_boundary_may_extend(True)
    thresholds = MC.RefinementIndicators()
    thresholds.mesh_quality = 2.0  # a lower bound every mesh crosses: refine now
    node.set_refinement_thresholds(thresholds)
    assert node.refine_geometry(1) == 1
    st = node.spacetime()
    assert st is not st0 and len(tops(st)) == len(tops(st0)) + 1 and len(vertex_ids(st)) == len(vertex_ids(st0)) + 1
    new = assert_surviving_edges_kept(before, edge_geometry(st), 3)
    apex = max(vertex_ids(st))
    assert all(apex in edge for edge in new)
    taus_after, residuals_after = check_reads(node, st, standalone, markings, residuals_before, "cone-in")
    print(f"\n[#977] cone-in on the 3x3 collar with phases: {len(before)} edges round-tripped bit-exactly, "
          f"{len(new)} apex edges {sorted(new.values())[0]}; residuals {residuals_before[0]:.3e} "
          f"{residuals_before[1]:.3e} -> {residuals_after[0]:.3e} {residuals_after[1]:.3e}; "
          f"tau {taus_before[0]:.12f} {taus_before[1]:.12f} -> {taus_after[0]:.12f} {taus_after[1]:.12f}; "
          f"leak had the phases been dropped {dropped[0]:.3e} {dropped[1]:.3e}")


# --------------------------------------------------------------------------- #
# (b), (c) a committed interior add and a committed dent through stage 1
# --------------------------------------------------------------------------- #
def test_committed_interior_add_keeps_the_phases(whitney_default):
    (qa, qb), seed, node, standalone, markings = collar_with_phases(3, seed_value=5)
    st0 = node.spacetime()
    before = edge_geometry(st0)
    residuals_before = residuals(node)
    objective = CellCountObjective(-1)
    node.set_objective(objective)
    passes = 0
    while len(tops(node.spacetime())) == 54 and passes < 4:
        node.run_stage1(max_steps=1, n_candidate_moves=48)
        passes += 1
    st = node.spacetime()
    assert len(tops(st)) == 54 + 3 and len(vertex_ids(st)) == 19, (len(tops(st)), len(vertex_ids(st)))
    assert st is not st0
    new = assert_surviving_edges_kept(before, edge_geometry(st), 4)
    added = max(vertex_ids(st))
    assert all(added in edge for edge in new), "the new edges are the added vertex's"
    assert node.bridge_phase_complete() and node.uncovered_input_faces() == []
    taus, residuals_after = check_reads(node, st, standalone, markings, residuals_before, "add")
    print(f"\n[#977] interior 1->4 add on the 3x3 collar with phases (committed after {passes} stage-1 pass(es)): "
          f"{len(before)} edges bit-exact, new edges {sorted(new.values())}; residuals "
          f"{residuals_before[0]:.3e} {residuals_before[1]:.3e} -> {residuals_after[0]:.3e} {residuals_after[1]:.3e}; "
          f"tau {taus[0]:.12f} {taus[1]:.12f}")


def test_committed_cone_out_dent_keeps_the_phases(whitney_default):
    """On a one-layer collar no cell can be dented (every apex is a boundary
    vertex of the other torus); the dent is exercised on a two-layer collar,
    whose torus-adjacent cells have an interior apex, as the bridge and
    block-surface tests do."""
    (qa, qb), seed, node, standalone, markings = collar_with_phases(3, layers=2, seed_value=7)
    ids_a, ids_b = seed.vertex_ids
    regions = [set(ids_a.values()), set(ids_b.values())]
    st0 = node.spacetime()
    assert len(tops(st0)) == 108 and node.bridge_phase_complete()
    before = edge_geometry(st0)
    residuals_before = residuals(node)
    objective = UncoveredTorusFacesObjective(mapped_faces(qa, ids_a) + mapped_faces(qb, ids_b))
    node.set_objective(objective)
    passes = 0
    # A dent is a cone-out, which exposes the removed cell's facets, so a node
    # with a declared boundary refuses it by default (#997). This case is
    # about what a committed dent preserves, so it asks for the old behaviour.
    node.set_boundary_may_extend(True)
    while len(tops(node.spacetime())) == 108 and passes < 6:
        node.run_stage1(max_steps=1, n_candidate_moves=48)
        passes += 1
    st = node.spacetime()
    assert len(tops(st)) == 107, "no torus-adjacent dent was committed"
    assert st is not st0
    dented = (set(tops(st0)) - set(tops(st))).pop()
    assert block_split(dented, regions) in ((3, 0), (0, 3))
    torus_face = tuple(sorted(v for v in dented if v in regions[0] | regions[1]))
    assert [tuple(f) for f in node.uncovered_input_faces()] == [torus_face]
    assert not node.bridge_phase_complete()
    assert vertex_ids(st) == vertex_ids(st0)
    assert_surviving_edges_kept(before, edge_geometry(st), 0)
    taus, residuals_after = check_reads(node, st, standalone, markings, residuals_before, "dent")
    print(f"\n[#977] cone-out dent of {dented} on the two-layer collar with phases (committed after {passes} "
          f"stage-1 pass(es)): {len(before)} edges bit-exact, no new edge; residuals {residuals_before[0]:.3e} "
          f"{residuals_before[1]:.3e} -> {residuals_after[0]:.3e} {residuals_after[1]:.3e}; "
          f"tau {taus[0]:.12f} {taus[1]:.12f}")


# --------------------------------------------------------------------------- #
# (d) an ordinary node is bit-identical to origin/main's build
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


def hexc(z):
    z = complex(z)
    return [float(z.real).hex(), float(z.imag).hex()]


def unhex(pair):
    return complex(float.fromhex(pair[0]), float.fromhex(pair[1]))


def ordinary_node_dump():
    """The degree-0 tetrahedral seed of the two-body tests (zero phases,
    jittered complex lengths, two degree-0 fibers and the flip-flop target),
    rebuilt through ``build`` by the constructor's eight precone cone-ins and
    once more by a committed refinement cone-in, dumped at full precision."""
    import itertools
    rng = np.random.default_rng(11)
    psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=0, precone=8, einstein_hilbert=True)
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
    cells_before = len(tops(node.spacetime()))
    # This case is about what a COMMITTED cone preserves, not about whether
    # one is allowed: a node with a declared boundary refuses cone-ins by
    # default (#997), so the old behaviour is asked for by name here.
    node.set_boundary_may_extend(True)
    thresholds = MC.RefinementIndicators()
    thresholds.mesh_quality = 2.0
    node.set_refinement_thresholds(thresholds)
    committed = node.refine_geometry(1)
    st = node.spacetime()
    edges = edge_geometry(st)
    return {
        "precone_cells": cells_before,
        "refined_cells": committed,
        "cells": [list(cell) for cell in tops(st)],
        "edges": [[edge[0], edge[1], hexc(length), hexc(phase)] for edge, (length, phase) in sorted(edges.items())],
        "objective": float(node.objective()).hex(),
        "r_u": float(node.r_u(st)).hex(),
        "block_residuals": [float(r).hex() for r in residuals(node)],
        "two_body_residual": float(node.two_body_residual()).hex(),
    }


def within_ulps(got, expected, ulps):
    return abs(got - expected) <= ulps * math.ulp(max(abs(expected), np.finfo(float).tiny))


# The saved dump was generated from origin/main's build (8ef13c1) by
#   python tests/cobordism/test_stage1_keeps_phases_python.py <path>
# Every phase of this node is zero, so the record it went through is the one
# that changed here and nothing else: the cells and every edge's length and
# phase must agree bit for bit. The objective and the residuals are compared
# at eight units in the last place because they vary at that level with the
# OpenMP thread count already on origin/main (measured: 1 and 4 threads agree,
# 8 threads differs in the last digit), not with the build.
def test_ordinary_node_is_bit_identical_to_the_saved_dump(whitney_default):
    expected = json.loads(DUMP.read_text())
    got = ordinary_node_dump()
    assert got["precone_cells"] == expected["precone_cells"] and got["refined_cells"] == expected["refined_cells"] == 1
    assert got["cells"] == expected["cells"]
    assert got["edges"] == expected["edges"], "an edge's length or phase differs bit for bit"
    assert all(unhex(edge[3]) == 0j for edge in got["edges"])
    for key in ("objective", "r_u", "two_body_residual"):
        assert within_ulps(float.fromhex(got[key]), float.fromhex(expected[key]), 8), key
    for mine, saved in zip(got["block_residuals"], expected["block_residuals"]):
        assert within_ulps(float.fromhex(mine), float.fromhex(saved), 8)
    print(f"\n[#977] ordinary node: {len(got['cells'])} cells, {len(got['edges'])} edges bit-identical to the saved dump; "
          f"objective {float.fromhex(got['objective'])!r} (saved {float.fromhex(expected['objective'])!r}), "
          f"r_U {float.fromhex(got['r_u'])!r}, blocks {[float.fromhex(r) for r in got['block_residuals']]}, "
          f"two-body {float.fromhex(got['two_body_residual'])!r}")


if __name__ == "__main__":
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(ordinary_node_dump(), sort_keys=True))
    print("wrote", sys.argv[1])
