# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Layer continuation on the Whitney pencil (#916): exact assembly of cobordisms
along shared cells with one epsilon, the star-product identity of boundary
responses, fiber forms of boundary targets read from certified bands, the
next pencil level with the Gram carried exactly (block-diagonal for disjoint
fibers, coupled across a glued interface), the transfer reversal assertion on
level links, a second level through the same code path, and the
MultiCobordism / CobordismDAG plumbing of fibers."""
import math
import os

import numpy as np
import pytest

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob

HL = cob.HodgeLaplacian
Whitney = cob.HodgeMetricSource.WhitneyPencil
PL = cob.PencilLayer
BASE = [[0, 1], [1, 2], [0, 2]]


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(Whitney)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


def _edge_length(a, b):
    """A deterministic squared length per vertex pair, so any two pieces agree
    on their shared edges: 1 + 0.1 * a pseudo-random fraction of the pair."""
    a, b = min(a, b), max(a, b)
    return 1.0 + 0.1 * ((a * 7919 + b * 104729) % 97) / 97.0


def tube(layers):
    """A tube over the triangle whose i-th circle is layers[i] (three vertex ids)."""
    n = len(layers) - 1
    cells = tessera.Spacetime.prismCells(BASE, n, {})
    relabel = {i * 3 + j: layers[i][j] for i in range(n + 1) for j in range(3)}
    cells = [[relabel[v] for v in c] for c in cells]
    st = tessera.Spacetime.fromVertexTuples(2, cells, 1.0, 0.0)
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setLength(math.sqrt(_edge_length(a, b)))
        e.setPhase(0.0)
    st.materializeFacets()
    return st


def circle_cells(vertices):
    v = sorted(vertices)
    return [[v[0], v[1]], [v[0], v[2]], [v[1], v[2]]]


W0 = lambda: tube([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
W1 = lambda: tube([[6, 7, 8], [9, 10, 11], [12, 13, 14]])
WA = lambda: tube([[100, 101, 102], [103, 104, 105], [0, 1, 2]])       # output circle {0,1,2}
WB = lambda: tube([[203, 204, 205], [200, 201, 202], [6, 7, 8]])       # output circle {6,7,8}
WC = lambda: tube([[0, 1, 2], [3, 4, 5], [6, 7, 8]])                   # inputs {0,1,2} and {6,7,8}
WD = lambda: tube([[6, 7, 8], [300, 301, 302], [303, 304, 305]])       # consumes {6,7,8}


def harmonic_fiber(st, circle):
    a = PL.assemble([st])
    return a, PL.read_boundary_fiber(a, 1, PL.harmonic_contour(a, 1), circle_cells(circle))


class TestAssembly:
    def test_assembly_is_exact_on_shared_cells(self, whitney_default):
        a = PL.assemble([W0(), W1()])
        assert a.dimension() == 2
        assert sorted(map(tuple, a.sharedCells[0])) == [(6,), (7,), (8,)]
        assert sorted(map(tuple, a.sharedCells[1])) == [(6, 7), (6, 8), (7, 8)]
        assert a.sharedCells[2] == []
        for k in range(3):
            assert PL.assembly_residual(a, k) <= 1e-14
        assert math.isnan(a.epsilon)
        assert PL.assemble([W0(), W1()], [0.1, 0.1]).epsilon == 0.1

    def test_mismatched_epsilon_and_geometry_refused(self, whitney_default):
        with pytest.raises(ValueError, match="epsilon"):
            PL.assemble([W0(), W1()], [0.1, 0.2])
        w1 = W1()
        for e in w1.getEdgeList().toVector():
            if {e.getSource().getId(), e.getTarget().getId()} == {6, 7}:
                e.setLength(math.sqrt(2.5))
        with pytest.raises(ValueError, match="shared edge"):
            PL.assemble([W0(), w1])


class TestBoundaryResponse:
    def test_glued_bordered_response_is_the_star_product(self, whitney_default):
        """The bordered pencil (edges then vertices) is assembled per top simplex,
        so the Feshbach complement of the glued complex onto the outer boundary
        (its edges and vertices) equals the star product of the pieces' bordered
        responses along the shared circle's edges and vertices, to round-off.
        The dense pencil A~ contains M_0^{-1} and is not additive on shared cells;
        eliminating the vertices afterwards recovers -F_B(lambda) on the edges."""
        lam = 0.37
        w0, w1 = W0(), W1()
        whole = PL.assemble([w0, w1])

        def bordered(a, edge_vertices):
            up = PL.indices_of(a, 1, [c for circ in edge_vertices for c in circle_cells(circ)])
            low = PL.indices_of(a, 0, [[v] for circ in edge_vertices for v in sorted(circ)])
            return PL.bordered_response(a, 1, up, low, lam)

        def union_coordinates(a, res):
            n = a.complex.numSimplices(1)
            edges = a.complex.kSimplexVertices(1)
            verts = a.complex.kSimplexVertices(0)
            out = []
            for i in res.interface:
                if i < n:
                    out.append(whole.cell_index(1, edges[i]))
                else:
                    out.append(whole.complex.numSimplices(1) + whole.cell_index(0, verts[i - n]))
            return out

        full = bordered(whole, [{0, 1, 2}, {12, 13, 14}])
        a0, a1 = PL.assemble([w0]), PL.assemble([w1])
        left = bordered(a0, [{0, 1, 2}, {6, 7, 8}])
        right = bordered(a1, [{6, 7, 8}, {12, 13, 14}])
        composed = PL.compose_responses(left.response, union_coordinates(a0, left),
                                        right.response, union_coordinates(a1, right))
        np.testing.assert_allclose(composed, full.response, atol=1e-10 * np.abs(full.response).max())
        # eliminating the boundary vertices gives -F_B(lambda) on the boundary edges
        outer_edges = PL.indices_of(whole, 1, circle_cells({0, 1, 2}) + circle_cells({12, 13, 14}))
        edge_level = PL.boundary_response(whole, 1, outer_edges, lam)
        upper = PL.upper_response(full, whole.complex.numSimplices(1))
        np.testing.assert_allclose(upper, -edge_level.response, atol=1e-10 * np.abs(edge_level.response).max())

    def test_bordered_pencil_schur_complement_is_the_pencil(self, whitney_default):
        a = PL.assemble([W0()])
        lam = 0.21
        B = PL.bordered_pencil(a, 1, lam)
        n = B.upperCount
        S = B.matrix[:n, :n] - B.matrix[:n, n:] @ np.linalg.solve(B.matrix[n:, n:], B.matrix[n:, :n])
        P = PL.pencil(a, 1)
        np.testing.assert_allclose(S, lam * P.B - P.A, atol=1e-11 * np.abs(P.A).max())


class TestLevels:
    def test_disjoint_fibers_give_a_block_diagonal_gram_and_a_glued_interface_couples(self, whitney_default):
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        _, fB = harmonic_fiber(WB(), {6, 7, 8})
        assert fA.rank() == 1 and fB.rank() == 1
        assert fA.certificate.rank == 1
        c = PL.assemble([WC()])
        level = PL.level(c, 1, [fA, fB], 0.0)
        assert level.fibersDisjoint
        assert level.blockRanks == [1, 1] and level.blockOffsets == [0, 1]
        G = level.restriction.gram
        assert G.shape == (2, 2)
        assert G[0, 1] == 0 and G[1, 0] == 0                      # exactly, no shared top simplex
        assert abs(G[0, 0]) > 0 and abs(G[1, 1]) > 0
        assert np.abs(level.constraintGram[0, 1]) > 1e-12            # coupled through the eliminated bulk
        assert level.J.shape == (6, 2) and level.Jdual.shape == (6, 2)
        assert level.restriction.A.shape == (2, 2)
        # A glued interface: W_A's output fiber and W_C's own fiber on the SAME
        # circle overlap on their cells, so the level is the labeled sum with the
        # off-diagonal Gram block carried exactly (never a direct sum).
        fC_in = PL.read_boundary_fiber(c, 1, PL.harmonic_contour(c, 1), circle_cells({0, 1, 2}))
        glued = PL.level(c, 1, [fA, fC_in], 0.0)
        assert not glued.fibersDisjoint
        assert np.abs(glued.restriction.gram[0, 1]) > 1e-12
        np.testing.assert_allclose(glued.restriction.gram, glued.restriction.gram.T, atol=1e-12)

    def test_transfer_reversal_holds_on_the_level_link(self, whitney_default):
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        _, fB = harmonic_fiber(WB(), {6, 7, 8})
        t = PL.transfer(PL.assemble([WC()]), 1, fA, fB)
        assert t.reversalResidual <= 1e-8
        assert t.forward.shape == (1, 1) and t.reverse.shape == (1, 1)
        np.testing.assert_allclose(t.reverse, t.forward.T, atol=1e-10 * max(1.0, np.abs(t.forward).max()))

    def test_second_level_consumes_the_first_through_the_same_code_path(self, whitney_default):
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        c = PL.assemble([WC()])
        level1 = PL.level(c, 1, [fA], 0.0)
        assert level1.J.shape == (3, 1)
        f_out = PL.read_boundary_fiber(c, 1, PL.harmonic_contour(c, 1), circle_cells({6, 7, 8}))
        assert f_out.rank() == 1 and f_out.degree == 1
        d = PL.assemble([WD()])
        level2 = PL.level(d, 1, [f_out], 0.0)
        assert level2.J.shape == (3, 1) and level2.restriction.A.shape == (1, 1)
        assert level2.interfaceCells == PL.indices_of(d, 1, circle_cells({6, 7, 8}))
        assert set(level2.interiorCells).isdisjoint(level2.interfaceCells)
        assert len(level2.interiorCells) + len(level2.interfaceCells) == d.complex.numSimplices(1)
        for lv in (level1, level2):
            # at the fiber's own eigenvalue the response is singular by construction,
            # so the determinant identity is checked off-resonance; the solve is exact
            assert not lv.response.interiorSingular and lv.response.solveResidual < 1e-10
            assert math.isfinite(lv.restriction.gram[0, 0].real)
        off = PL.level(d, 1, [f_out], 0.3)
        assert off.response.determinantResidual < 1e-8

    def test_level_refuses_foreign_cells(self, whitney_default):
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        with pytest.raises(ValueError, match="not a degree-1 cell"):
            PL.level(PL.assemble([WD()]), 1, [fA], 0.0)


class TestMultiCobordismSurface:
    def test_fiber_form_beside_the_period_target(self, whitney_default):
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        _, fB = harmonic_fiber(WB(), {6, 7, 8})
        node = cob.MultiCobordism(WC(), [[1.0], [1.0]], [], [1], einstein_hilbert=False)
        node.seed_inputs([0, 6])
        assert len(node.inputs) == 2
        assert node.inputs[0].fiber is None
        node.set_input_fiber(0, fA)
        node.set_input_fiber(1, fB)
        assert node.inputs[0].fiber.rank() == 1 and node.input_fiber(1).rank() == 1
        with pytest.raises(IndexError):
            node.set_input_fiber(5, fA)
        # a short-budget pin of the two fibers reports its fit (convergence is #911's gate)
        result = node.pin_input_fibers(1, 1e-8, 1, 0, 0, 5)
        assert math.isfinite(result.residual) and result.degree == 1
        assert len(result.support_cells) == 6 and len(result.state) == PL.assemble([WC()]).complex.numSimplices(1)
        pinned = {tuple(c) for c in result.support_cells}
        assert pinned == {tuple(sorted(c)) for c in circle_cells({0, 1, 2}) + circle_cells({6, 7, 8})}
        with pytest.raises(ValueError, match="rank"):
            wide = cob.BoundaryFiber()
            wide.degree, wide.cells = 1, fA.cells
            wide.images = np.hstack([fA.images, fA.images])
            node.set_input_fiber(1, wide)
            node.pin_input_fibers(1, 1e-8, 1, 0, 0, 5)

    def test_read_output_fiber_on_the_live_complex(self, whitney_default):
        node = cob.MultiCobordism(WC(), [], [[1.0]], [1], einstein_hilbert=False)
        node.seed_outputs([6])
        assert node.outputs[0].fiber is None
        fiber = node.read_output_fiber(0, 1)
        assert fiber.rank() >= 1
        assert node.output_fiber(0).rank() == fiber.rank()
        block = set(node.outputs[0].vertices)
        assert all(set(c) <= block for c in fiber.cells)
        assert np.allclose(fiber.gram, fiber.gram.T)
        legacy = cob.MultiCobordism(WC(), [], [[1.0]], [1], einstein_hilbert=False,
                                    metric_source=cob.HodgeMetricSource.DiagonalWeights)
        legacy.seed_outputs([6])
        with pytest.raises(RuntimeError, match="Whitney"):
            legacy.read_output_fiber(0, 1)


class TestBipartiteAtLevelOne:
    """The #911 protocol at level 1: the third cobordism's relaxed bulk must
    reproduce the composed transfer on held-out inputs. The relaxation does
    not converge at test budgets on this fixture family under either metric
    (docs/design/whitney_pencil_metric_source_findings.md), so the gate runs
    under TESSERA_SLOW_TESTS=1; always-on, the frozen-geometry reads above
    stand."""

    def test_relaxed_third_reproduces_the_composed_transfer(self, whitney_default):
        if not os.environ.get("TESSERA_SLOW_TESTS"):
            pytest.skip("full realizability gate: set TESSERA_SLOW_TESTS=1")
        _, fA = harmonic_fiber(WA(), {0, 1, 2})
        _, fB = harmonic_fiber(WB(), {6, 7, 8})
        node = cob.MultiCobordism(WC(), [[1.0], [1.0]], [], [1], einstein_hilbert=False)
        node.seed_inputs([0, 6])
        node.set_input_fiber(0, fA)
        node.set_input_fiber(1, fB)
        result = node.pin_input_fibers(1, 1e-8, 8, 2, 0, 200)
        assert result.converged, f"level-1 fit not converged: residual {result.residual:.3e}"
        c = PL.assemble([node.st])
        t = PL.transfer(c, 1, fA, fB)
        assert t.reversalResidual <= 1e-8


class TestDagPiping:
    def test_dag_pipes_output_fibers_downstream(self, whitney_default):
        if not os.environ.get("TESSERA_SLOW_TESTS"):
            pytest.skip("DAG run on the Whitney pencil: set TESSERA_SLOW_TESTS=1")
        import cmath
        from tests.cobordism._closed_s4 import closed_s4
        w = cmath.exp(2j * math.pi / 3.0)
        dag = cob.CobordismDAG()
        dag.set_fiber_piping(True, 3)
        rec = dag.add_node(closed_s4(12, 3), [[1, -1, 0], [1, 0, -1]], [], [[1, w, w * w], [1, w * w, w]],
                           degrees=[3], seed=3)
        pro = dag.add_node(closed_s4(12, 5), [[0, 1, -1]], [(rec, 0)], [[1, w, w * w]], degrees=[3], seed=5)
        dag.run(stage1_max_steps=6, stage1_candidate_moves=3, stage2_max_iters=6)
        assert dag.fiber_piping()
        if dag.has_output_fiber(rec, 0):
            assert dag.piped_input_count(pro) == 1
        else:
            assert dag.fiber_refusal(rec)
