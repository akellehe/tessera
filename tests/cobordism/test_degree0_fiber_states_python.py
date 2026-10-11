# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Degree-0 fiber states on a 3-simplex seed (#940, epic #938).

A node on a single Δ³ carries a four-term state as a rank-1 band of the
covariant degree-0 pencil read on the seed's four vertices. The block's target
is a fiber (cells + images); with `use_fiber_residuals` the block is scored by
the least-squares leak of the target in the band read on the block's own
pencil, folded into r_U so stage 1 and stage 2 descend it through the existing
paths. Coefficients come from lengths and connection values only.

The flat zero mode (band 0 at trivial holonomy) has the constant image and
carries no state: its residual against a non-constant target is fixed at
1 − |⟨1, ψ⟩|²/(4‖ψ‖²) and no geometry moves it; `flat_zero_mode_overlap` is 1
there. The default band is the lowest band above it.
"""
import math
import os

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
_FULL = bool(os.environ.get("TESSERA_SLOW_TESTS"))
CELLS = [[0], [1], [2], [3]]


@pytest.fixture
def whitney_default():
    previous = HL.default_metric_source()
    HL.set_default_metric_source(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.set_default_metric_source(previous)


def fiber_target(psi, contour=None):
    fiber = cob.BoundaryFiber()
    fiber.degree = 0
    fiber.cells = CELLS
    fiber.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    if contour is not None:
        fiber.contour = contour
    return fiber


def node_with_state(psi, contour=None, seed=0):
    """An input node: a single Δ³ whose WHOLE complex must carry the state on the
    seed's four vertices; no blocks, no period targets."""
    node = MC(MC.seed_simplex(3), [], [], degrees=[0], seed=seed, einstein_hilbert=False)
    node.set_whole_complex_fiber_target(fiber_target(psi, contour))
    node.use_fiber_residuals(True)
    return node


def python_residual(st, psi, band_index=1, contour=None):
    assembled = cob.PencilLayer.assemble([st])
    chosen = contour if contour is not None else cob.PencilLayer.band_contour(assembled, 0, band_index)
    read = cob.PencilLayer.read_boundary_fiber(assembled, 0, chosen, CELLS)
    Z = np.asarray(read.images)
    psi = np.asarray(psi, dtype=complex).reshape(4, 1)
    c = np.linalg.lstsq(Z, psi, rcond=None)[0]
    return float(np.linalg.norm(Z @ c - psi) ** 2 / np.linalg.norm(psi) ** 2), Z


class TestSeedSimplex:
    def test_seed_simplex_is_one_lorentzian_tetrahedron(self):
        st = MC.seed_simplex(3)
        assert all(len(f) == 3 for f in st.get_boundary())  # triangles bound a 3-simplex
        assert st.get_vertex_list().size() == 4 and st.get_edge_list().size() == 6
        assert len(st.get_boundary()) == 4
        for e in st.get_edge_list().to_vector():
            assert abs(complex(e.get_length()) ** 2) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="dimension must be at least one"):
            MC.seed_simplex(0)


class TestFiberResidual:
    def test_whole_complex_residual_matches_the_pencil_read_and_is_all_of_r_u(self, whitney_default):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], einstein_hilbert=False)
        assert node.r_u(node.spacetime()) == 0.0
        node.set_whole_complex_fiber_target(fiber_target(psi))
        assert not node.uses_fiber_residuals()
        assert node.r_u(node.spacetime()) == 0.0  # untouched while off
        node.use_fiber_residuals(True)
        expected, Z = python_residual(node.spacetime(), psi)
        assert Z.shape == (4, 3)  # the uniform seed's band above zero is the triple at 40
        measured = node.whole_complex_fiber_residual()
        assert measured == pytest.approx(expected, rel=1e-10, abs=1e-14)
        # on one tetrahedron the band above zero is {z : sum z = 0} whatever the lengths
        assert measured == pytest.approx(abs(psi.sum()) ** 2 / (4 * np.linalg.norm(psi) ** 2), rel=1e-10)
        assert node.r_u(node.spacetime()) == pytest.approx(measured, rel=1e-12)  # no near-kernel term
        read = node.read_whole_complex_fiber()
        assert np.asarray(read.images).shape == (4, 3) and read.degree == 0

    def test_block_form_reads_the_block_subcomplex(self, whitney_default):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0], einstein_hilbert=False)
        node.seed_inputs([0])
        node.set_input_fiber(0, fiber_target(psi))
        node.use_fiber_residuals(True)
        expected, _ = python_residual(node.spacetime(), psi)
        assert node.fiber_residual_for_input_block(0) == pytest.approx(expected, rel=1e-10, abs=1e-14)
        assert node.r_u(node.spacetime()) == pytest.approx(expected, rel=1e-12)
        with pytest.raises(IndexError):
            node.fiber_residual_for_input_block(1)

    def test_refusals(self, whitney_default):
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], einstein_hilbert=False)
        with pytest.raises(RuntimeError, match="no whole-complex fiber target"):
            node.whole_complex_fiber_residual()
        empty = cob.BoundaryFiber()
        with pytest.raises(ValueError, match="no images"):
            node.set_whole_complex_fiber_target(empty)
        diagonal = MC(MC.seed_simplex(3), [], [], degrees=[0], einstein_hilbert=False,
                      metric_source=cob.HodgeMetricSource.DiagonalWeights)
        diagonal.set_whole_complex_fiber_target(fiber_target([1, 0, 0, 0]))
        diagonal.use_fiber_residuals(True)
        with pytest.raises(RuntimeError, match="Whitney pencil"):
            diagonal.whole_complex_fiber_residual()

    def test_flat_band_carries_no_state(self, whitney_default):
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        st = MC.seed_simplex(3)
        flat = cob.PencilLayer.band_contour(cob.PencilLayer.assemble([st]), 0, 0)
        node = node_with_state(psi, contour=flat)
        assert node.whole_complex_fiber_residual() == pytest.approx(0.75, abs=1e-12)  # 1 - |<1, e0>|^2 / 4
        # no choice of LENGTHS moves it: the flat band's image is the constant
        # vector on a connected complex at trivial holonomy
        rng = np.random.default_rng(5)
        for e in node.spacetime().get_edge_list().to_vector():
            e.set_length(np.sqrt(complex(1.0 + 0.3 * rng.uniform(-1, 1), 0.2 * rng.uniform(-1, 1))))
        assert node.whole_complex_fiber_residual() == pytest.approx(0.75, abs=1e-12)
        # a link phase (flux) lifts the zero mode and the band inside the same
        # contour acquires content: stage 2's analytic phase gradient (#947) is
        # what descends it
        node.spacetime().get_edge_list().to_vector()[0].set_phase(0.3 + 0j)
        assert node.whole_complex_fiber_residual() != pytest.approx(0.75, abs=1e-6)
        node.spacetime().get_edge_list().to_vector()[0].set_phase(0j)  # back to trivial holonomy
        read = node.read_whole_complex_fiber()
        assembled = cob.PencilLayer.assemble([node.spacetime()])
        K = assembled.complex
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, assembled.lengths), ch.Connection(K, [1.0 + 0j] * 6))
        assert ch.FaceAnchor.flat_zero_mode_overlap(cov, np.asarray(read.dual_images), np.asarray(read.images)) == pytest.approx(1.0, abs=1e-10)

    def test_band_contour_indexing(self, whitney_default):
        st = MC.seed_simplex(3)
        assembled = cob.PencilLayer.assemble([st])
        ev = sorted(np.abs(np.asarray(assembled.op.spectrum(0).eigenvalues)))
        c0 = cob.PencilLayer.band_contour(assembled, 0, 0)
        assert abs(np.mean(np.asarray(c0.nodes))) < 1e-9  # centered on zero
        assert ev[0] < 1e-9
        with pytest.raises(ValueError, match="exceeds"):
            cob.PencilLayer.band_contour(assembled, 0, 10)


class TestDrive:
    """Measured on the uniform Δ³ seed. The bare seed is nearly stationary
    (its band above zero is the triple at 40, {z : Σz = 0} for equal lengths;
    stage 2 splits it and moves 0.1169 → 0.1167 in 60 iterations). With growth
    room from `precone` the geometry descends: two cone-ins 0.74 → 0.041, eight
    cone-ins 0.99 → 0.011 → 0.0072 over two stage-2 passes. Full drives
    (precone 8, six rounds of stage 1 with 8 candidates then stage 2 with 200
    iterations) reach the state or not depending on the draw: seed 0 reaches
    2.3e-31 in round 2 (14 vertices, 13 s); seed 3 reaches 8e-13 in round 0 but
    takes 740 s because every one of the 200 iterations descends, at ~3.5 s per
    numerical-gradient iteration; seeds 1 and 2 end at 0.13 and 0.02 after six
    rounds. Stage 1 accepts moves on some draws and none on others. The cost
    is the finite-difference register-residual path re-reading the band per
    edge; the analytic Riesz-projector gradient is the follow-up."""

    def test_geometry_descends_the_residual_with_growth_room(self, whitney_default):
        rng = np.random.default_rng(1)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], precone=2, einstein_hilbert=False)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        assert node.spacetime().get_vertex_list().size() == 6
        before = node.whole_complex_fiber_residual()
        node.run_stage2(beta=1.0, max_iters=60, tolerance=1e-15)
        after = node.whole_complex_fiber_residual()
        assert after < 0.25 * before, f"stage 2 did not descend the fiber residual: {before:.3e} -> {after:.3e}"
        # the piped form reads the same band the residual scored
        read = node.read_whole_complex_fiber()
        Z = np.asarray(read.images)
        c = np.linalg.lstsq(Z, psi.reshape(4, 1), rcond=None)[0]
        assert float(np.linalg.norm(Z @ c - psi.reshape(4, 1)) ** 2 / np.linalg.norm(psi) ** 2) == pytest.approx(after, rel=1e-9)

    def test_stage1_reports(self, whitney_default):
        rng = np.random.default_rng(2)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = node_with_state(psi)
        node.run_stage1(max_steps=3, n_candidate_moves=4)
        r = node.whole_complex_fiber_residual()
        assert math.isfinite(r) and node.spacetime().get_vertex_list().size() >= 4

    def test_state_is_approached(self, whitney_default):
        if not _FULL:
            pytest.skip("full gate: set TESSERA_SLOW_TESTS=1")
        rng = np.random.default_rng(3)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], precone=8, einstein_hilbert=False)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        residuals = [node.whole_complex_fiber_residual()]
        for _ in range(3):
            node.run_stage2(beta=1.0, max_iters=100, tolerance=1e-15)
            residuals.append(node.whole_complex_fiber_residual())
        assert min(residuals) < 1e-2, f"fiber residual trace {['%.2e' % r for r in residuals]}"

    def test_full_drive_descends_by_an_order_of_magnitude(self, whitney_default):
        """Seed 0 with growth, single-threaded: 0.63 → 0.31 → 0.25 → 2.3e-31
        (rounds 0–2). The draw is not process-deterministic (candidate
        scoring runs in OpenMP threads; the same seed with two threads gave
        0.63 → 0.31 → 0.23 → 0.055), so the gate asserts what every recorded
        draw did: an order of magnitude within four rounds."""
        if not _FULL:
            pytest.skip("full gate: set TESSERA_SLOW_TESTS=1")
        rng = np.random.default_rng(3)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], seed=0, precone=8, einstein_hilbert=False)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        residuals = [node.whole_complex_fiber_residual()]
        for _ in range(4):
            node.run_stage1(max_steps=4, n_candidate_moves=8)
            node.run_stage2(beta=1.0, max_iters=200, tolerance=1e-15)
            residuals.append(node.whole_complex_fiber_residual())
            if residuals[-1] < 1e-12:
                break
        assert min(residuals) < 0.1 * residuals[0], f"fiber residual trace {['%.2e' % r for r in residuals]}"


class TestDagFlag:
    def test_score_blocks_by_fiber_flag(self):
        dag = cob.CobordismDAG()
        assert not dag.scores_blocks_by_fiber()
        dag.set_fiber_piping(True, 0, True)
        assert dag.fiber_piping() and dag.scores_blocks_by_fiber()
