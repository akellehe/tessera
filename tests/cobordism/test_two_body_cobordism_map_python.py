# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The two-body cobordism map (#941, epic #938).

An interaction node W on a single Δ³ seed with growth room carries two piped
input fibers attached to two vertex-disjoint tetrahedra of its own complex
(the attachment order is the attachment permutation). The bulk between them
is read as the transfer between the full frames on the two attached cell
sets, T_AB = (Z_A^∨)ᵀ(Ãᵁ)_AB Z_B with unit images: the coupling block of the
whole between the two frames. `choi_decomposed` selects the reading reported
(vec(T_AB) as a state on the pair space, or the operator T_AB); the fit
residual is the projective Frobenius leak of χ in either reading. The test
system is the spin-3/2 XY flip-flop: χ = (Dψ)(Uφ)ᵀ + (Uψ)(Dφ)ᵀ.
"""
import itertools
import math
import os

import numpy as np
import pytest

from tessera import cobordism as cob

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
_FULL = bool(os.environ.get("TESSERA_SLOW_TESTS"))
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


def lowering():
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]  # Σ⁻|k⟩ = λ_k |k+1⟩
    return D


def flip_flop(psi, phi):
    D = lowering()
    U = D.T
    return np.outer(D @ psi, U @ phi) + np.outer(U @ psi, D @ phi)


def fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = [[0], [1], [2], [3]]  # upstream ids; replaced on attachment
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def disjoint_tetrahedra(st):
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(st).kSimplexVertices(3)]
    for a, b in itertools.combinations(tets, 2):
        if not set(a) & set(b):
            return a, b
    raise AssertionError("no two vertex-disjoint tetrahedra")


def interaction_node(psi, phi, precone=8, seed=0, choi=True):
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=seed, precone=precone, einstein_hilbert=False)
    st = node.spacetime()
    node.seed_inputs([0, 1])
    a, b = disjoint_tetrahedra(st)
    node.attach_input_fiber(0, fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, fiber(phi), [[v] for v in b])
    node.set_two_body_target(flip_flop(psi, phi), choi)
    node.use_fiber_residuals(True)
    return node, a, b


def python_transfer(st, a, b):
    assembled = cob.PencilLayer.assemble([st])
    fa, fb = cob.BoundaryFiber(), cob.BoundaryFiber()
    for f, cells in ((fa, a), (fb, b)):
        f.degree = 0
        f.cells = [[v] for v in cells]
        f.images = np.eye(4, dtype=complex)
        f.dualImages = np.eye(4, dtype=complex)
    return cob.PencilLayer.transfer(assembled, 0, fa, fb)


class TestAlgebra:
    def test_flip_flop_matches_the_stated_matrix(self):
        a, b, c, d, e, f, g, h = (np.random.default_rng(0).normal(size=8) + 0j)
        chi = flip_flop([a, b, c, d], [e, f, g, h])
        s3 = math.sqrt(3.0)
        expected = np.array([[0, 3 * b * e, 2 * s3 * b * f, 3 * b * g],
                             [3 * a * f, 2 * s3 * (a * g + c * e), 3 * a * h + 4 * c * f, 2 * s3 * c * g],
                             [2 * s3 * b * f, 4 * b * g + 3 * d * e, 2 * s3 * (b * h + d * f), 3 * d * g],
                             [3 * c * f, 2 * s3 * c * g, 3 * c * h, 0]])
        np.testing.assert_allclose(chi, expected, atol=1e-12)
        # generic inputs give Schmidt rank two (the two-body, non-quasi-free signature)
        assert np.linalg.matrix_rank(chi, tol=1e-10) == 2
        # total lowering number K + M = k + m: only such entries are populated
        chi01 = flip_flop([1, 0, 0, 0], [0, 1, 0, 0])
        assert set(zip(*np.nonzero(np.abs(chi01) > 1e-12))) == {(1, 0)}


class TestAttachmentAndResidual:
    def test_residual_equals_the_python_transfer_leak_in_both_readings(self, whitney_default):
        rng = np.random.default_rng(1)
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        for choi in (True, False):
            node, a, b = interaction_node(psi, phi, choi=choi)
            T = np.asarray(python_transfer(node.spacetime(), a, b).forward)
            chi = flip_flop(psi, phi)
            overlap = np.vdot(T, chi)
            expected = 1.0 - abs(overlap) ** 2 / (np.linalg.norm(T) ** 2 * np.linalg.norm(chi) ** 2)
            assert node.two_body_residual() == pytest.approx(expected, rel=1e-10, abs=1e-14)
            read = node.read_two_body()
            assert read.choi_decomposed is choi
            np.testing.assert_allclose(np.asarray(read.transfer), T, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.asarray(read.choi_state), T.flatten(order="F"), rtol=1e-12, atol=1e-14)
            assert read.residual == pytest.approx(expected, rel=1e-10, abs=1e-14)
            assert len(read.input_fiber_residuals) == 2 and read.schmidt_rank >= 1
            assert read.reversal_residual < 1e-8
            assert [tuple(c) for c in read.cells_a] == [(v,) for v in a]
            # r_U = the two block fiber residuals + the two-body residual (weight 1)
            assert node.r_u(node.spacetime()) == pytest.approx(sum(read.input_fiber_residuals) + expected, rel=1e-10)

    def test_attachment_permutation_permutes_the_transfer(self, whitney_default):
        rng = np.random.default_rng(2)
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        node, a, b = interaction_node(psi, phi)
        T = np.asarray(node.read_two_body().transfer)
        perm = [2, 0, 3, 1]
        node.attach_input_fiber(0, fiber(psi), [[a[p]] for p in perm])
        T2 = np.asarray(node.read_two_body().transfer)
        np.testing.assert_allclose(T2, T[perm, :], rtol=1e-12, atol=1e-14)

    def test_refusals(self, whitney_default):
        rng = np.random.default_rng(3)
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
                  precone=8, einstein_hilbert=False)
        node.seed_inputs([0, 1])
        a, b = disjoint_tetrahedra(node.spacetime())
        with pytest.raises(RuntimeError,
                           match="two attached input fibers, or four"):
            node.read_two_body()
        with pytest.raises(ValueError, match="one attachment cell per fiber row"):
            node.attach_input_fiber(0, fiber(psi), [[a[0]], [a[1]]])
        with pytest.raises(ValueError, match="absent from the live complex"):
            node.attach_input_fiber(0, fiber(psi), [[999], [a[1]], [a[2]], [a[3]]])
        node.attach_input_fiber(0, fiber(psi), [[v] for v in a])
        with pytest.raises(ValueError, match="overlaps input fiber 0"):
            node.attach_input_fiber(1, fiber(phi), [[v] for v in a])
        node.attach_input_fiber(1, fiber(phi), [[v] for v in b])
        with pytest.raises(RuntimeError, match="no two-body target"):
            node.two_body_residual()
        with pytest.raises(ValueError, match="nonzero matrix"):
            node.set_two_body_target(np.zeros((4, 4), dtype=complex))
        # with two fibers attached the target's shape is checked against the
        # transfer's at set time (the cell counts here; the frames' ranks once
        # both blocks carry a frame, test_period_frame_transfer_python.py)
        with pytest.raises(ValueError, match="attached cells give 4x4"):
            node.set_two_body_target(np.eye(3, dtype=complex))
        node.set_two_body_target(np.eye(4, dtype=complex))
        node.use_fiber_residuals(True)
        assert 0.0 <= node.two_body_residual() <= 1.0


class TestDrive:
    def test_stage2_descends_the_two_body_residual(self, whitney_default):
        rng = np.random.default_rng(4)
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        node, a, b = interaction_node(psi, phi)
        before = node.two_body_residual()
        node.run_stage2(beta=1.0, max_iters=12, tolerance=1e-15)  # ~3 s per FD iteration at this size
        after = node.two_body_residual()
        assert after < before, f"two-body residual did not descend: {before:.3e} -> {after:.3e}"

    def test_full_drive_reports(self, whitney_default):
        if not _FULL:
            pytest.skip("full gate: set TESSERA_SLOW_TESTS=1")
        rng = np.random.default_rng(5)
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        node, a, b = interaction_node(psi, phi, precone=8)
        trace = [node.read_two_body().residual]
        for _ in range(4):
            node.run_stage1(max_steps=4, n_candidate_moves=8)
            node.run_stage2(beta=1.0, max_iters=200, tolerance=1e-15)
            trace.append(node.read_two_body().residual)
        assert min(trace) < trace[0], f"two-body trace {['%.2e' % r for r in trace]}"


class TestDag:
    def test_attachment_and_target_plumbing(self, whitney_default):
        dag = cob.CobordismDAG()
        n = dag.add_node(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j]], [], [], [0], 1.0, 0)
        dag.set_fiber_piping(True, 0, True)
        dag.set_input_attachment(n, 0, [[0], [1], [2], [3]])
        dag.set_two_body_target(n, np.eye(4, dtype=complex), False)
        assert not dag.has_two_body_read(n)
        with pytest.raises(RuntimeError, match="no two-body reading"):
            dag.two_body_read(n)
        with pytest.raises(IndexError):
            dag.set_two_body_target(5, np.eye(4, dtype=complex))
