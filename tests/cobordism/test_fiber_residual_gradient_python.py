# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Analytic gradients of the fiber-mode residuals (#947, epic #938).

Both residuals are projective, so they are invariant under a common scaling
of every squared length: with the holomorphic sensitivity dF_e recovered from
the packed gradient g_e = (2 Re dF_e, −2 Im dF_e), the Euler identity is
Σ_e s_e dF_e = 0 exactly (round-off). That is the validation; a central
difference is reported beside it as a sanity check labeled at its own
accuracy, never as the reference. Stage 2 descends the fiber residual through
the analytic ascent when fiber residuals are on.
"""
import itertools
import math
import os
import time

import numpy as np
import pytest

from tessera import cobordism as cob

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
_FULL = bool(os.environ.get("TESSERA_SLOW_TESTS"))
CELLS = [[0], [1], [2], [3]]
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])


@pytest.fixture
def whitney_default():
    previous = HL.default_metric_source()
    HL.set_default_metric_source(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.set_default_metric_source(previous)


def fiber_target(psi, cells=CELLS):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = cells
    f.images = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return f


def jitter(node, rng, scale=0.25):
    """Generic complex squared lengths so nothing sits on a symmetric point."""
    for e in node.spacetime().get_edge_list().to_vector():
        s = 1.0 + scale * rng.uniform(-1, 1) + 1j * scale * rng.uniform(-1, 1)
        e.set_length(np.sqrt(complex(s)))


def squared_lengths(node):
    return np.array([complex(e.get_length()) ** 2 for e in node.spacetime().get_edge_list().to_vector()])


def holomorphic(packed):
    """dF from the packed (2 Re dF, −2 Im dF)."""
    packed = np.asarray(packed)
    return 0.5 * (packed.real - 1j * packed.imag)


def flip_flop(psi, phi):
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return np.outer(D @ psi, D.T @ phi) + np.outer(D.T @ psi, D @ phi)


def central_difference(node, evaluate, h=1e-6):
    edges = node.spacetime().get_edge_list().to_vector()
    out = np.zeros(len(edges), dtype=complex)
    for i, e in enumerate(edges):
        l0 = complex(e.get_length())
        s0 = l0 * l0
        parts = []
        for step in (h, 1j * h):
            e.set_length(np.sqrt(s0 + step))
            plus = evaluate()
            e.set_length(np.sqrt(s0 - step))
            minus = evaluate()
            parts.append((plus - minus) / (2 * h))
        e.set_length(l0)
        out[i] = complex(parts[0], parts[1])
    return out


class TestWholeComplexFiberGradient:
    def test_euler_identity_and_finite_difference_sanity(self, whitney_default):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], precone=4, einstein_hilbert=False)
        jitter(node, rng)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        lengths, phases = node.fiber_residual_gradient(node.whole_complex_fiber_target())
        lengths = np.asarray(lengths)
        assert lengths.shape == (node.spacetime().get_edge_list().size(),)
        s = squared_lengths(node)
        euler = abs(np.sum(s * holomorphic(lengths))) / max(np.abs(lengths).max() * np.abs(s).max(), 1e-300)
        assert euler < 1e-10, f"Euler identity violated: {euler:.3e}"
        fd = central_difference(node, node.whole_complex_fiber_residual)
        assert np.abs(fd - lengths).max() < 1e-5 * max(1.0, np.abs(lengths).max())  # FD sanity at its own accuracy
        assert np.asarray(phases).shape == lengths.shape  # degree 0 carries a phase gradient

    def test_block_and_ascent_agree_with_the_sum_of_terms(self, whitney_default):
        rng = np.random.default_rng(1)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], precone=4, einstein_hilbert=False)
        jitter(node, rng)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        lengths, phases = node.fiber_residual_gradient(node.whole_complex_fiber_target())
        total_l, total_p = node.fiber_mode_ascent()
        np.testing.assert_allclose(np.asarray(total_l), np.asarray(lengths), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(total_p), np.asarray(phases), rtol=1e-12, atol=1e-14)


class TestTwoBodyGradient:
    def _node(self, rng):
        psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
        node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
                  precone=8, einstein_hilbert=False)
        jitter(node, rng, 0.15)
        tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.from_spacetime(node.spacetime()).k_simplex_vertices(3)]
        a, b = next((x, y) for x, y in itertools.combinations(tets, 2) if not set(x) & set(y))
        node.seed_inputs([0, 1])
        node.attach_input_fiber(0, fiber_target(psi), [[v] for v in a])
        node.attach_input_fiber(1, fiber_target(phi), [[v] for v in b])
        node.set_two_body_target(flip_flop(psi, phi))
        node.use_fiber_residuals(True)
        return node

    def test_euler_identity_and_finite_difference_sanity(self, whitney_default):
        node = self._node(np.random.default_rng(2))
        lengths, phases = node.two_body_residual_gradient()
        lengths = np.asarray(lengths)
        s = squared_lengths(node)
        euler = abs(np.sum(s * holomorphic(lengths))) / max(np.abs(lengths).max() * np.abs(s).max(), 1e-300)
        assert euler < 1e-10, f"Euler identity violated: {euler:.3e}"
        fd = central_difference(node, node.two_body_residual)
        assert np.abs(fd - lengths).max() < 1e-5 * max(1.0, np.abs(lengths).max())

    def test_full_ascent_sums_blocks_and_two_body(self, whitney_default):
        node = self._node(np.random.default_rng(3))
        total_l, _ = node.fiber_mode_ascent()
        two_l, _ = node.two_body_residual_gradient()
        fd_total = central_difference(node, lambda: node.r_u(node.spacetime()))
        assert np.abs(fd_total - np.asarray(total_l)).max() < 1e-5 * max(1.0, np.abs(total_l).max())
        # the two-body part alone is not the whole ascent (the blocks contribute)
        assert np.abs(np.asarray(total_l) - np.asarray(two_l)).max() > 1e-8


class TestStage2UsesTheAnalyticAscent:
    def test_descent_and_speed(self, whitney_default):
        rng = np.random.default_rng(4)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        node = MC(MC.seed_simplex(3), [], [], degrees=[0], precone=8, einstein_hilbert=False)
        node.set_whole_complex_fiber_target(fiber_target(psi))
        node.use_fiber_residuals(True)
        before = node.whole_complex_fiber_residual()
        t0 = time.time()
        node.run_stage2(beta=1.0, max_iters=30, tolerance=1e-15)
        elapsed = time.time() - t0
        after = node.whole_complex_fiber_residual()
        assert after < before, f"analytic stage 2 did not descend: {before:.3e} -> {after:.3e}"
        assert elapsed < 60.0, f"30 analytic iterations took {elapsed:.1f} s"
